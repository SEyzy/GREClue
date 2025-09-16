from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix
import math
import torch
from torch import optim
import pytorch_lightning as pl
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils
from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.clustering_utils.priors import (
    Priors,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    init_mus_and_covs,
    compute_data_covs_hard_assignment,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.split_merge_operations import (
    update_models_parameters_split,
    split_step,
    merge_step,
    update_models_parameters_merge,
)
from src.clustering_models.clusternet_modules.models.Classifiers import MLP_Classifier, Subclustering_net
import ipdb
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj): 
        # A_hat = D^{-1/2} (A + I) D^{-1/2}
        device = X.device
        A = adj + torch.eye(adj.size(0), device=device)
        deg = A.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-6, -0.5))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt

        H = A_hat @ X
        H = F.relu(self.lin1(H))
        H = self.dropout(H)
        H = A_hat @ H
        H = self.lin2(H)
        return H  # (N, out_dim)

class CrossAttentionFuse(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out, n_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)
        self.attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim_out*2, dim_out), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(dim_out, dim_out))

    def forward(self, q_vec, kv_seq):

        Q = self.q_proj(q_vec).unsqueeze(1)      # (B,1,D)
        K = self.k_proj(kv_seq)                  # (B,L,D)
        V = self.v_proj(kv_seq)                  # (B,L,D)
        out, _ = self.attn(Q, K, V)              # (B,1,D)
        out = out.squeeze(1)                     # (B,D)
        fused = torch.cat([out, q_vec], dim=-1)  # 残差拼接
        return self.mlp(fused)                   # (B,D)

class ClusterNetModel(pl.LightningModule):
    def __init__(self, hparams, input_dim, init_k, feature_extractor=None, n_sub=2, centers=None, init_num=0):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.K = init_k
        self.n_sub = n_sub
        self.codes_dim = input_dim
        self.split_performed = False  # indicator to know whether a split was performed
        self.merge_performed = False
        self.feature_extractor = feature_extractor
        self.centers = centers
        if self.hparams.seed:
            pl.utilities.seed.seed_everything(self.hparams.seed)

        # initialize cluster_net
        self.cluster_net = MLP_Classifier(hparams, k=self.K, codes_dim=self.codes_dim)

        if not self.hparams.ignore_subclusters:
            # initialize subclustering net
            self.subclustering_net = Subclustering_net(hparams, codes_dim=self.codes_dim, k=self.K)
        else:
            self.subclustering_net = None
        self.last_key = self.K - 1  # variable to help with indexing the dict

        self.training_utils = training_utils(hparams)
        self.last_val_NMI = 0
        self.init_num = init_num
        self.prior_sigma_scale = self.hparams.prior_sigma_scale
        if self.init_num > 0 and self.hparams.prior_sigma_scale_step != 0:
            self.prior_sigma_scale = self.hparams.prior_sigma_scale / (self.init_num * self.hparams.prior_sigma_scale_step)
        self.use_priors = self.hparams.use_priors
        self.prior = Priors(hparams, K=self.K, codes_dim=self.codes_dim, prior_sigma_scale=self.prior_sigma_scale) # we will use for split and merges even if use_priors is false

        self.mus_inds_to_merge = None
        self.mus_ind_to_split = None

        self.starcoder_tok = AutoTokenizer.from_pretrained(self.hparams.starcoder_model_name, use_fast=True)
        self.starcoder = AutoModel.from_pretrained(self.hparams.starcoder_model_name)
        self.seq_hidden = self.starcoder.config.hidden_size

        if getattr(self.hparams, "freeze_starcoder", False):
            for p in self.starcoder.parameters():
                p.requires_grad = False


        self.token_embed = self.starcoder.get_input_embeddings() 
        self.node_type_emb = nn.Embedding(8, 32)  

        self.graph_encoder = SimpleGCN(in_dim=self.seq_hidden + 32 + 1, 
                                       hidden_dim=self.hparams.graph_hidden_dim,
                                       out_dim=self.hparams.graph_hidden_dim,
                                       dropout=self.hparams.dropout)


        self.fuser = CrossAttentionFuse(dim_q=self.seq_hidden,
                                        dim_kv=self.hparams.graph_hidden_dim,
                                        dim_out=self.hparams.fusion_hidden_dim,
                                        n_heads=4, dropout=self.hparams.dropout)


        self.proj = (nn.Identity() if self.hparams.fusion_hidden_dim == self.codes_dim
                     else nn.Linear(self.hparams.fusion_hidden_dim, self.codes_dim))


        self.cluster_net = MLP_Classifier(hparams, k=self.K, codes_dim=self.codes_dim)
        

    def _encode_seq_batch(self, seq_texts: List[str]):

        toks = self.starcoder_tok(seq_texts, padding=True, truncation=True,
                                max_length=self.hparams.max_seq_len, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.set_grad_enabled(not getattr(self.hparams, "freeze_starcoder", False)):
            out = self.starcoder(**toks).last_hidden_state  # (B, L, H)
        mask = toks["attention_mask"].unsqueeze(-1)        # (B, L, 1)
        seq_vec = (out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))  # mean pool
        return seq_vec  # (B, H)


    def _node_type_to_id(self, t: str) -> int:
        mapping = {"file":0, "line":1, "var":2, "test":3}
        return mapping.get(str(t).lower(), 7) 


    def _encode_graph_batch(self, graphs: List[Dict]):

        B = len(graphs)
        node_feats_list, adjs, masks = [], [], []
        max_nodes = max(max(1, len(g.get("nodes", []))) for g in graphs)

        for g in graphs:
            nodes = g.get("nodes", [])
            N = max(1, len(nodes))
            text_vecs = []
            type_ids = []
            sus_vals = []
            for nd in nodes:
                content = nd.get("content", nd.get("file", "")) if isinstance(nd, dict) else str(nd)
                toks = self.starcoder_tok(str(content), truncation=True, max_length=64, return_tensors="pt")
                input_ids = toks["input_ids"].to(self.device)         
                emb = self.token_embed(input_ids).mean(dim=1).squeeze(0) 
                text_vecs.append(emb)
                t = nd.get("type", "")
                type_ids.append(self._node_type_to_id(t))
                sus_vals.append(float(nd.get("sus", 0.0)))

            if N == 0:
                text_vecs = [torch.zeros(self.seq_hidden, device=self.device)]
                type_ids = [self._node_type_to_id("other")]
                sus_vals = [0.0]
                N = 1

            X_text = torch.stack(text_vecs, dim=0)                      
            X_type = self.node_type_emb(torch.tensor(type_ids, device=self.device))  
            X_sus  = torch.tensor(sus_vals, device=self.device).unsqueeze(-1)       
            X = torch.cat([X_text, X_type, X_sus], dim=-1)              
            edges = g.get("edges", [])
            adj = torch.zeros((N, N), device=self.device)
            for (u, v) in edges:
                u_idx = int(u) if u < N else N-1
                v_idx = int(v) if v < N else N-1
                adj[u_idx, v_idx] = 1.0
                adj[v_idx, u_idx] = 1.0 

            Z = self.graph_encoder(X, adj)        
            node_feats_list.append(Z)
            adjs.append(adj)

        Dg = node_feats_list[0].size(-1)
        graph_seq = torch.zeros((B, max_nodes, Dg), device=self.device)
        graph_mask = torch.zeros((B, max_nodes), device=self.device, dtype=torch.bool)
        graph_vec = torch.zeros((B, Dg), device=self.device)
        for i, Z in enumerate(node_feats_list):
            n = Z.size(0)
            graph_seq[i, :n] = Z
            graph_mask[i, :n] = True
            graph_vec[i] = Z.mean(dim=0)

        return graph_seq, graph_vec, graph_mask  # (B,L,Dg), (B,Dg), (B,L)
    

    def forward(self, batch):

        if isinstance(batch, (list, tuple)):
            x = batch[0]
            codes = x
        else:
            seq_vec = self._encode_seq_batch(batch["seq_texts"])         
            graph_seq, graph_vec, graph_mask = self._encode_graph_batch(batch["graphs"]) 
            fused = self.fuser(seq_vec, graph_seq)                        
            codes = self.proj(fused)                                      
        return self.cluster_net(codes)
    
        

    def on_train_epoch_start(self):
        #ipdb.set_trace()
        # get current training_stage
        self.current_training_stage = (
            "gather_codes" if self.current_epoch == 0 and not hasattr(self, "mus") else "train_cluster_net"
        )
        self.initialize_net_params(stage="train")
        if self.split_performed or self.merge_performed:
            self.split_performed = False
            self.merge_performed = False

    def on_validation_epoch_start(self):
        #ipdb.set_trace()
        self.initialize_net_params(stage="val")
        return super().on_validation_epoch_start()

    def initialize_net_params(self, stage="train"):
        #ipdb.set_trace()
        self.codes = []
        if stage == "train":
            if self.current_epoch > 0:
                del self.train_resp, self.train_resp_sub, self.train_gt
            self.train_resp = []
            self.train_resp_sub = []
            self.train_gt = []
        else:
            if self.current_epoch > 0:
                del self.val_resp, self.val_resp_sub, self.val_gt
            self.val_resp = []
            self.val_resp_sub = []
            self.val_gt = []

    def training_step(self, batch, batch_idx):

        logits = self.forward(batch)
        with torch.no_grad():
            seq_vec = self._encode_seq_batch(batch["seq_texts"])
            graph_seq, graph_vec, graph_mask = self._encode_graph_batch(batch["graphs"])
            fused = self.fuser(seq_vec, graph_seq)
            codes = self.proj(fused)

        cluster_loss = self.training_utils.cluster_loss_function(
            codes.view(-1, self.codes_dim), logits,
            model_mus=self.mus, K=self.K, codes_dim=self.codes_dim,
            model_covs=self.covs if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
            pi=self.pi
        )
        loss = self.hparams.cluster_loss_weight * cluster_loss

        if not self.hparams.ignore_subclusters and self.hparams.start_sub_clustering <= self.current_epoch:
            sublogits = self.subcluster(codes, logits)
            subcluster_loss = self.training_utils.subcluster_loss_function_new(
                codes.view(-1, self.codes_dim), logits, sublogits,
                self.K, self.n_sub, self.mus_sub,
                covs_sub=self.covs_sub if self.hparams.subcluster_loss in ("diag_NIG","KL_GMM_2") else None,
                pis_sub=self.pi_sub
            )
            loss = loss + self.hparams.subcluster_loss_weight * subcluster_loss

        y = batch["labels"].to(self.device)
        (
            self.codes,
            self.train_gt,
            self.train_resp,
            self.train_resp_sub,
        ) = self.training_utils.log_codes_and_responses(
            self.codes, getattr(self, "train_gt", []), getattr(self, "train_resp", []),
            getattr(self, "train_resp_sub", []),
            codes, logits.detach(), y, sublogits=None,
        )

        self.log("cluster_net_train/train/cluster_loss",
                self.hparams.cluster_loss_weight * cluster_loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def only_gather_codes(self, codes, y):
        """Only log codes for initialization

        Args:
            codes ([type]): The input data in the latent space
            y ([type]): The ground truth labels
            optimizer_idx ([type]): The optimizer index
        """
        # log only once
        #ipdb.set_trace()
        # if optimizer_idx == self.optimizers_dict_idx["cluster_net_opt"]:
        #     (
        #         self.codes,
        #         self.train_gt,
        #         _,
        #         _,
        #     ) = self.training_utils.log_codes_and_responses(
        #         model_codes=self.codes,
        #         model_gt=self.train_gt,
        #         model_resp=self.train_resp,
        #         model_resp_sub=self.train_resp_sub,
        #         codes=codes,
        #         y=y,
        #         logits=None,
        #     )
        return None

    def _pairwise_prf1_jaccard(self, y_true_np, y_pred_np):

        cm = contingency_matrix(y_true_np, y_pred_np, sparse=False)  # (G, K)
        def comb2(x): 
            return (x * (x - 1) / 2.0).sum()

        TP = comb2(cm)                                   
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        total_row_pairs = comb2(row_sums)                
        total_col_pairs = comb2(col_sums)               
        FP = total_col_pairs - TP                         
        FN = total_row_pairs - TP                       

        # 保护除零
        pr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * pr * rr / (pr + rr)) if (pr + rr) > 0 else 0.0
        jc = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
        fmi = math.sqrt(pr * rr)

        return pr, rr, f1, jc, fmi

    def validation_step(self, batch, batch_idx):
        #ipdb.set_trace()
        x, y = batch
        if self.feature_extractor is not None:
            with torch.no_grad():
                codes = torch.from_numpy(
                    self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                ).to(device=self.device)
        else:
            codes = x

        logits = self.cluster_net(codes)
        if batch_idx == 0 and (self.current_epoch < 5 or self.current_epoch % 50 == 0):
            self.log_logits(logits)

        if self.current_training_stage != "gather_codes":
            cluster_loss = self.training_utils.cluster_loss_function(
                codes.view(-1, self.codes_dim),
                logits,
                model_mus=self.mus,
                K=self.K,
                codes_dim=self.codes_dim,
                model_covs=self.covs
                if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2")
                else None,
                pi=self.pi
            )
            loss = self.hparams.cluster_loss_weight * cluster_loss
            self.log("cluster_net_train/val/cluster_loss", loss)

            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                subclusters = self.subcluster(codes, logits)
                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                    codes.view(-1, self.codes_dim),
                    logits,
                    subclusters,
                    self.K,
                    self.n_sub,
                    self.mus_sub,
                    covs_sub=self.covs_sub
                    if self.hparams.subcluster_loss in ("diag_NIG", "KL_GMM_2")
                    else None,
                    pis_sub=self.pi_sub
                )
                self.log("cluster_net_train/val/subcluster_loss", subcluster_loss)
                loss += self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                subclusters = None
        else:
            loss = torch.tensor(1.0)
            subclusters = None
            logits = None

        # log val data
        (
            self.codes,
            self.val_gt,
            self.val_resp,
            self.val_resp_sub,
        ) = self.training_utils.log_codes_and_responses(
            self.codes,
            self.val_gt,
            self.val_resp,
            model_resp_sub=self.val_resp_sub,
            codes=codes,
            logits=logits,
            y=y,
            sublogits=subclusters,
            stage="val",
        )

        return {"loss": loss}

    def on_training_epoch_end(self, outputs):
        """Perform logging operations and computes the clusters' and the subclusters' centers.
        Also perform split and merges steps

        Args:
            outputs ([type]): [description]
        """
        #ipdb.set_trace()
        if self.current_training_stage == "gather_codes":
            # Initalize plotting utils
            self.plot_utils = PlotUtils(
                self.hparams, self.logger, self.codes.view(-1, self.codes_dim)
            )
            # first time to compute mus
            self.prior.init_priors(self.codes.view(-1, self.codes_dim))
            if self.centers is not None:
                # we have initialization from somewhere
                self.mus = torch.from_numpy(self.centers).cuda()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init

            else:
                self.freeze_mus_after_init_until = 0
                self.mus, self.covs, self.pi, init_labels = init_mus_and_covs(
                    codes=self.codes.view(-1, self.codes_dim),
                    K=self.K,
                    how_to_init_mu=self.hparams.how_to_init_mu,
                    logits=self.train_resp,
                    use_priors=self.hparams.use_priors,
                    prior=self.prior,
                    random_state=0,
                    device=self.device,
                )
                if self.hparams.use_labels_for_eval:
                    if (self.train_gt < 0).any():
                        # some samples don't have label, e.g., stl10
                        gt = self.train_gt[self.train_gt > -1]
                        init_labels = init_labels[self.train_gt > -1]
                    else:
                        gt = self.train_gt
                    if len(gt) > 2 * (10 ** 5):
                        # sample only a portion of the codes
                        gt = gt[:2 * (10**5)]
                    init_nmi = normalized_mutual_info_score(gt, init_labels)
                    init_ari = adjusted_rand_score(gt, init_labels)
                    self.log("cluster_net_train/init_nmi", init_nmi)
                    self.log("cluster_net_train/init_ari", init_ari)
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch == 1):
                    self.plot_utils.visualize_embeddings(
                        self.hparams,
                        self.logger,
                        self.codes_dim,
                        vae_means=self.codes,
                        vae_labels=init_labels,
                        val_resp=None,
                        current_epoch=self.current_epoch,
                        y_hat=None,
                        centers=self.mus,
                        stage="init_Kmeans"
                    )
                if self.current_epoch == 0 and (self.hparams.log_emb in ("every_n_epochs", "only_sampled") and self.current_epoch % self.hparams.log_emb_every == 0):
                    perm = torch.randperm(self.train_gt.size(0))
                    idx = perm[:10000]
                    sampled_points = self.codes[idx]
                    sampled_labeled = self.train_gt[idx] if self.hparams.use_labels_for_eval else None
                    self.plot_utils.visualize_embeddings(
                        self.hparams,
                        self.logger,
                        self.codes_dim,
                        vae_means=sampled_points,
                        vae_labels=sampled_labeled,
                        val_resp=None,
                        current_epoch=self.current_epoch,
                        y_hat=None,
                        centers=None,
                        training_stage='train_sampled',
                        UMAP=False
                    )

        else:
            # add avg loss of all losses
            if not self.hparams.ignore_subclusters:
                clus_losses, subclus_losses = outputs[0], outputs[1]
            else:
                clus_losses = outputs
            avg_clus_loss = torch.stack([x["loss"] for x in clus_losses]).mean()
            self.log("cluster_net_train/train/avg_cluster_loss", avg_clus_loss)
            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                avg_subclus_loss = torch.stack([x["loss"] for x in subclus_losses]).mean()
                self.log("cluster_net_train/train/avg_subcluster_loss", avg_subclus_loss)

            # Compute mus and perform splits/merges
            perform_split = self.training_utils.should_perform_split(
                self.current_epoch
            ) and self.centers is None
            perform_merge = self.training_utils.should_perform_merge(
                self.current_epoch,
                self.split_performed
            ) and self.centers is None
            # do not compute the mus in the epoch(s) following a split or a merge
            if self.centers is not None:
                # we have initialization from somewhere
                self.mus = torch.from_numpy(self.centers).cuda()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
            freeze_mus = self.training_utils.freeze_mus(
                self.current_epoch,
                self.split_performed
            ) or self.current_epoch <= self.freeze_mus_after_init_until

            if not freeze_mus:
                (
                    self.pi,
                    self.mus,
                    self.covs,
                ) = self.training_utils.comp_cluster_params(
                    self.train_resp,
                    self.codes.view(-1, self.codes_dim),
                    self.pi,
                    self.K,
                    self.prior,
                )

            if (self.hparams.start_sub_clustering == self.current_epoch + 1) or (self.hparams.ignore_subclusters and (perform_split or perform_merge)):
                # first time to compute sub mus
                (
                    self.pi_sub,
                    self.mus_sub,
                    self.covs_sub,
                ) = self.training_utils.init_subcluster_params(
                    self.train_resp,
                    self.train_resp_sub,
                    self.codes.view(-1, self.codes_dim),
                    self.K,
                    self.n_sub,
                    self.prior,
                )
            elif (
                self.hparams.start_sub_clustering <= self.current_epoch
                and not freeze_mus and not self.hparams.ignore_subclusters
            ):
                (
                    self.pi_sub,
                    self.mus_sub,
                    self.covs_sub,
                ) = self.training_utils.comp_subcluster_params(
                    self.train_resp,
                    self.train_resp_sub,
                    self.codes,
                    self.K,
                    self.n_sub,
                    self.mus_sub,
                    self.covs_sub,
                    self.pi_sub,
                    self.prior,
                )
            if perform_split and not freeze_mus:
                # 基于簇内熵变化的分裂判定（取代 split_step）
                self.training_utils.last_performed = "split"
                split_decisions = self.training_utils.propose_splits_by_entropy(
                    mus=self.mus,
                    covs=self.covs,
                    pi=self.pi,
                    mus_sub=None if self.hparams.ignore_subclusters else self.mus_sub,
                    covs_sub=None if self.hparams.ignore_subclusters else self.covs_sub,
                    pi_sub=None if self.hparams.ignore_subclusters else self.pi_sub,
                    K=self.K,
                    n_sub=self.n_sub,
                    device=self.device,
                )
                if split_decisions.any():
                    self.split_performed = True
                    self.perform_split_operations(split_decisions)

            if perform_merge and not freeze_mus:
                # 基于簇内熵变化的聚合判定（取代 merge_step）
                self.training_utils.last_performed = "merge"
                mus_to_merge, highest_ll_mus = self.training_utils.propose_merges_by_entropy(
                    mus=self.mus, covs=self.covs, pi=self.pi, K=self.K, device=self.device
                )
                if len(mus_to_merge) > 0:
                    self.merge_performed = True
                    self.perform_merge(mus_to_merge, highest_ll_mus)


            # compute nmi, unique z, etc.
            if self.hparams.log_metrics_at_train and self.hparams.evaluate_every_n_epochs > 0 and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
                self.log_clustering_metrics()
            with torch.no_grad():
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch < 2):
                    self.plot_histograms()
                    self.plot_utils.visualize_embeddings(
                        self.hparams,
                        self.logger,
                        self.codes_dim,
                        vae_means=self.codes,
                        vae_labels=None if not self.hparams.use_labels_for_eval else self.train_gt,
                        val_resp=self.train_resp,
                        current_epoch=self.current_epoch,
                        y_hat=None,
                        centers=self.mus,
                        training_stage='train'
                    )
                    if self.hparams.dataset == "synthetic":
                        if self.split_performed or self.merge_performed:
                            self.plot_utils.update_colors(self.split_performed, self.mus_ind_to_split, self.mus_inds_to_merge)
                        elif self.hparams.use_labels_for_eval:
                            self.plot_utils.plot_cluster_and_decision_boundaries(samples=self.codes, labels=self.train_resp.argmax(-1), gt_labels=self.train_gt, net_centers=self.mus, net_covs=self.covs, n_epoch=self.current_epoch, cluster_net=self)
                    if self.current_epoch in (0, 1, 2, 3, 4, 5, 10, 100, 200, 300, 400, 500, 549, self.hparams.start_sub_clustering, self.hparams.start_sub_clustering+1) or self.split_performed or self.merge_performed:
                        self.plot_histograms(for_thesis=True)
        
        if self.split_performed or self.merge_performed:
            self.update_params_split_merge()
            print("Current number of clusters: ", self.K)

    def on_validation_epoch_end(self, outputs):
        #ipdb.set_trace()
        # Take mean of all batch losses
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("cluster_net_train/val/avg_val_loss", avg_loss)
        if self.current_training_stage != "gather_codes" and self.hparams.evaluate_every_n_epochs and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
            z = self.val_resp.argmax(axis=1).cuda()
            nmi = normalized_mutual_info_score(
                # curr_clusters_assign,
                self.val_gt,
                z,
            )
            self.last_val_NMI = nmi
            self.log_clustering_metrics(stage="val")
            if not (self.split_performed or self.merge_performed) and self.hparams.log_metrics_at_train:
                self.log_clustering_metrics(stage="total")

        if self.hparams.log_emb == "every_n_epochs" and self.current_epoch % 10 == 0 and len(self.val_gt) > 10:
            # not mock data
            self.plot_utils.visualize_embeddings(
                self.hparams,
                self.logger,
                self.codes_dim,
                vae_means=self.codes,
                vae_labels=self.val_gt,
                val_resp=self.val_resp if self.val_resp != [] else None,
                current_epoch=self.current_epoch,
                y_hat=None,
                centers=None,
                training_stage="val_thesis"
            )

        if self.current_epoch > self.hparams.start_sub_clustering and (self.current_epoch % 50 == 0 or self.current_epoch == self.hparams.train_cluster_net - 1):
            from pytorch_lightning.loggers.logger import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                self.plot_histograms(train=False, for_thesis=True)
        
    def subcluster(self, codes, logits, hard_assignment=True):
        #ipdb.set_trace()
        # cluster codes into subclusters
        sub_clus_resp = self.subclustering_net(codes)  # unnormalized
        z = logits.argmax(-1)

        # zero out irrelevant subclusters
        mask = torch.zeros_like(sub_clus_resp)
        mask[np.arange(len(z)), 2 * z] = 1.
        mask[np.arange(len(z)), 2 * z + 1] = 1.

        # perform softmax
        sub_clus_resp = torch.nn.functional.softmax(sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm, dim=1)
        return sub_clus_resp

    def update_subcluster_net_split(self, split_decisions):
        #ipdb.set_trace()
        # update the subcluster net to have the new K
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]

        # remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)

        self.subclustering_net.update_K_split(split_decisions, self.hparams.split_init_weights_sub)

        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())

    def perform_split_operations(self, split_decisions):
        #ipdb.set_trace()
        # split_decisions is a list of k boolean indicators of whether we would want to split cluster k
        # update the cluster net to have the new K
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # only one optimizer
            clus_opt = self.optimizers()

        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        self.cluster_net.update_K_split(
            split_decisions, self.hparams.init_new_weights, self.subclustering_net
        )
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        mus_ind_to_split = torch.nonzero(torch.tensor(split_decisions), as_tuple=False)
        (
            self.mus_new,
            self.covs_new,
            self.pi_new,
            self.mus_sub_new,
            self.covs_sub_new,
            self.pi_sub_new,
        ) = update_models_parameters_split(
            split_decisions,
            self.mus,
            self.covs,
            self.pi,
            mus_ind_to_split,
            self.mus_sub,
            self.covs_sub,
            self.pi_sub,
            self.codes,
            self.train_resp,
            self.train_resp_sub,
            self.n_sub,
            self.hparams.how_to_init_mu_sub,
            self.prior,
            use_priors=self.hparams.use_priors
        )
        # update K
        print(f"Splitting clusters {np.arange(self.K)[split_decisions.bool().tolist()]}")
        self.K += len(mus_ind_to_split)

        if not self.hparams.ignore_subclusters:
            # update subclusters_net
            self.update_subcluster_net_split(split_decisions)
        self.mus_ind_to_split = mus_ind_to_split

    def update_subcluster_nets_merge(self, merge_decisions, pairs_to_merge, highest_ll):
        #ipdb.set_trace()
        # update the cluster net to have the new K
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
        
        # remove old weights from the optimizer state
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_merge(merge_decisions, pairs_to_merge=pairs_to_merge, highest_ll=highest_ll, init_new_weights=self.hparams.merge_init_weights_sub)        
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())

    def perform_merge(self, mus_lists_to_merge, highest_ll_mus, use_priors=True):
        """A method that performs merges of clusters' centers

        Args:
            mus_lists_to_merge (list): a list of lists, each one contains 2 indices of mus that were chosen to be merged.
            highest_ll_mus ([type]): a list of the highest log likelihood index for each pair of mus
        """
        #ipdb.set_trace()
        print(f"Merging clusters {mus_lists_to_merge}")
        mus_lists_to_merge = torch.tensor(mus_lists_to_merge)
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        inds_to_mask[mus_lists_to_merge.flatten()] = 1
        (
            self.mus_new,
            self.covs_new,
            self.pi_new,
            self.mus_sub_new,
            self.covs_sub_new,
            self.pi_sub_new,
        ) = update_models_parameters_merge(
            mus_lists_to_merge,
            inds_to_mask,
            self.K,
            self.mus,
            self.covs,
            self.pi,
            self.mus_sub,
            self.covs_sub,
            self.pi_sub,
            self.codes,
            self.train_resp,
            self.prior,
            use_priors=self.hparams.use_priors,
            n_sub=self.n_sub,
            how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
        )
        # adjust k
        self.K -= len(highest_ll_mus)

        if not self.hparams.ignore_subclusters:
            # update the subclustering net
            self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus)

        # update the cluster net to have the new K
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            # only one optimizer
            clus_opt = self.optimizers()

        # remove old weights from the optimizer state
        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        
        # update cluster net
        self.cluster_net.update_K_merge(
            inds_to_mask,
            mus_lists_to_merge,
            highest_ll_mus,
            init_new_weights=self.hparams.init_new_weights,
        )
        # add parameters to the optimizer
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())

        self.cluster_net.class_fc2.to(self._device)
        self.mus_inds_to_merge = mus_lists_to_merge

    def configure_optimizers(self):
        # 原先只拿 cluster_net 参数；这里合并 StarCoder + GCN + Fusion + Proj
        base_params = []
        for n, p in self.cluster_net.named_parameters():
            if "class_fc2" not in n:
                base_params.append(p)

        # StarCoder 是否训练取决于 freeze_starcoder
        if not getattr(self.hparams, "freeze_starcoder", False):
            base_params += list(self.starcoder.parameters())
        # 始终更新：图编码器、node_type_emb、fuser、proj、token_embed（通常不更新 token_embed；若不想训练可移除）
        base_params += list(self.graph_encoder.parameters())
        base_params += list(self.node_type_emb.parameters())
        base_params += list(self.fuser.parameters())
        if isinstance(self.proj, nn.Linear):
            base_params += list(self.proj.parameters())
        # 可选：是否训练 token_embed。一般冻结：
        # for p in self.token_embed.parameters(): p.requires_grad = False

        cluster_net_opt = optim.Adam(base_params, lr=self.hparams.cluster_lr)
        # 最后一层单独 param group（便于 split/merge 时替换）
        cluster_net_opt.add_param_group({"params": self.cluster_net.class_fc2.parameters()})

        if self.hparams.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
        elif self.hparams.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode="min", factor=0.5, patience=4)
        else:
            cluster_scheduler = None

        self.optimizers_dict_idx = {"cluster_net_opt": 0}
        if not self.hparams.ignore_subclusters:
            sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr)
            self.optimizers_dict_idx["subcluster_net_opt"] = 1
            return (
                {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"},
                {"optimizer": sub_clus_opt}
            )
        return {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"} if cluster_scheduler else cluster_net_opt

    def update_params_split_merge(self):
        #ipdb.set_trace()
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.mus_sub = self.mus_sub_new
        self.covs_sub = self.covs_sub_new
        self.pi = self.pi_new
        self.pi_sub = self.pi_sub_new

    def init_covs_and_pis_given_mus(self):
        # each point will be hard assigned to its closest cluster and then compute covs and pis.
        # compute dist mat
        #ipdb.set_trace()
        if self.hparams.use_priors_for_net_params_init:
            _, cov_prior = self.prior.init_priors(self.mus)  # giving mus and nopt codes because we only need the dim for the covs
            self.covs = torch.stack([cov_prior for k in range(self.K)])
            p_counts = torch.ones(self.K) * 10
            self.pi = p_counts / float(self.K * 10)  # a uniform pi prior

        else:
            dis_mat = torch.empty((len(self.codes), self.K))
            for i in range(self.K):
                dis_mat[:, i] = torch.sqrt(((self.codes - self.mus[i]) ** 2).sum(axis=1))
            # get hard assingment
            hard_assign = torch.argmin(dis_mat, dim=1)

            # data params
            vals, counts = torch.unique(hard_assign, return_counts=True)
            if len(counts) < self.K:
                new_counts = []
                for k in range(self.K):
                    if k in vals:
                        new_counts.append(counts[vals == k])
                    else:
                        new_counts.append(0)
                counts = torch.tensor(new_counts)
            pi = counts / float(len(self.codes))
            data_covs = compute_data_covs_hard_assignment(hard_assign.numpy(), self.codes, self.K, self.mus.cuda(), self.prior)
            if self.use_priors:
                covs = []
                for k in range(self.K):
                    codes_k = self.codes[hard_assign == k]
                    cov_k = self.prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])
                    covs.append(cov_k)
                covs = torch.stack(covs)
            self.covs = covs
            self.pi = pi

    def log_logits(self, logits):
        #ipdb.set_trace()
        for k in range(self.K):
            max_k = logits[logits.argmax(axis=1) == k].detach().cuda().numpy()
            if len(max_k > 0):
                fig = plt.figure(figsize=(10, 3))
                for i in range(len(max_k[:20])):
                    if i == 0:
                        plt.bar(np.arange(self.K), max_k[i], fill=False, label=len(max_k))
                    else:
                        plt.bar(np.arange(self.K), max_k[i], fill=False)
                plt.xlabel("Clusters inds")
                plt.ylabel("Softmax histogram")
                plt.title(f"Epoch {self.current_epoch}: cluster {k}")
                plt.legend()

                # self.logger.log_image(f"cluster_net_train/val/logits_reaction_fig_cluster_{k}", fig)
                plt.close(fig)

    def plot_histograms(self, train=True, for_thesis=False):
        #ipdb.set_trace()
        pi = self.pi_new if self.split_performed or self.merge_performed else self.pi
        if self.hparams.ignore_subclusters:
            pi_sub = None
        else:
            pi_sub = (
                self.pi_sub_new
                if self.split_performed or self.merge_performed
                else self.pi_sub
                if self.hparams.start_sub_clustering <= self.current_epoch
                else None
            )

        fig = self.plot_utils.plot_weights_histograms(
            K=self.K,
            pi=pi,
            start_sub_clustering=self.hparams.start_sub_clustering,
            current_epoch=self.current_epoch,
            pi_sub=pi_sub,
            for_thesis=for_thesis
        )
        if for_thesis:
            stage = "val_for_thesis"
        else:
            stage = "train" if train else "val"

        from pytorch_lightning.loggers.logger import DummyLogger
        if not isinstance(self.logger, DummyLogger):
            self.logger.log_image(f"cluster_net_train/{stage}/clusters_weights_fig", fig)
        plt.close(fig)

    def plot_clusters_high_dim(self, stage="train"):
        #ipdb.set_trace()
        resps = {
            "train": (self.train_resp, self.train_resp_sub),
            "val": (self.val_resp, self.val_resp_sub),
        }
        gt = {"train": self.train_gt, "val": self.val_gt}
        (resp, resp_sub) = resps[stage]
        cluster_net_labels = self.training_utils.update_labels_after_split_merge(
            resp.argmax(-1),
            self.split_performed,
            self.merge_performed,
            self.mus,
            self.mus_ind_to_split,
            self.mus_inds_to_merge,
            resp_sub,
        )
        fig = self.plot_utils.plot_clusters_colored_by_label(
            samples=self.codes,
            y_gt=gt[stage],
            n_epoch=self.current_epoch,
            K=len(torch.unique(gt[stage])),
        )
        plt.close(fig)
        self.logger.log_image(f"cluster_net_train/{stage}/clusters_fig_gt_labels", fig)
        fig = self.plot_utils.plot_clusters_colored_by_net(
            samples=self.codes,
            y_net=cluster_net_labels,
            n_epoch=self.current_epoch,
            K=len(torch.unique(cluster_net_labels)),
        )
        self.logger.log_image("cluster_net_train/train/clusters_fig_net_labels", fig)
        plt.close(fig)

    ### CHANGED
    def log_clustering_metrics(self, stage="train"):
        # 统一计算：pred_K, FMI, JC, PR, RR, F1
        print("Evaluating...")

        if stage == "train":
            gt = self.train_gt
            resp = self.train_resp
        elif stage == "val":
            gt = self.val_gt
            resp = self.val_resp
            self.log("cluster_net_train/Networks_k", self.K)
        elif stage == "total":
            gt = torch.cat([self.train_gt, self.val_gt])
            resp = torch.cat([self.train_resp, self.val_resp])

        # 预测标签（argmax）
        z = resp.argmax(axis=1).detach().cpu().numpy()
        pred_K = int(len(np.unique(z)))

        # 过滤 -1（未标注）
        if isinstance(gt, torch.Tensor):
            gt_np = gt.detach().cpu().numpy()
        else:
            gt_np = np.asarray(gt)
        mask = gt_np > -1
        z = z[mask]
        gt_np = gt_np[mask]

        if len(gt_np) <= 1 or len(z) <= 1 or pred_K <= 0:
            pr = rr = f1 = jc = fmi = 0.0
        else:
            pr, rr, f1, jc, fmi = self._pairwise_prf1_jaccard(gt_np, z)

        # 记录 6 个指标
        self.log(f"cluster_net_train/{stage}/pred_K", pred_K, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/FMI", fmi, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/JC", jc, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/PR", pr, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/RR", rr, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/F1", f1, on_epoch=True, on_step=False)

        # 控制台打印（离线时）
        if self.hparams.offline and ((self.hparams.log_metrics_at_train and stage == "train") or (not self.hparams.log_metrics_at_train and stage != "train")):
            print(f"[{stage}] K={pred_K} | FMI={fmi:.4f} JC={jc:.4f} PR={pr:.4f} RR={rr:.4f} F1={f1:.4f}")

        # 可选：起止时刻的“alt_”日志保持一致（非必须）
        if self.current_epoch in (0, 1, self.hparams.train_cluster_net - 1):
            alt_stage = "start" if self.current_epoch in (0,1) else "end"
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_pred_K", pred_K, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_FMI", fmi, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_JC", jc, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_PR", pr, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_RR", rr, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_F1", f1, on_epoch=True, on_step=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        #ipdb.set_trace()
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--init_k", default=3, type=int, help="number of initial clusters"
        )
        parser.add_argument(
            "--clusternet_hidden",
            type=int,
            default=50,
            help="The dimensions of the hidden dim of the clusternet. Defaults to 50.",
        )
        parser.add_argument(
            "--clusternet_hidden_layer_list",
            type=int,
            nargs="+",
            default=[50],
            help="The hidden layers in the clusternet. Defaults to [50, 50].",
        )
        parser.add_argument(
            "--transform_input_data",
            type=str,
            default="normalize",
            choices=["normalize", "min_max", "standard", "standard_normalize", "None", None],
            help="Use normalization for embedded data",
        )
        parser.add_argument(
            "--cluster_loss_weight",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--init_cluster_net_weights",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--when_to_compute_mu",
            type=str,
            choices=["once", "every_epoch", "every_5_epochs"],
            default="every_epoch",
        )
        parser.add_argument(
            "--how_to_compute_mu",
            type=str,
            choices=["kmeans", "soft_assign"],
            default="soft_assign",
        )
        parser.add_argument(
            "--how_to_init_mu",
            type=str,
            choices=["kmeans", "soft_assign", "kmeans_1d"],
            default="kmeans",
        )
        parser.add_argument(
            "--how_to_init_mu_sub",
            type=str,
            choices=["kmeans", "soft_assign", "kmeans_1d"],
            default="kmeans_1d",
        )
        parser.add_argument(
            "--log_emb_every",
            type=int,
            default=20,
        )
        parser.add_argument(
            "--log_emb",
            type=str,
            default="never",
            choices=["every_n_epochs", "only_sampled", "never"]
        )
        parser.add_argument(
            "--train_cluster_net",
            type=int,
            default=300,
            help="Number of epochs to pretrain the cluster net",
        )
        parser.add_argument(
            "--cluster_lr",
            type=float,
            default=0.0005,
        )
        parser.add_argument(
            "--subcluster_lr",
            type=float,
            default=0.005,
        )
        parser.add_argument(
            "--lr_scheduler", type=str, default="ReduceOnP", choices=["StepLR", "None", "ReduceOnP"]
        )
        parser.add_argument(
            "--start_sub_clustering",
            type=int,
            default=35,
        )
        parser.add_argument(
            "--subcluster_loss_weight",
            type=float,
            default=1.0,
        )
        parser.add_argument(
            "--start_splitting",
            type=int,
            default=45,
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=10.0,
        )
        parser.add_argument(
            "--softmax_norm",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--subcluster_softmax_norm",
            type=float,
            default=1,
        )
        parser.add_argument(
            "--split_prob",
            type=float,
            default=None,
            help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
        )
        parser.add_argument(
            "--merge_prob",
            type=float,
            default=None,
            help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
        )
        parser.add_argument(
            "--init_new_weights",
            type=str,
            default="same",
            choices=["same", "random", "subclusters"],
            help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
        )
        parser.add_argument(
            "--start_merging",
            type=int,
            default=45,
            help="The epoch in which to start consider merge proposals",
        )
        parser.add_argument(
            "--merge_init_weights_sub",
            type=str,
            default="highest_ll",
            help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
        )
        parser.add_argument(
            "--split_init_weights_sub",
            type=str,
            default="random",
            choices=["same_w_noise", "same", "random"],
            help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
        )
        parser.add_argument(
            "--split_every_n_epochs",
            type=int,
            default=10,
            help="Example: if set to 10, split proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--split_merge_every_n_epochs",
            type=int,
            default=30,
            help="Example: if set to 10, split proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--merge_every_n_epochs",
            type=int,
            default=10,
            help="Example: if set to 10, merge proposals will be made every 10 epochs",
        )
        parser.add_argument(
            "--raise_merge_proposals",
            type=str,
            default="brute_force_NN",
            help="how to raise merge proposals",
        )
        parser.add_argument(
            "--cov_const",
            type=float,
            default=0.005,
            help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
        )
        parser.add_argument(
            "--freeze_mus_submus_after_splitmerge",
            type=int,
            default=2,
            help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
        )
        parser.add_argument(
            "--freeze_mus_after_init",
            type=int,
            default=5,
            help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
        )
        parser.add_argument(
            "--use_priors",
            type=int,
            default=1,
            help="Whether to use priors when computing model's parameters",
        )
        parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
        parser.add_argument(
            "--pi_prior", type=str, default="uniform", choices=["uniform", None]
        )
        parser.add_argument(
            "--prior_dir_counts",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--prior_kappa",
            type=float,
            default=0.0001,
        )
        parser.add_argument(
            "--NIW_prior_nu",
            type=float,
            default=None,
            help="Need to be at least codes_dim + 1",
        )
        parser.add_argument(
            "--prior_mu_0",
            type=str,
            default="data_mean",
        )
        parser.add_argument(
            "--prior_sigma_choice",
            type=str,
            default="isotropic",
            choices=["iso_005", "iso_001", "iso_0001", "data_std"],
        )
        parser.add_argument(
            "--prior_sigma_scale",
            type=float,
            default=".005",
        )
        parser.add_argument(
            "--prior_sigma_scale_step",
            type=float,
            default=1.,
            help="add to change sigma scale between alternations"
        )
        parser.add_argument(
            "--compute_params_every",
            type=int,
            help="How frequently to compute the clustering params (mus, sub, pis)",
            default=1,
        )
        parser.add_argument(
            "--start_computing_params",
            type=int,
            help="When to start to compute the clustering params (mus, sub, pis)",
            default=25,
        )
        parser.add_argument(
            "--cluster_loss",
            type=str,
            help="What kind og loss to use",
            default="KL_GMM_2",
            choices=["diag_NIG", "isotropic", "isotropic_2", "isotropic_3", "isotropic_4", "KL_GMM_2"],
        )
        parser.add_argument(
            "--subcluster_loss",
            type=str,
            help="What kind og loss to use",
            default="isotropic",
            choices=["diag_NIG", "isotropic", "KL_GMM_2"],
        )
        
        parser.add_argument(
            "--use_priors_for_net_params_init",
            type=bool,
            default=True,
            help="when the net is re-initialized after an AE round, if centers are given, if True it will initialize the covs and the pis using the priors, if false it will compute them using min dist."
        )
        parser.add_argument(
            "--ignore_subclusters",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--log_metrics_at_train",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--evaluate_every_n_epochs",
            type=int,
            default=5,
            help="How often to evaluate the net"
        )
        return parser
