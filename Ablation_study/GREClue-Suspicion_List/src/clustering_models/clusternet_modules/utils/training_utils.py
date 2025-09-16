import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from torchvision.utils import save_image, make_grid


from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    compute_pi_k,
    compute_mus,
    compute_covs,
    init_mus_and_covs_sub,
    compute_mus_covs_pis_subclusters
)


class training_utils:
    def __init__(self, hparams):
        self.hparams = hparams
        self.pretraining_complete = False
        self.alt_count = 0
        self.last_performed = "merge"
        self.device = "cuda" if torch.cuda.is_available() and hparams.gpus is not None else "cpu"

    @staticmethod
    def change_model_requires_grad(model, require_grad_bool=True):
        for param in model.parameters():
            param.requires_grad = require_grad_bool

    @staticmethod
    def log_codes_and_responses(
        model_codes,
        model_gt,
        model_resp,
        model_resp_sub,
        codes,
        logits,
        y,
        sublogits=None,
        stage="train",
    ):
        """A function to log data used to compute model's parameters.

        Args:
            codes (torch.tensor): the current batch codes (in emedding space)
            logits (torch.tensor): the clustering net responses to the codes
            y (torch.tensor): the ground truth labels
            sublogits ([type], optional): [description]. Defaults to None. The subclustering nets response to the codes
        """
        if model_gt == []:
            # first batch of the epoch
            if codes is not None:
                model_codes = codes.detach().cuda()
            model_gt = y.detach().cuda()
            if logits is not None:
                model_resp = logits.detach().cuda()
            if sublogits is not None:
                model_resp_sub = sublogits.detach().cuda()
        else:
            if codes is not None:
                model_codes = torch.cat([model_codes, codes.detach().cuda()])
            model_gt = torch.cat([model_gt, y.detach().cuda()])
            if logits is not None:
                model_resp = torch.cat([model_resp, logits.detach().cuda()])
            if sublogits is not None:
                model_resp_sub = torch.cat([model_resp_sub, sublogits.detach().cuda()])
        return model_codes, model_gt, model_resp, model_resp_sub

    @staticmethod
    def log_vae_encodings(vae_means, vae_labels, means, labels):
        if vae_means == []:
            # start of an epoch
            vae_means = means.detach().cuda()
            vae_labels = labels.detach().cuda()
        else:
            vae_means = torch.cat([vae_means, means.detach().cuda()])
            vae_labels = torch.cat([vae_labels, labels.detach().cuda()])
        return vae_means, vae_labels

    def should_perform_split(self, current_epoch):
        # computes whether a split step should be performed in the current epoch
        return (
            self.hparams.start_splitting <= current_epoch
            and (
                (current_epoch - self.hparams.start_splitting)
                % self.hparams.split_merge_every_n_epochs
                == 0
            )
            and self.last_performed == "merge"
        )

    def should_perform_merge(self, current_epoch, split_performed):
        # computes whether a merge step should be performed in the current epoch
        return (
            self.hparams.start_merging <= current_epoch
            and (
                (current_epoch - self.hparams.start_merging)
                % self.hparams.split_merge_every_n_epochs
                == 0
            )
            and not split_performed
            and self.last_performed == "split"
        )

    def freeze_mus(self, current_epoch, split_performed):
        if (
            current_epoch < self.hparams.start_computing_params
            or (self.hparams.compute_params_every != 1 and current_epoch % self.hparams.compute_params_every != 0)
        ):
            return True
        else:
            split_occured = torch.tensor(
                    [
                        self.should_perform_split(current_epoch - epoch)
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            merge_occured = torch.tensor(
                    [
                        self.should_perform_merge(
                            current_epoch - epoch, split_performed
                        )
                        for epoch in range(
                            1,
                            self.hparams.freeze_mus_submus_after_splitmerge + 1,
                            1,
                        )
                    ]
                ).any()
            return split_occured or merge_occured

    def comp_cluster_params(self, train_resp, codes, pi, K, prior=None):
        # compute pi
        pi = compute_pi_k(train_resp, prior=prior if self.hparams.use_priors else None)
        mus = compute_mus(
            codes=codes,
            logits=train_resp,
            pi=pi,
            K=K,
            how_to_compute_mu=self.hparams.how_to_compute_mu,
            use_priors=self.hparams.use_priors,
            prior=prior,
        )

        covs = compute_covs(
            logits=train_resp,
            codes=codes,
            K=K,
            mus=mus,
            use_priors=self.hparams.use_priors,
            prior=prior,
        )
        return pi, mus, covs

    def comp_subcluster_params(
        self,
        train_resp,
        train_resp_sub,
        codes,
        K,
        n_sub,
        mus_sub,
        covs_sub,
        pi_sub,
        prior=None,
    ):

        mus_sub, covs_sub, pi_sub = compute_mus_covs_pis_subclusters(
            codes=codes, logits=train_resp, logits_sub=train_resp_sub,
            mus_sub=mus_sub, K=K, n_sub=n_sub, use_priors=self.hparams.use_priors, prior=prior
        )
        return pi_sub, mus_sub, covs_sub

    def init_subcluster_params(
        self, train_resp, train_resp_sub, codes, K, n_sub, prior=None
    ):
        mus_sub, covs_sub, pi_sub = [], [], []
        for k in range(K):
            mus, covs, pis = init_mus_and_covs_sub(
                codes=codes,
                k=k,
                n_sub=n_sub,
                how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
                logits=train_resp,
                logits_sub=train_resp_sub,
                prior=prior,
                use_priors=self.hparams.use_priors,
                device=self.device
            )
            mus_sub.append(mus)
            covs_sub.append(covs)
            pi_sub.append(pis)
        mus_sub = torch.cat(mus_sub)
        covs_sub = torch.cat(covs_sub)
        pi_sub = torch.cat(pi_sub)

        return pi_sub, mus_sub, covs_sub

    def cluster_loss_function(
        self, c, r, model_mus, K, codes_dim, model_covs=None, pi=None, logger=None, 
    ):
        if self.hparams.cluster_loss == "isotropic":
            # Isotropic
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (r_tag * ((torch.norm(C_tag - mus_tag.to(device=self.device), dim=1)) ** 2)).mean()

        elif self.hparams.cluster_loss == "diag_NIG":
            # NIG prior
            # K * N, D
            C_tag = c.repeat(1, K).view(-1, codes_dim)
            sigmas = torch.sqrt(model_covs).repeat(c.shape[0], 1)
            mus_tag = model_mus.repeat(c.shape[0], 1)
            r_tag = r.flatten()
            return (
                r_tag
                * ((torch.norm((C_tag - mus_tag.to(device=self.device)) / sigmas.to(device=self.device), dim=1)) ** 2)
            ).mean()

        elif self.hparams.cluster_loss == "KL_GMM_2":
            r_gmm = []
            for k in range(K):
                gmm_k = MultivariateNormal(model_mus[k].double().to(device=self.device), model_covs[k].double().to(device=self.device))
                prob_k = gmm_k.log_prob(c.detach().double())
                r_gmm.append((prob_k + torch.log(pi[k])).double())
            r_gmm = torch.stack(r_gmm).T
            max_values, _ = r_gmm.max(axis=1, keepdim=True)
            r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
            r_gmm = torch.exp(r_gmm)
            eps = 0.00001
            r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
            r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
            
            return nn.KLDivLoss(reduction="batchmean")(
                torch.log(r),
                r_gmm.float().to(device=self.device),
            )

        raise NotImplementedError("No such loss")

    def subcluster_loss_function(
        self, codes, logits, subresp, K, n_sub, mus_sub, covs_sub=None, pis_sub=None
    ):
        if self.hparams.subcluster_loss == "isotropic":
            # Isotropic
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]

                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        loss += (
                            r * ((torch.norm(codes_k - mus_tag.to(device=self.device), dim=1)) ** 2)
                        ).sum()
            return loss / float(len(codes))

        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k: 2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        gmm_k = MultivariateNormal(mus_sub[2 * k + k_sub].double().to(device=self.device), covs_sub[2 * k + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(codes_k.detach().double())
                        r_gmm.append((prob_k + torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(
                            covs_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        )
                        loss += (
                            r
                            * (
                                (
                                    torch.norm(
                                        (codes_k - mus_tag.to(device=self.device) / sigma_sub.to(device=self.device)),
                                        dim=1,
                                    )
                                )
                                ** 2
                            )
                        ).sum()
            return loss

        raise NotImplementedError("No such loss!")

    def subcluster_loss_function_new(
        self, codes, logits, subresp, K, n_sub, mus_sub, covs_sub=None, pis_sub=None
    ):
        if self.hparams.subcluster_loss == "isotropic":
            # Isotropic

            C_tag = codes.repeat(1, 2 * K).view(-1, codes.size(1))
            mus_tag = mus_sub.repeat(codes.shape[0], 1)
            r_tag = subresp.flatten()
            return (r_tag * ((torch.norm(C_tag - mus_tag.to(device=self.device), dim=1)) ** 2)).sum() / float(len(codes))

        elif self.hparams.cluster_loss == "KL_GMM_2":
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                r = subresp[z == k, 2 * k: 2 * k + 2]
                if len(codes_k) > 0:
                    r_gmm = []
                    for k_sub in range(n_sub):
                        gmm_k = MultivariateNormal(mus_sub[2 * k + k_sub].double().to(device=self.device), covs_sub[2 * k + k_sub].double().to(device=self.device))
                        prob_k = gmm_k.log_prob(codes_k.detach().double())
                        r_gmm.append((prob_k + torch.log(pis_sub[2 * k + k_sub])).double())
                    r_gmm = torch.stack(r_gmm).T
                    max_values, _ = r_gmm.max(axis=1, keepdim=True)
                    r_gmm -= torch.log(torch.exp((r_gmm - max_values)).sum(axis=1, keepdim=True)) + max_values
                    r_gmm = torch.exp(r_gmm)
                    eps = 0.00001
                    r_gmm = (r_gmm + eps) / (r_gmm + eps).sum(axis=1, keepdim=True)
                    r = (r + eps) / (r + eps).sum(axis=1, keepdim=True)
                    loss += nn.KLDivLoss(reduction="batchmean")(
                            torch.log(r),
                            r_gmm.float().to(device=self.device),
                    )
            return loss

        elif self.hparams.subcluster_loss == "diag_NIG":
            # NIG
            loss = 0
            for k in range(K):
                z = logits.argmax(-1)
                codes_k = codes[z == k]
                if codes_k.shape[0] > 0:
                    # else: handle empty clusters, won't necessarily happen
                    for k_sub in range(n_sub):
                        r = subresp[z == k, k, :][:, 2 * k + k_sub]
                        mus_tag = mus_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        sigma_sub = torch.sqrt(
                            covs_sub[2 * k + k_sub].repeat(codes_k.shape[0], 1)
                        )
                        loss += (
                            r
                            * (
                                (
                                    torch.norm(
                                        (codes_k - mus_tag.to(device=self.device) / sigma_sub.to(device=self.device)),
                                        dim=1,
                                    )
                                )
                                ** 2
                            )
                        ).sum()
            return loss

        raise NotImplementedError("No such loss!")

    def comp_std(self, codes, hard_assignments, K):
        stds = []
        for k in range(K):
            codes_k = codes[hard_assignments == k]
            if len(codes_k > 0):
                per_dim_std = codes_k.std(axis=0)
            else:
                per_dim_std = torch.sqrt(codes.std(axis=0))
            stds.append(per_dim_std)
        return torch.stack(stds)

    def autoencoder_kl_dist_loss_function(
        z, mu, log_var, hard_assign, model_mus, model_std, mean=False
    ):
        z = z.detach()
        p = torch.distributions.Normal(model_mus[hard_assign], model_std[hard_assign])
        log_prob_p_z = p.log_prob(z)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        log_prob_q_z = q.log_prob(z)

        dist_kl = log_prob_p_z - log_prob_q_z
        dist_kl *= log_prob_p_z.exp()
        if mean:
            dist_kl = dist_kl.mean()
        else:
            dist_kl = dist_kl.sum()
        return dist_kl

    def update_labels_after_split_merge(
        self,
        hard_assign,
        split_performed,
        merge_performed,
        mus,
        mus_ind_to_split,
        mus_inds_to_merge,
        resp_sub,
    ):
        cluster_net_labels = hard_assign
        if split_performed or merge_performed:
            if split_performed:
                label_map = {}
                count = 0
                count_split = 0
                second_subcluster_inds = torch.tensor([])
                for mu_ind in range(len(mus)):
                    if mu_ind in mus_ind_to_split:
                        mask_current_mu = cluster_net_labels == mu_ind
                        # first cluster
                        label_map[mu_ind] = (
                            len(mus) - len(mus_ind_to_split) + count_split
                        )
                        # list second_subcluster so we will remember to increase its label by one
                        sub_assign = resp_sub[
                            mask_current_mu, mu_ind, 2 * mu_ind: 2 * mu_ind + 2
                        ].argmax(-1)
                        inds_current_mu = mask_current_mu.nonzero(as_tuple=False)
                        second_subcluster_inds = torch.cat(
                            [second_subcluster_inds, inds_current_mu[sub_assign == 1]]
                        )
                        count_split += 2
                    else:
                        label_map[mu_ind] = count
                        count += 1
                new_labels = torch.zeros_like(cluster_net_labels) - 1
                for key, value in label_map.items():
                    new_labels[cluster_net_labels == key] = value
                new_labels[
                    second_subcluster_inds.clone().detach().type(torch.long)
                ] += 1
            elif merge_performed:
                count = 0
                label_map = {}
                pairs = torch.zeros(len(mus_inds_to_merge))

                for mu_ind in range(len(mus)):
                    if mu_ind in mus_inds_to_merge.flatten():
                        which_pair = (mus_inds_to_merge == mu_ind).nonzero(
                            as_tuple=False
                        )[0][0]
                        if pairs[which_pair] == 0:
                            # first, open new cluster
                            label_map[mu_ind] = (
                                len(mus) - len(mus_inds_to_merge.flatten()) + which_pair
                            )
                            pairs[which_pair] += 1
                        else:
                            # second, join the already opened cluster
                            # find the first of this pair
                            which_pair_col = (mus_inds_to_merge == mu_ind).nonzero(
                                as_tuple=False
                            )[0][1]
                            first = mus_inds_to_merge[
                                which_pair, (which_pair_col + 1) % 2
                            ]
                            label_map[mu_ind] = label_map[first.item()]
                    else:
                        label_map[mu_ind] = count
                        count += 1
                new_labels = torch.zeros_like(cluster_net_labels)
                for key, value in label_map.items():
                    new_labels[cluster_net_labels == key] = value
            return new_labels

    def should_init_em(self, split_performed, merge_performed, previous_training_stage, current_stage):
        # A flag to reinitialized the EM object. Should occur in two occasions:
        # 1. The embeddings have changed
        # 2. K has changes
        K_changed = split_performed or merge_performed
        embeddings_changed = previous_training_stage in ["pretrain_ae", 'only_ae', "train_ae_w_add_loss", "only_ae_w_cluster_loss", "train_together", ]
        if (K_changed or embeddings_changed) and current_stage == "only_cluster_net":
            return True
        return False

    def should_perform_em(
        self, current_epoch, split_performed, merge_performed, previous_training_stage, current_stage
    ):
        return current_epoch > 0 and (
            self.should_init_em(
                split_performed, merge_performed, previous_training_stage, current_stage
            )
            and self.hparams.cluster_loss == "KL_GMM"
        )

    @staticmethod
    def update_following_split(mus, mus_ind_to_split, train_resp_sub, cluster_net_labels):
        label_map = {}
        count = 0
        count_split = 0
        second_subcluster_inds = torch.tensor([])
        for mu_ind in range(len(mus)):
            if mu_ind in mus_ind_to_split:
                mask_current_mu = cluster_net_labels == mu_ind
                # first cluster
                label_map[mu_ind] = (
                    len(mus) - len(mus_ind_to_split) + count_split
                )
                # list second_subcluster so we will remember to increase its label by one
                sub_assign = train_resp_sub[
                    mask_current_mu, mu_ind, 2 * mu_ind: 2 * mu_ind + 2
                ].argmax(-1)
                inds_current_mu = mask_current_mu.nonzero(as_tuple=False)
                second_subcluster_inds = torch.cat(
                    [second_subcluster_inds, inds_current_mu[sub_assign == 1]]
                )
                count_split += 2

            else:
                label_map[mu_ind] = count
                count += 1
        new_labels = torch.zeros_like(cluster_net_labels) - 1
        for key, value in label_map.items():
            new_labels[cluster_net_labels == key] = value
        new_labels[
            second_subcluster_inds.clone().detach().type(torch.long)
        ] += 1
        return new_labels

    @staticmethod
    def update_following_merge(mus, mus_inds_to_merge, cluster_net_labels):
        count = 0
        label_map = {}
        pairs = torch.zeros(len(mus_inds_to_merge))
        for mu_ind in range(len(mus)):
            if mu_ind in mus_inds_to_merge.flatten():
                which_pair = (mus_inds_to_merge == mu_ind).nonzero(
                    as_tuple=False
                )[0][0]
                if pairs[which_pair] == 0:
                    # first, open new cluster
                    label_map[mu_ind] = (
                        len(mus)
                        - len(mus_inds_to_merge.flatten())
                        + which_pair
                    )
                    pairs[which_pair] += 1
                else:
                    # second, join the already opened cluster
                    # find the first of this pair
                    which_pair_col = (mus_inds_to_merge == mu_ind).nonzero(
                        as_tuple=False
                    )[0][1]
                    first = mus_inds_to_merge[
                        which_pair, (which_pair_col + 1) % 2
                    ]
                    label_map[mu_ind] = label_map[first.item()]
            else:
                label_map[mu_ind] = count
                count += 1
        new_labels = torch.zeros_like(cluster_net_labels)
        for key, value in label_map.items():
            new_labels[cluster_net_labels == key] = value
        return new_labels

    def log_metric(self, metric_name, metric_val):
        self.log(metric_name, metric_val)

    @staticmethod
    def get_updated_net_labels(cluster_net_labels, split_performed, merge_performed, mus, mus_ind_to_split, mus_inds_to_merge, train_resp_sub):
        """ Compute the updated net labels if a split/merge has occured in this epoch
        """
        if split_performed:
            return training_utils.update_following_split(mus, mus_ind_to_split, train_resp_sub, cluster_net_labels)
        elif merge_performed:
            return training_utils.update_following_merge(mus, mus_inds_to_merge, cluster_net_labels)

    @staticmethod
    def _best_cluster_fit(y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_assignment(w.max() - w)
        map_dict = {}
        for j in range(len(col_ind)):
            map_dict[col_ind[j]] = row_ind[j]
        y_true_new = np.array([map_dict[i] for i in y_true])
        return y_true_new, row_ind, col_ind, w
        # ======== 新增：稳定计算 log|Σ| 的小工具 ========
    @staticmethod
    def _logdet_stable(cov, eps=1e-6):
        # cov: [..., D, D]
        cov = cov.double()
        # 数值稳定：必要时加抖动
        sign, logabsdet = torch.slogdet(cov)
        needs_fix = (sign <= 0) | torch.isnan(logabsdet)
        if torch.any(needs_fix):
            I = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
            cov = cov + eps * I
            sign, logabsdet = torch.slogdet(cov)
        return logabsdet

    @staticmethod
    def _merge_cov(mu_i, cov_i, w_i, mu_j, cov_j, w_j):
        """
        以权重 w_i, w_j（可用 π）合并两个高斯，返回合并后的协方差（总协方差=类内+类间）。
        """
        w_sum = (w_i + w_j).clamp(min=1e-12)
        mu_m = (w_i * mu_i + w_j * mu_j) / w_sum
        # 类内项
        cov_within = (w_i * cov_i + w_j * cov_j) / w_sum
        # 类间项
        diff_i = (mu_i - mu_m).unsqueeze(-1)  # D x 1
        diff_j = (mu_j - mu_m).unsqueeze(-1)
        cov_between = (w_i * (diff_i @ diff_i.transpose(-1, -2)) + w_j * (diff_j @ diff_j.transpose(-1, -2))) / w_sum
        return (cov_within + cov_between).double()

    def propose_merges_by_entropy(self, mus, covs, pi, K, device="cpu"):
        """
        基于“合并代价 = 合并后簇的 log|Σ| - 原两簇 log|Σ| 的平均”。
        若某对 (i,j) 的代价 < 其他所有对代价均值的 1/3，则判定合并该对。
        返回：pairs_to_merge (list of [i,j]), highest_ll_mus (list of [ [idx_kept] ])
        """
        pairs = []
        costs = []

        # 预先算各簇 log|Σ|
        logdets = self._logdet_stable(covs.to(device))  # [K]
        for i in range(K):
            for j in range(i + 1, K):
                cov_m = self._merge_cov(mus[i], covs[i], pi[i], mus[j], covs[j], pi[j])
                logdet_m = self._logdet_stable(cov_m.to(device))
                cost_ij = (logdet_m - 0.5 * (logdets[i] + logdets[j])).item()
                pairs.append([i, j])
                costs.append(cost_ij)

        if len(pairs) == 0:
            return [], []

        costs_t = torch.tensor(costs, device=device, dtype=torch.double)
        pairs_to_merge = []
        highest_ll_mus = []
        for idx, (i, j) in enumerate(pairs):
            if len(pairs) > 1:
                # 其他对的平均代价
                mask = torch.ones(len(pairs), dtype=torch.bool, device=device)
                mask[idx] = False
                mean_other = costs_t[mask].mean().item()
                threshold = mean_other / 3.0
            else:
                # 只有一对时，用 cost < 0 作为“远小于”的近似标准
                threshold = 0.0

            if costs[idx] <= threshold:
                pairs_to_merge.append([i, j])
                # 用权重大的一簇作为“保留权重”的簇（兼容原 update_K_merge 的 highest_ll 语义）
                keep = i if pi[i] >= pi[j] else j
                highest_ll_mus.append([torch.tensor(keep)])

        return pairs_to_merge, highest_ll_mus

    def propose_splits_by_entropy(self, mus, covs, pi, mus_sub, covs_sub, pi_sub, K, n_sub, device="cpu"):
        """
        基于“分裂代价 = 两个子簇 log|Σ| 的平均 - 原簇 log|Σ|”。
        若某簇 k 的代价 < 其他簇分裂代价均值的 1/4，则判定分裂该簇。
        需要已有子簇参数 (mus_sub/covs_sub/pi_sub)；若未启用子簇则全部 False。
        返回：Bool tensor 长度 K
        """
        split_decisions = torch.zeros(K, dtype=torch.bool, device=device)

        if (mus_sub is None) or (covs_sub is None):
            return split_decisions  # 未启用子簇则不分裂

        # 父簇与子簇的 log|Σ|
        logd_parent = self._logdet_stable(covs.to(device))  # [K]

        # 每个父簇对应两个子簇：索引为 2*k 与 2*k+1
        costs = []
        valid = []
        for k in range(K):
            idx0, idx1 = 2 * k, 2 * k + 1
            if idx1 >= covs_sub.size(0):
                costs.append(float("inf"))
                valid.append(False)
                continue
            logd_s0 = self._logdet_stable(covs_sub[idx0].to(device))
            logd_s1 = self._logdet_stable(covs_sub[idx1].to(device))
            cost_k = (0.5 * (logd_s0 + logd_s1) - logd_parent[k]).item()
            costs.append(cost_k)
            valid.append(True)

        if not any(valid):
            return split_decisions

        costs_t = torch.tensor([c for c, v in zip(costs, valid) if v], device=device, dtype=torch.double)
        idx_map = [i for i, v in enumerate(valid) if v]

        for loc, k in enumerate(idx_map):
            if len(idx_map) > 1:
                mask = torch.ones(len(idx_map), dtype=torch.bool, device=device)
                mask[loc] = False
                mean_other = costs_t[mask].mean().item()
                threshold = mean_other / 4.0
            else:
                # 只有一个可分裂候选时，用 cost < 0 作为“远小于”的近似标准
                threshold = 0.0

            if costs[k] <= threshold:
                split_decisions[k] = True

        return split_decisions

    @staticmethod
    def cluster_acc(y_true, y_pred, y_pred_top5=None):
        y_true_new, row_ind, col_ind, w = training_utils._best_cluster_fit(y_true.numpy(), y_pred.numpy())
        if y_pred_top5 is not None:
            y_true_new = torch.from_numpy(y_true_new).unsqueeze(0).repeat(5, 1)
            acc_top5 = (y_pred_top5.T == y_true_new).any(axis=0).sum() * 1.0 / y_pred.numpy().size
            acc_top5 = acc_top5.item()
        else:
            acc_top5 = 0.
        
        return acc_top5, np.round(w[row_ind, col_ind].sum() * 1.0 / y_pred.numpy().size, 5)

    def save_cluster_examples(self, logits, x_for_vis, y, epoch, init_num=0, num_img=20, grid_size=8):
        # save 20 examples of each cluster and also record its true label class.
        K = logits.shape[1]
        hard_assign = logits.argmax(-1)
        for k in range(K):
            # take images of the cluster
            x_k = x_for_vis[hard_assign == k][:num_img]
            y_gt = y[hard_assign == k][:num_img]
            # save each image
            for i in range(min(num_img, x_k.shape[0])):
                save_image(x_k[i], f"{self.hparams.dataset}_imgs/clusternet{init_num}_epoch{epoch}_clus{k}_label{y_gt[i]}_{i}.jpeg")
            # save as a grid
            num_imgs = min(grid_size, x_k.shape[0])
            if num_imgs > 0:
                grid = make_grid(x_k[:num_imgs], nrow=num_imgs)
                save_image(grid, f"{self.hparams.dataset}_imgs/clusternet{init_num}_epoch{epoch}_clus{k}.jpeg")

    def save_batch_of_images(self, x_for_vis, nrow=8, rows=5):
        x = x_for_vis[:rows * nrow]
        grid = make_grid(x, nrow=nrow)
        save_image(grid, f"{self.hparams.dataset}_imgs/{self.hparams.dataset}_grid.jpeg")
