import pytorch_lightning as pl
from src.clustering_models.clusternet_modules.clusternetasmodel import ClusterNetModel

class ClusterNetTrainer:
    def __init__(self, args, init_k, latent_dim=None, feature_extractor=None, centers=None, init_num=0):
        self.args = args
        latent_dim = latent_dim or getattr(args, "fusion_hidden_dim", 512)
        self.cluster_model = ClusterNetModel(
            hparams=args,
            input_dim=latent_dim,
            init_k=init_k,
            feature_extractor=feature_extractor,
            centers=centers,
            init_num=init_num
        )
        if self.args.seed:
            pl.utilities.seed.seed_everything(self.args.seed)

    def fit(self, train_loader, val_loader, logger, n_epochs, centers=None):
        from pytorch_lightning.loggers import NeptuneLogger
        from pytorch_lightning.loggers.logger import DummyLogger
        import torch

        if isinstance(logger, NeptuneLogger) and getattr(logger, "api_key", None) in (None, "", "your_API_token"):
            print("Neptune API token error")
            logger = DummyLogger()

        use_gpu = (getattr(self.args, "device", "cuda") == "cuda") and torch.cuda.is_available()
        devices = getattr(self.args, "gpus", None)
        if devices is None:
            devices = 1 if use_gpu else None

        cluster_trainer = pl.Trainer(
            logger=logger if logger is not None else False,
            max_epochs=n_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=devices,
            num_nodes=getattr(self.args, "num_nodes", 1),
            deterministic=bool(self.args.seed),
            num_sanity_val_steps=0,
        )
        if self.args.seed:
            pl.utilities.seed.seed_everything(self.args.seed)
        if centers is not None:
            self.cluster_model.centers = centers
        cluster_trainer.fit(self.cluster_model, train_loader, val_loader)

    def get_current_K(self):
        return self.cluster_model.K

    def get_clusters_centers(self):
        return self.cluster_model.mus.detach().cpu().numpy()

    def get_clusters_covs(self):
        return self.cluster_model.covs.detach().cpu().numpy()

    def get_clusters_pis(self):
        return self.cluster_model.pi.detach().cpu().numpy()
