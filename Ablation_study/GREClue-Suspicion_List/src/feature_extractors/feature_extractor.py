from src.feature_extractors.autoencoder import AutoEncoder, ConvAutoEncoder
from src.feature_extractors.llmextractor import LLMExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class FeatureExtractor(nn.Module):
    def __init__(self, args, input_dim):
        super(FeatureExtractor, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.feature_extractor = None
        self.autoencoder = None
        self.latent_dim = None

        if args.fe_type == 'codellama':
            self.feature_extractor = LLMExtractor(
                model_name_or_path="/home/sdu/yangzhenyu/LLM/StarCoder-3b/",  # e.g., "codellama/CodeLlama-7b-hf"
                device=args.device,
                latent_dim=args.latent_dim,
                cls_strategy=args.cls_strategy
            )
            self.latent_dim = args.latent_dim

        elif args.fe_type == 'contrastive':
            self.feature_extractor = self.get_fe_model(output_dim=args.latent_dim)
            self.latent_dim = args.latent_dim

        elif args.fe_type == 'autoencoder':
            if self.args.dataset == "usps":
                self.autoencoder = ConvAutoEncoder(args, input_dim=self.input_dim)
            else:
                self.autoencoder = AutoEncoder(args, input_dim=self.input_dim)
            self.latent_dim = self.autoencoder.latent_dim

        else:
            raise ValueError(f"Unknown fe_type: {args.fe_type}")

    def forward(self, X, latent=False):
        if self.feature_extractor:
            return self.feature_extractor(X, latent=True if latent else False)

        if self.autoencoder:
            output = self.autoencoder.encoder(X)
            if latent:
                return output
            return self.autoencoder.decoder(output)

        return X  # Shouldn't reach here

    def decode(self, latent_X):
        if self.autoencoder:
            return self.autoencoder.decoder(latent_X)
        raise NotImplementedError("decode() not implemented for feature_extractor")

    def extract_features(self, x):
        return self.forward(x, latent=True)

    def get_fe_model(self, output_dim=128):
        backbone = self._get_backbone()
        model = ContrastiveModel(backbone=backbone, features_dim=output_dim)

        # Load pretrained weights
        if self.args.pretrain_path is not None and os.path.exists(self.args.pretrain_path):
            state = torch.load(self.args.pretrain_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print("Loaded pretrained weights")
        return model

    def _get_backbone(self):
        if self.args.dataset in ('cifar-10', 'cifar-20'):
            from src.feature_extractors.resnet_cifar import resnet18
            backbone = resnet18()
        elif self.args.dataset == 'stl-10':
            from src.feature_extractors.resnet_stl import resnet18
            backbone = resnet18()
        elif 'imagenet' in self.args.dataset:
            from src.feature_extractors.resnet import resnet50
            backbone = resnet50()
        return backbone


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.features_dim = features_dim
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)
        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_dim, features_dim)
            )
        else:
            raise ValueError(f"Invalid head type: {head}")

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x))
        return F.normalize(features, dim=1)
