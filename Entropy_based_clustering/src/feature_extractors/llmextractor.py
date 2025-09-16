import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM

class LLMExtractor(nn.Module):
    def __init__(self, model_name_or_path="/home/sdu/yangzhenyu/LLM/StarCoder-3b/", device="cuda", latent_dim=128, cls_strategy="mean_pool"):
        super(LLMExtractor, self).__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.cls_strategy = cls_strategy

        # Load pre-trained model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()  # Optionally fine-tune later

        # Projection layer for dimensionality reduction
        hidden_size = self.model.config.hidden_size
        self.projector = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, latent=True):
        """
        Args:
            x: List[str] or Tokenized Tensor of input texts
        Returns:
            latent features (N x latent_dim)
        """
        if isinstance(x, list):
            encoding = self.tokenizer(
                x,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)
        else:
            encoding = x.to(self.device)

        with torch.no_grad():  # Set to train if you want fine-tuning
            output = self.model(**encoding, output_hidden_states=True)

        hidden_states = output.last_hidden_state  # (B, T, D)

        if self.cls_strategy == "cls":
            cls_rep = hidden_states[:, 0]  # [CLS] token
        elif self.cls_strategy == "mean_pool":
            cls_rep = hidden_states.mean(dim=1)  # Average pooling
        elif self.cls_strategy == "last_token":
            lengths = (encoding['attention_mask'].sum(dim=1) - 1)
            cls_rep = torch.stack([
                hidden_states[i, l] for i, l in enumerate(lengths)
            ])
        else:
            raise ValueError(f"Unknown strategy {self.cls_strategy}")

        projected = self.projector(cls_rep)
        return F.normalize(projected, dim=1)
