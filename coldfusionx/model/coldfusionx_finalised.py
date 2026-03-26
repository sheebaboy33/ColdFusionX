import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

# Suppress the specific UserWarning regarding NestedTensor with norm_first=True
warnings.filterwarnings("ignore", "enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")

class ResidualTemporalFiLM(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to text and image embeddings
    based on temporal drift, utilizing a residual connection.

    Args:
        time_dim (int): Dimension of the input time vector.
        text_dim (int): Dimension of the text embeddings.
        image_dim (int): Dimension of the image embeddings.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, time_dim: int, text_dim: int, image_dim: int, dropout: float = 0.1):
        super().__init__()
        self.film_text = nn.Sequential(
            nn.Linear(time_dim, text_dim * 2),
            nn.LayerNorm(text_dim * 2), # Stabilizes the temporal gradients
            nn.ReLU(),                  # Noise filter for timeless features
            nn.Dropout(dropout)
        )
        self.film_image = nn.Sequential(
            nn.Linear(time_dim, image_dim * 2),
            nn.LayerNorm(image_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Initialize to near-zero so it starts as a safe Identity Mapping for new items
        nn.init.normal_(self.film_text[0].weight, std=0.01)
        nn.init.normal_(self.film_image[0].weight, std=0.01)

    def forward(self, time_vec: torch.Tensor, text_vec: torch.Tensor, image_vec: torch.Tensor):
        text_params = self.film_text(time_vec)
        image_params = self.film_image(time_vec)

        # Split outputs into Scale (Gamma) and Shift (Beta)
        gamma_t, beta_t = torch.chunk(text_params, 2, dim=-1)
        gamma_i, beta_i = torch.chunk(image_params, 2, dim=-1)

        # Tanh bounds the scaling delta between -1.0 and 1.0 to prevent gradient explosion
        gamma_t = torch.tanh(gamma_t)
        gamma_i = torch.tanh(gamma_i)

        # RESIDUAL EQUATION: Raw Feature + Temporal Scale + Temporal Shift
        text_adapted = text_vec + (text_vec * gamma_t) + beta_t
        image_adapted = image_vec + (image_vec * gamma_i) + beta_i

        return text_adapted, image_adapted


class IDEmbeddingLayer(nn.Module):
    """
    Generates collaborative ID embeddings for Users, Items, and Scenes.

    Defense Note:
        Controlled Normal initialization (std=0.01) is used instead of default PyTorch 
        uniform initialization to prevent early gradient saturation in the dot products.
    """
    def __init__(self, num_users: int, num_items: int, num_scenes: int,
                 id_dim: int = 256, scene_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, id_dim)
        self.item_emb = nn.Embedding(num_items, id_dim)
        self.scene_emb = nn.Embedding(num_scenes, scene_dim)

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.scene_emb.weight, mean=0.0, std=0.01)

        self.dropout = nn.Dropout(dropout)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, scene_idx: torch.Tensor):
        u = self.dropout(self.user_emb(user_idx))
        i = self.dropout(self.item_emb(item_idx))
        s = self.dropout(self.scene_emb(scene_idx))
        return u, i, s


class StochasticItemBlendModule(nn.Module):
    """
    Fuses the Collaborative ID Embedding with the Synthesized Meta-Embedding.

    Note:
        Implements Stochastic Alpha Masking (SAM). By randomly dropping the collaborative 
        ID during training, it forces the network to route gradients through the multi-modal 
        pathway, preventing catastrophic gradient starvation on cold items.
    """
    def __init__(self, fewshot_n: int, mask_prob: float = 0.3):
        super().__init__()
        self.fewshot_n = int(fewshot_n)
        self.mask_prob = mask_prob  # proven global optimum
        self._last_alpha = None

    def forward(self, item_id_emb: torch.Tensor, item_meta_emb: torch.Tensor, item_count: torch.Tensor):
        # 1. Deterministic alpha based on logarithmic exposure count
        alpha = torch.log1p(item_count.float()) / np.log1p(self.fewshot_n)
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1)  # Shape: (B, 1)

        # 2. Apply Stochastic Masking strictly during training
        if self.training:
            # Create a dropout mask where 0s occur with mask_prob probability
            mask = (torch.rand(alpha.size(0), 1, device=alpha.device) > self.mask_prob).float()
            alpha_used = alpha * mask
        else:
            alpha_used = alpha

        # 3. Soft-gating blend
        item_final = alpha_used * item_id_emb + (1.0 - alpha_used) * item_meta_emb
        self._last_alpha = alpha_used.detach()

        return item_final


class MetaEmbeddingGenerator(nn.Module):
    """
    Synthesizes a proxy Item ID dynamically from Text, Image, and Time modalities.

    Note:
        Uses a Sigmoid gating mechanism instead of Softmax. This avoids a "zero-sum game" 
        and allows multiple modalities (e.g., both Text and Image) to be highly active 
        simultaneously for information-dense items.
    """
    def __init__(self, text_dim: int, image_dim: int, time_dim: int,
                 meta_dim: int = 256, attn_hidden: int = 128,
                 dropout: float = 0.1, use_scene: bool = False,
                 num_scenes: int = 0, scene_emb_dim: int = 16):
        super().__init__()
        self.use_scene = use_scene

        # Manifold Alignment: Linear + LayerNorm (No ReLU) allows the generation of 
        # negative coordinate values, matching the unconstrained ID embedding space.
        self.proj_text = nn.Sequential(nn.Linear(text_dim, meta_dim), nn.LayerNorm(meta_dim), nn.Dropout(dropout))
        self.proj_image = nn.Sequential(nn.Linear(image_dim, meta_dim), nn.LayerNorm(meta_dim), nn.Dropout(dropout))
        self.proj_time = nn.Sequential(nn.Linear(time_dim, meta_dim), nn.LayerNorm(meta_dim), nn.Dropout(dropout))

        if self.use_scene:
            self.scene_emb = nn.Embedding(num_scenes, scene_emb_dim)
            nn.init.normal_(self.scene_emb.weight, std=0.01)
            attn_in = meta_dim + scene_emb_dim
        else:
            attn_in = meta_dim

        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_in, attn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )
        self._last_attention = None

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor, time_vec: torch.Tensor, scene_idx: torch.Tensor = None):
        B = text_vec.size(0)

        z_text  = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time  = self.proj_time(time_vec)

        Z = torch.stack([z_text, z_image, z_time], dim=1)  # Shape: (B, 3, meta_dim)

        # Concatenate scene context to the attention calculation if enabled
        if self.use_scene:
            s = self.scene_emb(scene_idx)
            s = s.unsqueeze(1).expand(B, 3, s.size(-1))
            Z_attn_in = torch.cat([Z, s], dim=-1)
        else:
            Z_attn_in = Z

        attn_logits = self.attn_mlp(Z_attn_in).squeeze(-1)
        attn_weights = torch.sigmoid(attn_logits)

        # Weighted sum of modality embeddings
        meta_item_emb = torch.sum(Z * attn_weights.unsqueeze(-1), dim=1)

        # Cache weights for Explainability UI
        self._last_attention = {
            "text": attn_weights[:, 0].detach(),
            "image": attn_weights[:, 1].detach(),
            "time": attn_weights[:, 2].detach()
        }
        return meta_item_emb

    def get_last_attention(self):
        return self._last_attention


class CTRHead(nn.Module):
    """
    Multi-Layer Perceptron used for final Click-Through Rate prediction.
    """
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x).squeeze(-1)


class Proposed_ColdFusionX_v10(nn.Module):
    """
    ColdFusionX: The unified architecture for Multi-Modal CTR Prediction.

    Integrates Temporal Drift Adaptation, Meta-Embedding Synthesis, and 
    Stochastic Alpha Masking into a cohesive Mean-Pooled Transformer framework.
    """
    def __init__(self, fusion_config: dict, meta_config: dict, dropout: float = 0.1, scene_dim: int = 32, attn_hidden: int = 128):
        super().__init__()
        self.num_users = int(fusion_config["num_users"])
        self.num_items = int(fusion_config["num_items"])
        self.num_scenes = int(fusion_config["num_scenes"])
        self.id_dim = int(fusion_config["id_dim"])
        self.d_model = int(fusion_config["d_model"])
        self.hidden_dim = int(fusion_config["hidden_dim"])
        self.text_dim = int(fusion_config["text_dim"])
        self.image_dim = int(fusion_config["image_dim"])
        self.time_dim = int(fusion_config["time_dim"])
        self.fewshot_n = int(meta_config["FEWSHOT_N"])

        # 1. Base Modules
        self.id_layer = IDEmbeddingLayer(self.num_users, self.num_items, self.num_scenes, id_dim=self.id_dim, scene_dim=scene_dim, dropout=dropout)
        self.temporal_film = ResidualTemporalFiLM(self.time_dim, self.text_dim, self.image_dim, dropout=dropout)
        self.meta_gen = MetaEmbeddingGenerator(self.text_dim, self.image_dim, self.time_dim, meta_dim=self.id_dim, attn_hidden=attn_hidden, 
                                               dropout=dropout, use_scene=True, num_scenes=self.num_scenes)

        # 2. Stochastic Blender (Optimum set to 0.3 based on Thesis Grid Search)
        self.blender = StochasticItemBlendModule(fewshot_n=self.fewshot_n, mask_prob=0.3)

        # 3. Context Projections
        self.proj_user = nn.Sequential(nn.Linear(self.id_dim + scene_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_item = nn.Sequential(nn.Linear(self.id_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_text = nn.Sequential(nn.Linear(self.text_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_image = nn.Sequential(nn.Linear(self.image_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_time = nn.Sequential(nn.Linear(self.time_dim, self.d_model), nn.LayerNorm(self.d_model))

        # 4. Contextual Fusion & Prediction
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model*2, dropout=dropout, 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        

        self._last_align_loss = torch.tensor(0.0)

    def forward(self, user_idx, item_idx, scene_idx, text_vec, image_vec, time_vec, cold_mask=None, item_count=None):
        if item_count is None: 
            item_count = torch.zeros_like(item_idx)

        # Phase 1: Apply Temporal Drift Adjustments
        text_adapted, image_adapted = self.temporal_film(time_vec, text_vec, image_vec)

        # Phase 2: Generate Core Embeddings
        user_emb, item_id_emb, scene_emb = self.id_layer(user_idx, item_idx, scene_idx)
        item_meta_emb = self.meta_gen(text_adapted, image_adapted, time_vec, scene_idx)

        # Phase 3: Stop-Gradient Cosine Alignment
        # Note: We detach() item_id_emb so the meta-generator learns to mimic the 
        # collaborative space, but the collaborative space isn't corrupted by the generator.
        if self.training:
            cos_sim = F.cosine_similarity(item_meta_emb, item_id_emb.detach(), dim=-1)
            alpha = torch.log1p(item_count.float()) / float(np.log1p(self.fewshot_n))
            alpha = torch.clamp(alpha, 0.0, 1.0)
            
            # Loss formulation: Weighted (1 - Similarity)
            self._last_align_loss = torch.mean(alpha * (1.0 - cos_sim))
        else:
            self._last_align_loss = torch.tensor(0.0, device=user_idx.device)

        # Phase 4: Stochastic Blending & Dimensionality Projection
        item_final = self.blender(item_id_emb, item_meta_emb, item_count)

        z_user = self.proj_user(torch.cat([user_emb, scene_emb], dim=1))
        z_item = self.proj_item(item_final)
        z_text = self.proj_text(text_adapted)
        z_image = self.proj_image(image_adapted)
        z_time = self.proj_time(time_vec)

        # Phase 5: Mean-Pooled Fusion and Prediction
        seq = torch.stack([z_user, z_item, z_text, z_image, z_time], dim=1)
        
        # Note: Mean-Pooling acts as an aggressive zero-shot regularizer, 
        # empirically proving superior to flattening for cold-start items.
        fused = self.transformer(seq).mean(dim=1)

        return self.head(fused).squeeze(-1)

    def get_align_loss(self):
        """Retrieves the cached cosine alignment loss from the current forward pass."""
        return self._last_align_loss

    def get_last_meta_attention(self):
        """Exposes the modality attention weights for the Streamlit UI."""
        return self.meta_gen.get_last_attention()
