import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

# Suppress the specific UserWarning regarding NestedTensor with norm_first=True
warnings.filterwarnings("ignore", "enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True")

class ResidualTemporalFiLM(nn.Module):
    """
    Micro-Drift Feature-wise Linear Modulation (FiLM) Layer.
    
    This module captures cyclical shifts in user intent (e.g., Hour of Day, Day of Week) 
    and dynamically modulates the multi-modal text and image embeddings. 
    
    Note:
        It employs a Residual architecture with Zero-Initialization on the final layers. 
        This guarantees the module begins training as a perfect Identity mapping, preventing 
        early gradient instability while allowing the optimizer to scale the temporal 
        parameters only if they mathematically lower the loss.
    """
    def __init__(self, time_dim: int = 5, text_dim: int = 768, image_dim: int = 768, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # Bottleneck architecture (hidden_dim=64) prevents over-parameterization 
        # and overfitting on sparse temporal representations.
        self.film_text = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(), # GELU provides a smoother gradient flow than ReLU, ideal for cyclical data
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, text_dim * 2)
        )
        
        self.film_image = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, image_dim * 2)
        )

        # Zero-Initialization Trick: Sets weights/biases of the final linear layers to 0.
        # This forces the initial Scale (Gamma) to 0 and Shift (Beta) to 0.
        nn.init.zeros_(self.film_text[-1].weight)
        nn.init.zeros_(self.film_text[-1].bias)
        nn.init.zeros_(self.film_image[-1].weight)
        nn.init.zeros_(self.film_image[-1].bias)

    def forward(self, time_vec: torch.Tensor, text_vec: torch.Tensor, image_vec: torch.Tensor):
        # Generate the modulation parameters from the micro-drift features
        text_params = self.film_text(time_vec)
        image_params = self.film_image(time_vec)

        # Chunk the outputs evenly into Scale (Gamma) and Shift (Beta) vectors
        gamma_t, beta_t = torch.chunk(text_params, 2, dim=-1)
        gamma_i, beta_i = torch.chunk(image_params, 2, dim=-1)

        # Residual Modulation Equation: Feature * (1 + Scale) + Shift
        # Because Gamma/Beta start at 0, this initially evaluates to: Feature * (1) + 0
        text_adapted = text_vec * (1.0 + gamma_t) + beta_t
        image_adapted = image_vec * (1.0 + gamma_i) + beta_i

        return text_adapted, image_adapted


class IDEmbeddingLayer(nn.Module):
    """
    Generates standard collaborative ID embeddings for Users, Items, and Scenes.

    Note:
        Controlled Normal initialization (mean=0.0, std=0.01) is used instead of 
        the default PyTorch uniform initialization. This tighter bound prevents 
        early gradient saturation when computing dot products in the Transformer.
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
    Stochastic Alpha Masking (SAM) module.
    
    Dynamically routes the network's attention between the Collaborative ID 
    and the Synthesized Meta-Embedding based on an item's historical exposure.

    Note:
        During training, the collaborative ID is randomly dropped out with a 
        probability of `mask_prob`. This forces the network to route gradients 
        through the multi-modal meta-generator, preventing the generator from 
        starving when warm items dominate the batch.
    """
    def __init__(self, fewshot_n: int, mask_prob: float = 0.3):
        super().__init__()
        self.fewshot_n = int(fewshot_n)
        self.mask_prob = mask_prob 
        self._last_alpha = None

    def forward(self, item_id_emb: torch.Tensor, item_meta_emb: torch.Tensor, item_count: torch.Tensor):
        # Calculate a deterministic alpha ratio using logarithmic scaling.
        # 0 exposures = 0.0 alpha (100% Meta-Embedding)
        # >= fewshot_n exposures = 1.0 alpha (100% Collaborative ID)
        alpha = torch.log1p(item_count.float()) / np.log1p(self.fewshot_n)
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1)  # Shape: (Batch, 1)

        if self.training:
            # Generate a dropout mask: 1.0 most of the time, 0.0 with `mask_prob` probability
            mask = (torch.rand(alpha.size(0), 1, device=alpha.device) > self.mask_prob).float()
            alpha_used = alpha * mask
        else:
            # During inference (testing/production), use the pure deterministic ratio
            alpha_used = alpha

        # Soft-gating blend equation
        item_final = alpha_used * item_id_emb + (1.0 - alpha_used) * item_meta_emb
        self._last_alpha = alpha_used.detach()

        return item_final


class MetaEmbeddingGenerator(nn.Module):
    """
    Synthesizes a proxy Item ID by fusing Text, Image, and Time modalities.

    Note:
        - Attention Gating: Uses a Sigmoid mechanism instead of Softmax. Softmax 
          forces a zero-sum game (if Text is 0.9, Image must be 0.1). Sigmoid allows 
          multiple modalities to be highly active simultaneously for information-dense items.
        - Manifold Alignment: The projection layers intentionally omit ReLU activations. 
          This allows the generator to produce negative coordinate values, matching the 
          unconstrained vector space of standard Collaborative IDs.
    """
    def __init__(self, text_dim: int, image_dim: int, time_dim: int,
                 meta_dim: int = 256, attn_hidden: int = 128,
                 dropout: float = 0.1, use_scene: bool = False,
                 num_scenes: int = 0, scene_emb_dim: int = 16):
        super().__init__()
        self.use_scene = use_scene

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

        # Project modalities into the shared meta-dimensional space
        z_text = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time = self.proj_time(time_vec)

        Z = torch.stack([z_text, z_image, z_time], dim=1)  # Shape: (Batch, 3, meta_dim)

        # Inject scene context to make the attention 'Scenario-Aware'
        if self.use_scene:
            s = self.scene_emb(scene_idx)
            s = s.unsqueeze(1).expand(B, 3, s.size(-1))
            Z_attn_in = torch.cat([Z, s], dim=-1)
        else:
            Z_attn_in = Z

        # Calculate attention logits and apply Sigmoid gating
        attn_logits = self.attn_mlp(Z_attn_in).squeeze(-1)
        attn_weights = torch.sigmoid(attn_logits)

        # Generate the final proxy ID via a weighted sum
        meta_item_emb = torch.sum(Z * attn_weights.unsqueeze(-1), dim=1)

        # Cache the weights to power the Explainability UI dashboard
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
    Standard Multi-Layer Perceptron (MLP) for final Click-Through Rate prediction.
    Compresses the fused representation sequence down to a single logit.
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


class Proposed_ColdFusionX_v12(nn.Module):
    """
    ColdFusionX (Final Proposed Architecture).
    
    A Dual-Pathway architecture that solves the Cold-Start CTR dilemma. It integrates 
    Micro-Drift Temporal Adaptation, Meta-Embedding Synthesis, and Stochastic Alpha Masking 
    into a cohesive Mean-Pooled Transformer framework.
    """
    def __init__(self, fusion_config: dict, meta_config: dict, dropout: float = 0.1, scene_dim: int = 32, attn_hidden: int = 128):
        super().__init__()
        
        # Parse configurations
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

        # 1. Base Feature Extractors & Generators
        self.id_layer = IDEmbeddingLayer(self.num_users, self.num_items, self.num_scenes, id_dim=self.id_dim, scene_dim=scene_dim, dropout=dropout)
        self.temporal_film = ResidualTemporalFiLM(self.time_dim, self.text_dim, self.image_dim, dropout=dropout)
        self.meta_gen = MetaEmbeddingGenerator(self.text_dim, self.image_dim, self.time_dim, meta_dim=self.id_dim, attn_hidden=attn_hidden, 
                                               dropout=dropout, use_scene=True, num_scenes=self.num_scenes)

        # 2. The Bridge: Stochastic Blender
        self.blender = StochasticItemBlendModule(fewshot_n=self.fewshot_n, mask_prob=0.3)

        # 3. Dimensionality Projection (Aligns all features to Transformer d_model)
        self.proj_user = nn.Sequential(nn.Linear(self.id_dim + scene_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_item = nn.Sequential(nn.Linear(self.id_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_text = nn.Sequential(nn.Linear(self.text_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_image = nn.Sequential(nn.Linear(self.image_dim, self.d_model), nn.LayerNorm(self.d_model))
        self.proj_time = nn.Sequential(nn.Linear(self.time_dim, self.d_model), nn.LayerNorm(self.d_model))

        # 4. Global Context Fusion (Transformer) & MLP Head
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model*2, dropout=dropout, 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = CTRHead(input_dim=self.d_model, hidden_dim=self.hidden_dim, dropout=dropout)

        # Cache for Joint-Optimization loss
        self._last_align_loss = torch.tensor(0.0)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, scene_idx: torch.Tensor, text_vec: torch.Tensor, image_vec: torch.Tensor, time_vec: torch.Tensor, cold_mask=None, item_count=None):
        if item_count is None: 
            item_count = torch.zeros_like(item_idx)

        # Phase 1: Modulate Visual/Text features based on Micro-Temporal Drift
        text_adapted, image_adapted = self.temporal_film(time_vec, text_vec, image_vec)

        # Phase 2: Look up Collaborative IDs and Synthesize the Proxy ID
        user_emb, item_id_emb, scene_emb = self.id_layer(user_idx, item_idx, scene_idx)
        item_meta_emb = self.meta_gen(text_adapted, image_adapted, time_vec, scene_idx)

        # Phase 3: Stop-Gradient Cosine Alignment (Joint Optimization)
        if self.training:
            # Note: We strictly detach() the item_id_emb. This forces the Meta-Generator 
            # to map its output into the established collaborative space, preventing the 
            # proxy from corrupting the established Warm Item embeddings.
            cos_sim = F.cosine_similarity(item_meta_emb, item_id_emb.detach(), dim=-1)

            # The alignment loss is weighted by alpha, ensuring that the model heavily 
            # anchors its alignment on items that actually have reliable historical data.
            alpha = torch.log1p(item_count.float()) / float(np.log1p(self.fewshot_n))
            alpha = torch.clamp(alpha, 0.0, 1.0)
            self._last_align_loss = torch.mean(alpha * (1.0 - cos_sim))
        else:
            self._last_align_loss = torch.tensor(0.0, device=user_idx.device)

        # Phase 4: Dynamic Blending and Projection
        item_final = self.blender(item_id_emb, item_meta_emb, item_count)

        z_user = self.proj_user(torch.cat([user_emb, scene_emb], dim=1))
        z_item = self.proj_item(item_final)
        z_text = self.proj_text(text_adapted)
        z_image = self.proj_image(image_adapted)
        z_time = self.proj_time(time_vec)

        # Phase 5: Mean-Pooled Fusion and Final Prediction
        seq = torch.stack([z_user, z_item, z_text, z_image, z_time], dim=1)
        
        # Note: Mean-Pooling the sequence acts as an aggressive zero-shot regularizer.
        # Empirically, this prevents overfitting better than flattening the sequence.
        fused = self.transformer(seq).mean(dim=1)

        return self.head(fused)

    def get_align_loss(self):
        """Retrieves the cached cosine alignment loss calculated during the forward pass."""
        return self._last_align_loss

    def get_last_meta_attention(self):
        """Exposes the modality attention weights to power the Streamlit Explainability UI."""
        return self.meta_gen.get_last_attention()