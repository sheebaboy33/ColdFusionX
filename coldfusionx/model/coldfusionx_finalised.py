# coldfusionx/model/coldfusionx_finalised.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

# Import your sub-modules from the local layers file
from .layers import (
    ResidualTemporalFiLM, 
    IDEmbeddingLayer, 
    StochasticItemBlendModule, 
    MetaEmbeddingGenerator, 
    CTRHead
)

# Suppress the specific UserWarning regarding NestedTensor with norm_first=True
warnings.filterwarnings("ignore", "enable_nested_tensor is True...")


class Proposed_ColdFusionX_v12(nn.Module):
    """
    ColdFusionX (Final Proposed Architecture).
    
    A Dual-Pathway architecture that solves the Cold-Start CTR dilemma. It integrates 
    Micro-Drift Temporal Adaptation, Meta-Embedding Synthesis, and Stochastic Alpha Masking 
    into a cohesive Mean-Pooled Transformer framework.
    """
    def __init__(self, fusion_config: dict, meta_config: dict, dropout: float, scene_dim: int, attn_hidden: int):
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
        self.id_layer = IDEmbeddingLayer(num_users=self.num_users, num_items=self.num_items, num_scenes=self.num_scenes, 
                                         id_dim=self.id_dim, scene_dim=scene_dim, dropout=dropout
        )

        self.temporal_film = ResidualTemporalFiLM(time_dim=self.time_dim, text_dim=self.text_dim, image_dim=self.image_dim,
                                                  hidden_dim=64, dropout=dropout)
        
        self.meta_gen = MetaEmbeddingGenerator(text_dim=self.text_dim, image_dim=self.image_dim, time_dim=self.time_dim, meta_dim=self.id_dim, 
                                               attn_hidden=attn_hidden, dropout=dropout, use_scene=True, num_scenes=self.num_scenes, scene_emb_dim=16 )

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

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, scene_idx: torch.Tensor, text_vec: torch.Tensor, image_vec: torch.Tensor, 
                time_vec: torch.Tensor, cold_mask=None, item_count=None):
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