import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------
# SUB-MODULES (The Building Blocks)
# ---------------------------------------------------------

class TransformerFusion(nn.Module):
    """
    Replaces ModalityAttentionFusion.
    Uses a Transformer Encoder to allow modalities to INTERACT non-linearly.
    Input: Sequence of [User, Item, Text, Image, Time]
    Output: A single compressed d_model vector that captures cross-modal relationships.
    """
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*2, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Crucial for convergence stability with high-variance embeddings
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Final compression: Flatten 5 tokens -> 1 vector matching d_model
        self.compress = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, z_user, z_item, z_text, z_image, z_time):
        # 1. Stack as a Sequence: (B, 5, D)
        seq = torch.stack([z_user, z_item, z_text, z_image, z_time], dim=1)
        
        # 2. Pass through Transformer -> (B, 5, D)
        transformed = self.transformer(seq)
        
        # 3. Flatten and Compress -> (B, D)
        flat = transformed.reshape(transformed.size(0), -1)
        fused = self.compress(flat)
        
        return fused


class ScenarioAwareMMoE(nn.Module):
    """
    Scenario-aware Mixture-of-Experts (MMoE).
    Routes the Transformer's fused representation based on the user's scene.
    """
    def __init__(
        self, d_model: int, hidden_dim: int, num_experts: int, num_scenes: int,
        gate_embed_dim: int = 16, expert_dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_scenes = num_scenes

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(expert_dropout),
            )
            for _ in range(num_experts)
        ])

        self.scene_embed = nn.Embedding(num_scenes, gate_embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(gate_embed_dim, gate_embed_dim),
            nn.ReLU(),
            nn.Linear(gate_embed_dim, num_experts)
        )
        self._last_gates: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, scene_idx: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        s_emb = self.scene_embed(scene_idx)           
        gate_logits = self.gate(s_emb)                
        gates = F.softmax(gate_logits, dim=-1)        
        y = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        self._last_gates = gates.detach()
        return y

    def get_last_gates(self) -> Optional[torch.Tensor]:
        return self._last_gates


class IDEmbeddingLayer(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_scenes: int,
                 id_dim: int = 256, scene_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, id_dim)
        self.item_emb = nn.Embedding(num_items, id_dim)
        self.scene_emb = nn.Embedding(num_scenes, scene_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, scene_idx: torch.Tensor):
        u = self.dropout(self.user_emb(user_idx))
        i = self.dropout(self.item_emb(item_idx))
        s = self.dropout(self.scene_emb(scene_idx))
        return u, i, s


class ItemBlendModule(nn.Module):
    def __init__(self, fewshot_n: int):
        super().__init__()
        self.fewshot_n = int(fewshot_n)
        self._last_alpha = None

    def forward(self, item_id_emb: torch.Tensor, item_meta_emb: torch.Tensor, item_count: torch.Tensor) -> torch.Tensor:
        alpha = torch.log1p(item_count.float()) / np.log1p(self.fewshot_n)
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1) 
        item_final = alpha * item_id_emb + (1.0 - alpha) * item_meta_emb
        self._last_alpha = alpha.detach()
        return item_final

    def get_last_alpha(self):
        return self._last_alpha


class MetaEmbeddingGenerator(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, time_dim: int, meta_dim: int = 256,
                 attn_hidden: int = 128, dropout: float = 0.1, use_scene: bool = False,
                 num_scenes: int = 0, scene_emb_dim: int = 16):
        super().__init__()
        self.meta_dim = meta_dim
        self.use_scene = use_scene

        self.proj_text = nn.Sequential(nn.Linear(text_dim, meta_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_image = nn.Sequential(nn.Linear(image_dim, meta_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_time = nn.Sequential(nn.Linear(time_dim, meta_dim), nn.ReLU(), nn.Dropout(dropout))

        if self.use_scene:
            self.scene_emb = nn.Embedding(num_scenes, scene_emb_dim)
            attn_in = meta_dim + scene_emb_dim
        else:
            attn_in = meta_dim

        self.attn_mlp = nn.Sequential(nn.Linear(attn_in, attn_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(attn_hidden, 1))
        self._last_attention = None  

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor, time_vec: torch.Tensor, scene_idx: torch.Tensor = None) -> torch.Tensor:
        B = text_vec.size(0)
        z_text  = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time  = self.proj_time(time_vec)

        Z = torch.stack([z_text, z_image, z_time], dim=1)  

        if self.use_scene:
            s = self.scene_emb(scene_idx)                   
            s = s.unsqueeze(1).expand(B, 3, s.size(-1))     
            Z_attn_in = torch.cat([Z, s], dim=-1)           
        else:
            Z_attn_in = Z                                   

        attn_logits = self.attn_mlp(Z_attn_in).squeeze(-1)  
        attn_weights = F.softmax(attn_logits, dim=1)        
        meta_item_emb = torch.sum(Z * attn_weights.unsqueeze(-1), dim=1)

        self._last_attention = {
            "text": attn_weights[:, 0].detach(),
            "image": attn_weights[:, 1].detach(),
            "time": attn_weights[:, 2].detach()
        }
        return meta_item_emb

    def get_last_attention(self):
        return self._last_attention


class CTRHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) 
        )
    def forward(self, x):
        return self.net(x).squeeze(-1) 


# ---------------------------------------------------------
# MAIN MODEL: ColdFusionX (V6 Champion Architecture)
# ---------------------------------------------------------

class ColdFusionXModel(nn.Module):
    """
    The final architecture featuring:
      1. Meta-Embedding Generation for item warm-up.
      2. LayerNorm-stabilized Decoupled Projections.
      3. Explicit Coldness-Conditioned Feature Boosting.
      4. Transformer-based Cross-Modal Fusion.
      5. Scenario-Aware MMoE routing.
    """
    def __init__(self, fusion_config, meta_config, dropout=0.1, scene_dim=32):
        super().__init__()

        # Configs
        self.num_users = int(fusion_config["num_users"])
        self.num_items = int(fusion_config["num_items"])
        self.num_scenes = int(fusion_config["num_scenes"])

        self.text_dim = int(fusion_config["text_dim"])
        self.image_dim = int(fusion_config["image_dim"])
        self.time_dim = int(fusion_config["time_dim"])

        self.d_model = int(fusion_config["d_model"])
        self.hidden_dim = int(fusion_config["hidden_dim"])
        self.num_experts = int(fusion_config["num_experts"])
        
        self.fewshot_n = int(meta_config["FEWSHOT_N"])
        self.meta_dim = int(meta_config["meta_dim"])
        assert self.meta_dim == self.d_model, "Recommended: meta_dim should match d_model."

        # 1) ID Embeddings
        self.id_layer = IDEmbeddingLayer(
            num_users=self.num_users, num_items=self.num_items, num_scenes=self.num_scenes,
            id_dim=self.d_model, scene_dim=scene_dim, dropout=dropout
        )

        # 2) Meta embedding generator 
        self.meta_gen = MetaEmbeddingGenerator(
            text_dim=self.text_dim, image_dim=self.image_dim, time_dim=self.time_dim,
            meta_dim=self.meta_dim, attn_hidden=128, dropout=dropout, use_scene=False
        )

        # 3) Smooth α-blend
        self.blender = ItemBlendModule(fewshot_n=self.fewshot_n)

        # 4) Decoupled Projections (LayerNorm ensures Transformer inputs are well-scaled)
        self.proj_user  = nn.Sequential(nn.Linear(self.d_model + scene_dim, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(dropout))
        self.proj_item  = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(dropout))
        self.proj_text  = nn.Sequential(nn.Linear(self.text_dim, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(dropout))
        self.proj_image = nn.Sequential(nn.Linear(self.image_dim, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(dropout))
        self.proj_time  = nn.Sequential(nn.Linear(self.time_dim, self.d_model), nn.LayerNorm(self.d_model), nn.Dropout(dropout))

        # 5) Transformer Fusion (Replaces ModalityAttentionFusion)
        self.fusion = TransformerFusion(
            d_model=self.d_model, 
            nhead=4, 
            dropout=dropout
        )

        # 6) Scenario-aware routing
        self.mmoe = ScenarioAwareMMoE(
            d_model=self.d_model, hidden_dim=self.hidden_dim, num_experts=self.num_experts,
            num_scenes=self.num_scenes, gate_embed_dim=16, expert_dropout=dropout
        )

        # 7) CTR head
        self.head = CTRHead(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, dropout=dropout)

        # Diagnostics handles
        self._last_alpha = None
        self._last_coldness = None

    def _compute_alpha_coldness(self, item_count: torch.Tensor):
        alpha = torch.log1p(item_count.float()) / float(np.log1p(self.fewshot_n))
        alpha = torch.clamp(alpha, 0.0, 1.0)
        return alpha, 1.0 - alpha

    def forward(self, user_idx, item_idx, scene_idx, text_vec, image_vec, time_vec, cold_mask=None, item_count=None):
        # Safe defaults for inference
        if item_count is None: item_count = torch.zeros_like(item_idx)
            
        # --- A. Embeddings & Blending ---
        user_emb, item_id_emb, scene_emb = self.id_layer(user_idx, item_idx, scene_idx)
        item_meta_emb = self.meta_gen(text_vec, image_vec, time_vec)
        
        item_final = self.blender(item_id_emb, item_meta_emb, item_count)
        
        # Calculate Coldness for feature boosting
        alpha, coldness = self._compute_alpha_coldness(item_count)
        self._last_alpha = alpha.detach()
        self._last_coldness = coldness.detach()

        # --- B. Decoupled Projections ---
        z_user = self.proj_user(torch.cat([user_emb, scene_emb], dim=1))
        z_item = self.proj_item(item_final)

        # --- C. Feature Boosting (The Cold-Start Secret) ---
        # Amplifies text & image vectors when the item is cold
        boost = 1.0 + (2.0 * coldness.unsqueeze(1)) 
        
        z_text = self.proj_text(text_vec * boost)
        z_image = self.proj_image(image_vec * boost)
        z_time = self.proj_time(time_vec)

        # --- D. Transformer Fusion & Routing ---
        fused = self.fusion(z_user, z_item, z_text, z_image, z_time)
        routed = self.mmoe(fused, scene_idx)

        # --- E. Prediction ---
        return self.head(routed)

    # Diagnostics helpers
    def get_last_alpha(self):
        return self._last_alpha
        
    def get_last_coldness(self):
        return self._last_coldness

    def get_last_meta_attention(self):
        return self.meta_gen.get_last_attention()

    def get_last_gates(self):
        return self.mmoe.get_last_gates()