import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

# SUB-MODULES (The Building Blocks) - Extracted from Notebooks

class ModalityAttentionFusion(nn.Module):
    """
    Projects {id, text, image, time} into a shared d_model space and applies
    per-sample attention over modalities.
    """
    def __init__(
        self,
        id_dim: int,
        text_dim: int,
        image_dim: int,
        time_dim: int,
        d_model: int = 256,
        attn_hidden: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        # Projections into shared d_model
        self.proj_id = nn.Sequential(
            nn.Linear(id_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_image = nn.Sequential(
            nn.Linear(image_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proj_time = nn.Sequential(
            nn.Linear(time_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention MLP produces 1 scalar score per modality per sample
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

        self._last_attention: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        id_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_vec: torch.Tensor,
        time_vec: torch.Tensor
    ) -> torch.Tensor:

        # Project each modality to (B, d_model)
        z_id = self.proj_id(id_vec)
        z_text = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time = self.proj_time(time_vec)

        # Stack: (B, M=4, d_model)
        Z = torch.stack([z_id, z_text, z_image, z_time], dim=1)

        # Compute attention logits per modality: (B, 4, 1) -> (B, 4)
        logits = self.attn_mlp(Z).squeeze(-1)

        # Softmax over modalities -> (B, 4)
        attn = F.softmax(logits, dim=1)

        # Weighted sum: (B, 4, 1) * (B, 4, d_model) -> (B, d_model)
        fused = torch.sum(attn.unsqueeze(-1) * Z, dim=1)

        # Save for diagnostics (detach for safe logging)
        self._last_attention = {
            "id": attn[:, 0].detach(),
            "text": attn[:, 1].detach(),
            "image": attn[:, 2].detach(),
            "time": attn[:, 3].detach(),
        }

        return fused

    def get_last_attention(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_attention


class ScenarioAwareMMoE(nn.Module):
    """
    Scenario-aware Mixture-of-Experts (MMoE).
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        num_scenes: int,
        gate_embed_dim: int = 16,
        expert_dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_scenes = num_scenes

        # Experts
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

        # Scene embedding used for gating
        self.scene_embed = nn.Embedding(num_scenes, gate_embed_dim)

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(gate_embed_dim, gate_embed_dim),
            nn.ReLU(),
            nn.Linear(gate_embed_dim, num_experts)
        )

        self._last_gates: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, scene_idx: torch.Tensor) -> torch.Tensor:
        # Expert outputs: (B, E, hidden_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Compute gate weights
        s_emb = self.scene_embed(scene_idx)           # (B, gate_embed_dim)
        gate_logits = self.gate(s_emb)                # (B, num_experts)
        gates = F.softmax(gate_logits, dim=-1)        # (B, num_experts)

        # Combine experts
        y = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)

        self._last_gates = gates.detach()
        return y

    def get_last_gates(self) -> Optional[torch.Tensor]:
        return self._last_gates


class IDEmbeddingLayer(nn.Module):
    """
    Embedding tables for user/item/scene.
    """
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
    """
    Smooth alpha blending based on TRAIN item frequency.
    """
    def __init__(self, fewshot_n: int):
        super().__init__()
        self.fewshot_n = int(fewshot_n)
        self._last_alpha = None

    def forward(
        self,
        item_id_emb: torch.Tensor,     # (B, D)
        item_meta_emb: torch.Tensor,   # (B, D)
        item_count: torch.Tensor       # (B,)
    ) -> torch.Tensor:

        # Smooth alpha: log-scaled by threshold
        alpha = torch.log1p(item_count.float()) / np.log1p(self.fewshot_n)
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1)  # (B,1)

        item_final = alpha * item_id_emb + (1.0 - alpha) * item_meta_emb
        self._last_alpha = alpha.detach()
        return item_final

    def get_last_alpha(self):
        return self._last_alpha


class MetaEmbeddingGenerator(nn.Module):
    """
    Generates meta_item_emb from non-ID modalities.
    """
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        time_dim: int,
        meta_dim: int = 256,
        attn_hidden: int = 128,
        dropout: float = 0.1,
        use_scene: bool = False,
        num_scenes: int = 0,
        scene_emb_dim: int = 16
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.use_scene = use_scene

        # Projections into shared space
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_image = nn.Sequential(
            nn.Linear(image_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.proj_time = nn.Sequential(
            nn.Linear(time_dim, meta_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Optional scene conditioning
        if self.use_scene:
            assert num_scenes > 0, "num_scenes must be provided if use_scene=True"
            self.scene_emb = nn.Embedding(num_scenes, scene_emb_dim)
            attn_in = meta_dim + scene_emb_dim
        else:
            attn_in = meta_dim

        # Attention scoring network
        self.attn_mlp = nn.Sequential(
            nn.Linear(attn_in, attn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1)
        )

        self._last_attention = None

    def forward(
        self,
        text_vec: torch.Tensor,
        image_vec: torch.Tensor,
        time_vec: torch.Tensor,
        scene_idx: torch.Tensor = None
    ) -> torch.Tensor:
        B = text_vec.size(0)

        # Project modalities
        z_text  = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time  = self.proj_time(time_vec)

        # Stack into (B, M=3, meta_dim)
        Z = torch.stack([z_text, z_image, z_time], dim=1)

        if self.use_scene:
            assert scene_idx is not None, "scene_idx required when use_scene=True"
            s = self.scene_emb(scene_idx)                   
            s = s.unsqueeze(1).expand(B, 3, s.size(-1))     
            Z_attn_in = torch.cat([Z, s], dim=-1)           
        else:
            Z_attn_in = Z                                   

        # Compute attention logits
        attn_logits = self.attn_mlp(Z_attn_in).squeeze(-1)  # (B, 3)
        attn_weights = F.softmax(attn_logits, dim=1)        # (B, 3)

        # Weighted sum
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


# MAIN MODEL

class ColdFusionXModel(nn.Module):
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

        self.meta_dim = int(meta_config["meta_dim"])
        assert self.meta_dim == self.d_model, "Recommended: meta_dim should match d_model."

        # 1) ID embeddings
        self.id_layer = IDEmbeddingLayer(
            num_users=self.num_users,
            num_items=self.num_items,
            num_scenes=self.num_scenes,
            id_dim=self.d_model,
            scene_dim=scene_dim,
            dropout=dropout
        )

        # 2) Meta embedding generator
        self.meta_gen = MetaEmbeddingGenerator(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            time_dim=self.time_dim,
            meta_dim=self.meta_dim,
            attn_hidden=128,
            dropout=dropout,
            use_scene=False
        )

        # 3) α-blend
        self.blender = ItemBlendModule(fewshot_n=int(meta_config["FEWSHOT_N"]))

        # Project concat([user_emb, item_final, scene_emb]) -> d_model
        self.id_proj = nn.Sequential(
            nn.Linear(self.d_model + self.d_model + scene_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 4) Fusion
        self.fusion = ModalityAttentionFusion(
            id_dim=self.d_model,
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            time_dim=self.time_dim,
            d_model=self.d_model,
            attn_hidden=128,
            dropout=dropout
        )

        # 5) Scenario-aware routing
        self.mmoe = ScenarioAwareMMoE(
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            num_scenes=self.num_scenes,
            gate_embed_dim=16,
            expert_dropout=dropout
        )

        # 6) CTR head
        self.head = CTRHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout
        )

        # Diagnostics handles
        self._last_alpha = None

    def forward(self, user_idx, item_idx, scene_idx, text_vec, image_vec, time_vec, cold_mask=None, item_count=None):
        """
        Main Forward Pass.
        
        Note: cold_mask is kept for compatibility but not strictly used in blend (alpha uses item_count).
        item_count defaults to 0 if not provided (treating as Pure Cold Start).
        """
        
        # Safe defaults for inference if not provided
        if item_count is None:
            item_count = torch.zeros_like(item_idx)

        # ID embeddings
        user_emb, item_id_emb, scene_emb = self.id_layer(
            user_idx,
            item_idx,
            scene_idx
        )

        # Meta item embedding
        item_meta_emb = self.meta_gen(
            text_vec=text_vec,
            image_vec=image_vec,
            time_vec=time_vec
        )

        # Unified item embedding (smooth α blend)
        item_final = self.blender(
            item_id_emb=item_id_emb,
            item_meta_emb=item_meta_emb,
            item_count=item_count
        )
        self._last_alpha = self.blender.get_last_alpha()

        # Build id_vec
        id_concat = torch.cat([user_emb, item_final, scene_emb], dim=1)
        id_vec = self.id_proj(id_concat)

        # Fusion + routing
        fused = self.fusion(
            id_vec=id_vec,
            text_vec=text_vec,
            image_vec=image_vec,
            time_vec=time_vec
        )
        routed = self.mmoe(fused, scene_idx)

        # Logit
        logit = self.head(routed)
        return logit

    # Diagnostics helpers
    def get_last_alpha(self):
        return self._last_alpha

    def get_last_fusion_attention(self):
        return self.fusion.get_last_attention()

    def get_last_meta_attention(self):
        return self.meta_gen.get_last_attention()

    def get_last_gates(self):
        return self.mmoe.get_last_gates()