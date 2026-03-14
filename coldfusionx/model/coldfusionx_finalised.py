import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IDEmbeddingLayer(nn.Module):
    """
    Learnable embedding tables for user IDs, item IDs, and scenario IDs.

    This module provides the discrete embedding backbone used by ColdFusionX.
    It maps categorical identifiers into dense trainable vectors that can be
    fused later with metadata-based representations.

    Outputs
    -------
    user_emb : torch.Tensor
        User embedding tensor of shape (B, id_dim).
    item_emb : torch.Tensor
        Item embedding tensor of shape (B, id_dim).
    scene_emb : torch.Tensor
        Scenario embedding tensor of shape (B, scene_dim).

    Parameters
    ----------
    num_users : int
        Number of unique users.
    num_items : int
        Number of unique items.
    num_scenes : int
        Number of unique scenarios.
    id_dim : int, optional
        Embedding dimension used for user and item IDs.
    scene_dim : int, optional
        Embedding dimension used for scenario IDs.
    dropout : float, optional
        Dropout probability applied to the embedding outputs.

    Notes
    -----
    Small-normal initialization is used to keep the initial embedding scale
    controlled, which can help stabilize early optimization and reduce the
    risk of large initial activations dominating the multimodal pipeline.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_scenes: int,
        id_dim: int = 256,
        scene_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.user_emb = nn.Embedding(num_users, id_dim)
        self.item_emb = nn.Embedding(num_items, id_dim)
        self.scene_emb = nn.Embedding(num_scenes, scene_dim)

        # Controlled small-scale initialization helps keep the embedding
        # magnitudes stable during the early training phase.
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.scene_emb.weight, mean=0.0, std=0.01)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        scene_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve dense embeddings for user, item, and scenario IDs.

        Parameters
        ----------
        user_idx : torch.Tensor
            User ID indices of shape (B,).
        item_idx : torch.Tensor
            Item ID indices of shape (B,).
        scene_idx : torch.Tensor
            Scenario ID indices of shape (B,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - user embeddings of shape (B, id_dim)
            - item embeddings of shape (B, id_dim)
            - scenario embeddings of shape (B, scene_dim)
        """
        user_emb = self.dropout(self.user_emb(user_idx))
        item_emb = self.dropout(self.item_emb(item_idx))
        scene_emb = self.dropout(self.scene_emb(scene_idx))

        return user_emb, item_emb, scene_emb
    

class ResidualTemporalFiLM(nn.Module):
    """
    Residual temporal FiLM-style modulation for text and image embeddings.

    This module uses time features to generate feature-wise affine modulation
    parameters for the text and image representations. A residual formulation
    is used so that the original modality embedding is preserved while still
    allowing time-dependent adaptation.

    For each modality:
        adapted = original + (original * gamma) + beta

    where gamma and beta are generated from the time vector.

    Parameters
    ----------
    time_dim : int
        Dimensionality of the time feature vector.
    text_dim : int
        Dimensionality of the text embedding.
    image_dim : int
        Dimensionality of the image embedding.
    dropout : float, optional
        Dropout probability used in the FiLM parameter generators.
    """

    def __init__(
        self,
        time_dim: int,
        text_dim: int,
        image_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.film_text = nn.Sequential(
            nn.Linear(time_dim, text_dim * 2),
            nn.LayerNorm(text_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.film_image = nn.Sequential(
            nn.Linear(time_dim, image_dim * 2),
            nn.LayerNorm(image_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Small initialization helps keep early modulation stable.
        nn.init.normal_(self.film_text[0].weight, std=0.01)
        nn.init.normal_(self.film_image[0].weight, std=0.01)

    def forward(
        self,
        time_vec: torch.Tensor,
        text_vec: torch.Tensor,
        image_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal modulation to text and image embeddings.

        Parameters
        ----------
        time_vec : torch.Tensor
            Time features of shape (B, time_dim).
        text_vec : torch.Tensor
            Text embeddings of shape (B, text_dim).
        image_vec : torch.Tensor
            Image embeddings of shape (B, image_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Time-adapted text and image embeddings.
        """
        text_params = self.film_text(time_vec)
        image_params = self.film_image(time_vec)

        gamma_t, beta_t = torch.chunk(text_params, 2, dim=-1)
        gamma_i, beta_i = torch.chunk(image_params, 2, dim=-1)

        # Bound multiplicative modulation to keep updates stable.
        gamma_t = torch.tanh(gamma_t)
        gamma_i = torch.tanh(gamma_i)

        text_adapted = text_vec + (text_vec * gamma_t) + beta_t
        image_adapted = image_vec + (image_vec * gamma_i) + beta_i

        return text_adapted, image_adapted
    

class MetaEmbeddingGenerator(nn.Module):
    """
    Generate a metadata-driven item embedding from text, image, and time inputs.

    This module first projects each modality into a common latent space and then
    applies learnable modality-wise gating to produce a single metadata-based
    item representation. When enabled, scenario embeddings are concatenated into
    the gating network so that modality weighting can adapt across scenes.

    Important note:
    ----------------
    The weighting mechanism uses sigmoid gates rather than softmax-normalized
    attention. This means each modality receives an independent importance score
    rather than competing under a sum-to-one constraint. In the thesis, this is
    better described as modality gating or attention-like weighting.

    Parameters
    ----------
    text_dim : int
        Dimensionality of the text embedding.
    image_dim : int
        Dimensionality of the image embedding.
    time_dim : int
        Dimensionality of the time feature vector.
    meta_dim : int, optional
        Output dimension of the metadata item embedding.
    attn_hidden : int, optional
        Hidden size of the modality gating MLP.
    dropout : float, optional
        Dropout probability.
    use_scene : bool, optional
        Whether to condition modality gates on scenario embeddings.
    num_scenes : int, optional
        Number of scenarios. Required if use_scene=True.
    scene_emb_dim : int, optional
        Scenario embedding size.
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
        scene_emb_dim: int = 16,
    ) -> None:
        super().__init__()

        self.meta_dim = meta_dim
        self.use_scene = use_scene
        self.num_modalities = 3
        self._last_attention: Optional[Dict[str, torch.Tensor]] = None

        # Progressive modality compression into a shared meta space.
        self.proj_text = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, meta_dim),
            nn.LayerNorm(meta_dim),
        )

        self.proj_image = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, meta_dim),
            nn.LayerNorm(meta_dim),
        )

        self.proj_time = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, meta_dim),
            nn.LayerNorm(meta_dim),
        )

        if self.use_scene:
            if num_scenes <= 0:
                raise ValueError("num_scenes must be > 0 when use_scene=True.")
            self.scene_emb = nn.Embedding(num_scenes, scene_emb_dim)
            nn.init.normal_(self.scene_emb.weight, std=0.01)
            gate_in_dim = meta_dim + scene_emb_dim
        else:
            gate_in_dim = meta_dim

        self.attn_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, attn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden, 1),
        )

    def forward(
        self,
        text_vec: torch.Tensor,
        image_vec: torch.Tensor,
        time_vec: torch.Tensor,
        scene_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build a metadata-based item embedding by gating text, image, and time.

        Parameters
        ----------
        text_vec : torch.Tensor
            Text embeddings of shape (B, text_dim).
        image_vec : torch.Tensor
            Image embeddings of shape (B, image_dim).
        time_vec : torch.Tensor
            Time features of shape (B, time_dim).
        scene_idx : Optional[torch.Tensor]
            Scenario IDs of shape (B,). Required if use_scene=True.

        Returns
        -------
        torch.Tensor
            Metadata item embedding of shape (B, meta_dim).
        """
        if self.use_scene and scene_idx is None:
            raise ValueError("scene_idx must be provided when use_scene=True.")

        batch_size = text_vec.size(0)

        z_text = self.proj_text(text_vec)
        z_image = self.proj_image(image_vec)
        z_time = self.proj_time(time_vec)

        # Shape: (B, 3, meta_dim)
        stacked_modalities = torch.stack([z_text, z_image, z_time], dim=1)

        if self.use_scene:
            scene_context = self.scene_emb(scene_idx)  # (B, scene_emb_dim)
            scene_context = scene_context.unsqueeze(1).expand(
                batch_size,
                self.num_modalities,
                scene_context.size(-1),
            )
            gate_input = torch.cat([stacked_modalities, scene_context], dim=-1)
        else:
            gate_input = stacked_modalities

        # Independent modality gates in [0, 1].
        gate_logits = self.attn_mlp(gate_input).squeeze(-1)
        gate_weights = torch.sigmoid(gate_logits)

        # Weighted sum over modalities.
        meta_item_emb = torch.sum(
            stacked_modalities * gate_weights.unsqueeze(-1),
            dim=1,
        )

        self._last_attention = {
            "text": gate_weights[:, 0].detach(),
            "image": gate_weights[:, 1].detach(),
            "time": gate_weights[:, 2].detach(),
        }

        return meta_item_emb

    def get_last_attention(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the most recent modality gating weights."""
        return self._last_attention
    

class StochasticItemBlendModule(nn.Module):
    """
    Blend ID-based and metadata-based item embeddings using causal item history.

    The blending factor alpha is derived from the item's prior training exposure
    count. Items with little prior history rely more heavily on the metadata
    embedding, while mature items rely more heavily on the ID embedding.

    During training, stochastic masking is applied to alpha to reduce over-
    dependence on ID embeddings and to encourage stronger metadata learning.

    Parameters
    ----------
    fewshot_n : int
        Threshold used to normalize the exposure count into the [0, 1] range.
    mask_prob : float, optional
        Probability of masking the learned alpha during training.
    """

    def __init__(self, fewshot_n: int, mask_prob: float = 0.3) -> None:
        super().__init__()

        if fewshot_n <= 0:
            raise ValueError("fewshot_n must be > 0.")

        self.fewshot_n = int(fewshot_n)
        self.mask_prob = mask_prob
        self._last_alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        item_id_emb: torch.Tensor,
        item_meta_emb: torch.Tensor,
        item_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend ID and metadata item embeddings.

        Parameters
        ----------
        item_id_emb : torch.Tensor
            ID-based item embedding of shape (B, D).
        item_meta_emb : torch.Tensor
            Metadata-based item embedding of shape (B, D).
        item_count : torch.Tensor
            Prior causal training exposure count of shape (B,).

        Returns
        -------
        torch.Tensor
            Final blended item embedding of shape (B, D).
        """
        alpha = torch.log1p(item_count.float()) / math.log1p(self.fewshot_n)
        alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1)

        if self.training:
            random_mask = (
                torch.rand(alpha.size(0), 1, device=alpha.device) > self.mask_prob
            ).float()
            alpha_used = alpha * random_mask
        else:
            alpha_used = alpha

        item_final = alpha_used * item_id_emb + (1.0 - alpha_used) * item_meta_emb
        self._last_alpha = alpha_used.detach()

        return item_final

    def get_last_alpha(self) -> Optional[torch.Tensor]:
        """Return the most recent alpha values used for item blending."""
        return self._last_alpha


class AdaptedMMoEHead(nn.Module):
    """
    Scenario-conditioned MMoE-style prediction head.

    This module is an adapted Mixture-of-Experts (MMoE) routing head designed
    for a single CTR prediction task under multiple scenarios. It should be
    described as "MMoE-style" or "adapted MMoE-style" rather than as a literal
    reproduction of canonical multi-task MMoE.

    Architecture:
        1. A pool of shared experts transforms the fused representation.
        2. Each scenario has its own gate that computes routing weights over
           the shared experts using the fused input representation.
        3. Each scenario also has its own prediction tower that maps the
           mixed expert representation to a single logit.

    Parameters
    ----------
    d_model : int
        Dimensionality of the fused input representation.
    hidden_dim : int
        Hidden size used inside each expert.
    num_experts : int
        Number of shared experts.
    num_scenes : int
        Number of distinct scenarios.
    dropout : float, optional
        Dropout probability used inside experts.

    Input
    -----
    x : torch.Tensor
        Tensor of shape (B, d_model), where B is batch size.
    scene_idx : torch.Tensor
        Tensor of shape (B,) containing scenario IDs.

    Output
    ------
    torch.Tensor
        Tensor of shape (B, 1) containing logits.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        num_scenes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_experts = num_experts
        self.num_scenes = num_scenes
        self.expert_out_dim = hidden_dim // 2

        # Shared experts: each expert maps fused multimodal features
        # into a shared latent representation.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, self.expert_out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

        # Scenario-specific gates: each scenario learns how to mix the shared
        # experts based on the current fused representation x.
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, self.expert_out_dim),
                    nn.ReLU(),
                    nn.Linear(self.expert_out_dim, num_experts),
                )
                for _ in range(num_scenes)
            ]
        )

        # Scenario-specific output towers: after expert mixing, each scenario
        # uses its own linear head to produce the final CTR logit.
        self.towers = nn.ModuleList(
            [nn.Linear(self.expert_out_dim, 1) for _ in range(num_scenes)]
        )

    def forward(self, x: torch.Tensor, scene_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for scenario-conditioned expert routing.

        Parameters
        ----------
        x : torch.Tensor
            Fused representation of shape (B, d_model).
        scene_idx : torch.Tensor
            Scenario IDs of shape (B,).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        # Compute all expert outputs once for the full batch.
        # Shape: (B, num_experts, expert_out_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Match dtype/device to expert outputs, which helps mixed precision runs.
        y = torch.zeros(x.size(0), 1, dtype=expert_outputs.dtype, device=x.device)

        # Process each scenario subset independently.
        for scene_id in range(self.num_scenes):
            mask = scene_idx == scene_id
            if not mask.any():
                continue

            x_scene = x[mask]  # (B_scene, d_model)

            # Scenario gate produces routing weights across shared experts.
            # Shape before unsqueeze: (B_scene, num_experts)
            gate_weights = F.softmax(self.gates[scene_id](x_scene), dim=-1)

            # Convert to batch matrix-multiplication format:
            # (B_scene, 1, num_experts) x (B_scene, num_experts, expert_out_dim)
            # -> (B_scene, 1, expert_out_dim)
            mixed_rep = torch.bmm(
                gate_weights.unsqueeze(1),
                expert_outputs[mask],
            ).squeeze(1)

            # Scenario-specific prediction tower.
            y[mask] = self.towers[scene_id](mixed_rep)

        return y


class Proposed_ColdFusionX_ThesisEdition(nn.Module):
    """
    Final ColdFusionX thesis architecture with adapted scenario-aware routing.

    The model combines:
        1. Residual temporal FiLM modulation for time-sensitive text/image adaptation
        2. Scene-aware metadata embedding generation from text, image, and time
        3. Stop-gradient cosine alignment between metadata and ID item embeddings
        4. Stochastic item blending based on causal train-history exposure counts
        5. Transformer-based multimodal fusion over user/item/text/image/time tokens
        6. An adapted MMoE-style scenario-conditioned prediction head

    Notes
    -----
    - This model assumes `IDEmbeddingLayer` is defined elsewhere and returns:
          user_emb, item_id_emb, scene_emb
    - The final fusion representation is formed by flattening the transformer
      token outputs rather than mean-pooling them. This preserves token-position-
      specific multimodal information before expert routing.

    Parameters
    ----------
    fusion_config : dict
        Configuration dictionary containing model sizes and dataset counts.
    meta_config : dict
        Configuration dictionary containing metadata/blending parameters.
    dropout : float, optional
        Global dropout probability used across modules.
    scene_dim : int, optional
        Dimensionality of the scene embedding returned by IDEmbeddingLayer.
    attn_hidden : int, optional
        Hidden size used by the metadata gating MLP.
    """

    def __init__(
        self,
        fusion_config: dict,
        meta_config: dict,
        dropout: float = 0.1,
        scene_dim: int = 32,
        attn_hidden: int = 128,
    ) -> None:
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
        self.num_experts = int(fusion_config.get("num_experts", 4))

        # External embedding layer expected to produce:
        # user_emb:  (B, id_dim)
        # item_emb:  (B, id_dim)
        # scene_emb: (B, scene_dim)
        self.id_layer = IDEmbeddingLayer(
            self.num_users,
            self.num_items,
            self.num_scenes,
            id_dim=self.id_dim,
            scene_dim=scene_dim,
            dropout=dropout,
        )

        self.temporal_film = ResidualTemporalFiLM(
            time_dim=self.time_dim,
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            dropout=dropout,
        )

        self.meta_gen = MetaEmbeddingGenerator(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            time_dim=self.time_dim,
            meta_dim=self.id_dim,
            attn_hidden=attn_hidden,
            dropout=dropout,
            use_scene=True,
            num_scenes=self.num_scenes,
        )

        self.blender = StochasticItemBlendModule(
            fewshot_n=self.fewshot_n,
            mask_prob=0.3,
        )

        # Project all modalities/tokens into a common transformer dimension.
        self.proj_user = nn.Sequential(
            nn.Linear(self.id_dim + scene_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.proj_item = nn.Sequential(
            nn.Linear(self.id_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.proj_text = nn.Sequential(
            nn.Linear(self.text_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.proj_image = nn.Sequential(
            nn.Linear(self.image_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.proj_time = nn.Sequential(
            nn.Linear(self.time_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=self.d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )

        # Token order is fixed as:
        # [user, item, text, image, time]
        self.seq_len = 5
        self.flattened_dim = self.d_model * self.seq_len

        self.mmoe = AdaptedMMoEHead(
            d_model=self.flattened_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            num_scenes=self.num_scenes,
            dropout=dropout,
        )

        self._last_align_loss = torch.tensor(0.0)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        scene_idx: torch.Tensor,
        text_vec: torch.Tensor,
        image_vec: torch.Tensor,
        time_vec: torch.Tensor,
        cold_mask: Optional[torch.Tensor] = None,
        item_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for ColdFusionX.

        Parameters
        ----------
        user_idx : torch.Tensor
            User ID indices of shape (B,).
        item_idx : torch.Tensor
            Item ID indices of shape (B,).
        scene_idx : torch.Tensor
            Scenario ID indices of shape (B,).
        text_vec : torch.Tensor
            Text embeddings of shape (B, text_dim).
        image_vec : torch.Tensor
            Image embeddings of shape (B, image_dim).
        time_vec : torch.Tensor
            Time features of shape (B, time_dim).
        cold_mask : Optional[torch.Tensor]
            Unused in this forward path, retained for training-loop compatibility.
        item_count : Optional[torch.Tensor]
            Causal prior train-history exposure counts of shape (B,).

        Returns
        -------
        torch.Tensor
            Logits of shape (B,).
        """
        del cold_mask  # Retained only for compatibility with existing loaders/training code.

        if item_count is None:
            item_count = torch.zeros_like(item_idx)

        # 1. Time-adapt text and image embeddings.
        text_adapted, image_adapted = self.temporal_film(
            time_vec=time_vec,
            text_vec=text_vec,
            image_vec=image_vec,
        )

        # 2. Obtain base ID embeddings and scene embedding.
        user_emb, item_id_emb, scene_emb = self.id_layer(user_idx, item_idx, scene_idx)

        # 3. Generate metadata-driven item embedding.
        item_meta_emb = self.meta_gen(
            text_vec=text_adapted,
            image_vec=image_adapted,
            time_vec=time_vec,
            scene_idx=scene_idx,
        )

        # 4. Stop-gradient cosine alignment between metadata and item ID embeddings.
        # Alignment is weighted by causal item maturity, so more mature items
        # contribute more strongly to the alignment objective.
        if self.training:
            cos_sim = F.cosine_similarity(item_meta_emb, item_id_emb.detach(), dim=-1)
            maturity = torch.log1p(item_count.float()) / math.log1p(self.fewshot_n)
            maturity = torch.clamp(maturity, 0.0, 1.0)
            self._last_align_loss = torch.mean(maturity * (1.0 - cos_sim))
        else:
            self._last_align_loss = torch.tensor(0.0, device=user_idx.device)

        # 5. Blend item ID and metadata embeddings using causal item history.
        item_final = self.blender(
            item_id_emb=item_id_emb,
            item_meta_emb=item_meta_emb,
            item_count=item_count,
        )

        # 6. Project each token into a shared transformer space.
        z_user = self.proj_user(torch.cat([user_emb, scene_emb], dim=1))
        z_item = self.proj_item(item_final)
        z_text = self.proj_text(text_adapted)
        z_image = self.proj_image(image_adapted)
        z_time = self.proj_time(time_vec)

        # 7. Build multimodal token sequence.
        # Shape: (B, 5, d_model)
        seq = torch.stack([z_user, z_item, z_text, z_image, z_time], dim=1)

        # 8. Contextual multimodal fusion.
        # Instead of mean pooling, keep all token positions by flattening.
        fused = self.transformer(seq).flatten(start_dim=1)

        # 9. Scenario-conditioned routing head.
        output = self.mmoe(fused, scene_idx).squeeze(-1)

        return output

    def get_align_loss(self) -> torch.Tensor:
        """
        Return the most recent auxiliary alignment loss.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the latest alignment loss.
        """
        return self._last_align_loss