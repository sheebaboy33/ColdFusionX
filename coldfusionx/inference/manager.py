import torch
import torch.nn.functional as F
import os
from ..config.loader import ConfigLoader
from ..data.feature_store import DemoFeatureStore
from ..model.coldfusionx_finalised import Proposed_ColdFusionX_v12

class InferenceManager:
    def __init__(self, artifact_dir: str, device: str = "cpu"):
        self.device = device
        self.artifact_dir = artifact_dir
        
        print(f"[InferenceManager] Loading artifacts from {artifact_dir}...")
        
        # Load Configs
        self.loader = ConfigLoader(artifact_dir)
        self.fusion_config, self.meta_config = self.loader.load_model_configs()
        
        # Initialize Data Store
        self.store = DemoFeatureStore(artifact_dir)
        
        # Initialize Model
        print("[InferenceManager] Building ColdFusionX Model...")
        self.model = Proposed_ColdFusionX_v12(
            fusion_config=self.fusion_config,
            meta_config=self.meta_config,
            dropout=0.0,
            scene_dim=32,
            attn_hidden=128  
        )
        
        # Load Weights with dynamic renaming patch
        weights_path = os.path.join(artifact_dir, "model", "coldfusionx_final.pt")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Model weights not found at {weights_path}")
             
        raw_state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        
        patched_state_dict = {}
        for key, tensor in raw_state_dict.items():
            if key.startswith("head.") and not key.startswith("head.net."):
                new_key = key.replace("head.", "head.net.")
                patched_state_dict[new_key] = tensor
            else:
                patched_state_dict[key] = tensor

        self.model.load_state_dict(patched_state_dict)
        self.model.to(device)
        self.model.eval()
        print("[InferenceManager] System Ready.")


    def predict_demo_case(self, case_index: int, override_time_vec: torch.Tensor = None, 
                          override_scene_idx: torch.Tensor = None, override_item_count: int = None,
                          mute_text: bool = False, mute_image: bool = False):
        """
        Runs full inference on a pre-loaded demo case.
        Includes advanced diagnostic overrides for X-Ray UI.
        """
        batch = self.store.get_case_by_index(case_index, device=self.device)
        
        # Context Overrides (For Campaign Optimizer)
        time_input = override_time_vec if override_time_vec is not None else batch["time_vec"]
        scene_input = override_scene_idx if override_scene_idx is not None else batch["scene_idx"]
        
        # Diagnostic Overrides (For Model X-Ray)
        if override_item_count is not None:
            batch["item_count"] = torch.tensor([override_item_count], dtype=torch.long, device=self.device)
            
        if mute_text:
            batch["text_vec"] = torch.zeros_like(batch["text_vec"])
        if mute_image:
            batch["image_vec"] = torch.zeros_like(batch["image_vec"])
        
        # Run Model
        with torch.no_grad():
            logit = self.model(
                user_idx=batch["user_idx"],
                item_idx=batch["item_idx"],
                scene_idx=scene_input,
                text_vec=batch["text_vec"],
                image_vec=batch["image_vec"],
                time_vec=time_input,
                item_count=batch["item_count"]
            )
            prob = torch.sigmoid(logit).item()
            
            # Extracting Explainability Diagnostics
            meta_attn = self.model.get_last_meta_attention()
            
            try:
                alpha = self.model.blender._last_alpha.item()
            except AttributeError:
                alpha = 0.5 

        return {
            "probability": prob,
            "fusion_attention": {k: v.item() for k, v in meta_attn.items()}, 
            "meta_attention": {k: v.item() for k, v in meta_attn.items()},
            "alpha": alpha
        }

    def get_similar_cases(self, case_index: int, top_k: int = 3):
        """
        Performs Cosine Similarity search across the entire feature store 
        using concatenated Text and Image embeddings.
        """
        # 1. Fetch the target ad's multi-modal representation
        target_text = torch.tensor(self.store.text_emb[case_index], dtype=torch.float32)
        target_image = self.store.image_emb[case_index].clone().detach().to(torch.float32)
        target_vec = torch.cat([target_text, target_image], dim=-1).unsqueeze(0) # Shape: [1, D]

        # 2. Fetch ALL ads in the database
        all_text = torch.tensor(self.store.text_emb, dtype=torch.float32)
        all_image = self.store.image_emb.clone().detach().to(torch.float32)
        all_vecs = torch.cat([all_text, all_image], dim=-1) # Shape: [N, D]

        # 3. Calculate Cosine Similarity across the whole matrix
        sims = F.cosine_similarity(target_vec, all_vecs, dim=-1)

        # 4. Filter out the target ad itself and get Top K
        sims[case_index] = -1.0 
        top_scores, top_indices = torch.topk(sims, top_k)

        # 5. Package the results for the UI
        results = []
        for score, idx in zip(top_scores, top_indices):
            meta = self.store.get_raw_metadata(idx.item())
            meta['similarity_score'] = score.item()
            results.append(meta)

        return results