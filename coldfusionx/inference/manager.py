import torch
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
            dropout=0.0 # Strict eval mode
        )
        
        # Load Weights
        weights_path = os.path.join(artifact_dir, "model", "coldfusionx_final.pt")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Model weights not found at {weights_path}")
             
        # 1. Load the raw dictionary
        raw_state_dict = torch.load(weights_path, map_location=device)
        
        # 2. THE PATCH: Rename 'head.X' to 'head.net.X' dynamically
        patched_state_dict = {}
        for key, tensor in raw_state_dict.items():
            if key.startswith("head.") and not key.startswith("head.net."):
                new_key = key.replace("head.", "head.net.")
                patched_state_dict[new_key] = tensor
            else:
                patched_state_dict[key] = tensor

        # 3. Load the patched dictionary into the model
        self.model.load_state_dict(patched_state_dict)
        self.model.to(device)
        self.model.eval()
        print("[InferenceManager] System Ready.")


    def predict_demo_case(self, case_index: int, override_time_vec: torch.Tensor = None, override_scene_idx: torch.Tensor = None):
        """
        Runs full inference on a pre-loaded demo case.
        Allows UI to inject an 'override_time_vec' to simulate context shifts.
        """
        # Get Tensors from Feature Store
        batch = self.store.get_case_by_index(case_index, device=self.device)
        
        # 2. Allow UI to inject Time OR Scene overrides
        time_input = override_time_vec if override_time_vec is not None else batch["time_vec"]
        scene_input = override_scene_idx if override_scene_idx is not None else batch["scene_idx"]
        
        with torch.no_grad():
            logit = self.model(
                user_idx=batch["user_idx"],
                item_idx=batch["item_idx"],
                scene_idx=scene_input, # 3. Pass the overridden scene here!
                text_vec=batch["text_vec"],
                image_vec=batch["image_vec"],
                time_vec=time_input,
                item_count=batch["item_count"]
                )
            
            prob = torch.sigmoid(logit).item()
            
            # Extracting Explainability Diagnostics specifically for v12
            # The Meta-Generator now holds all our Text vs Image vs Time attention weights
            meta_attn = self.model.get_last_meta_attention()
            
            # Extract the SAM (Stochastic Alpha Masking) ratio from the blender
            try:
                alpha = self.model.blender._last_alpha.item()
            except AttributeError:
                alpha = 0.5 # Safe fallback

        # We map meta_attn to "fusion_attention" here so the UI doesn't break, 
        # as the UI is expecting that key to drive the "AI Insight" text box!
        return {
            "probability": prob,
            "fusion_attention": {k: v.item() for k, v in meta_attn.items()}, 
            "meta_attention": {k: v.item() for k, v in meta_attn.items()},
            "alpha": alpha
        }