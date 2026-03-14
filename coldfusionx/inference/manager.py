import torch
import os
from ..config.loader import ConfigLoader
from ..data.feature_store import DemoFeatureStore
from ..model.coldfusionx_finalised import ColdFusionXModel

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
        self.model = ColdFusionXModel(
            fusion_config=self.fusion_config,
            meta_config=self.meta_config,
            dropout=0.0 # No dropout during inference
        )
        
        # Load Weights
        weights_path = os.path.join(artifact_dir, "model", "coldfusionx_final.pt")
        if not os.path.exists(weights_path):
             raise FileNotFoundError(f"Model weights not found at {weights_path}")
             
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        print("[InferenceManager] System Ready.")


    def predict_demo_case(self, case_index: int):
        """
        Runs full inference on a pre-loaded demo case.
        Returns: CTR Probability, Attention Weights, Gate Weights
        """
        # Get Tensors
        batch = self.store.get_case_by_index(case_index, device=self.device)
        
        # Run Model (No Grad)
        with torch.no_grad():
            logit = self.model(
                user_idx=batch["user_idx"],
                item_idx=batch["item_idx"],
                scene_idx=batch["scene_idx"],
                text_vec=batch["text_vec"],
                image_vec=batch["image_vec"],
                time_vec=batch["time_vec"],
                item_count=batch["item_count"]
            )
            prob = torch.sigmoid(logit).item()
            
            # Extract Explainability Data
            fusion_attn = self.model.get_last_fusion_attention() # {text: ..., image: ...}
            meta_attn = self.model.get_last_meta_attention()
            gates = self.model.get_last_gates()
            alpha = self.model.get_last_alpha()

        # Return clean dictionary
        return {
            "probability": prob,
            "fusion_attention": {k: v.item() for k, v in fusion_attn.items()},
            "meta_attention": {k: v.item() for k, v in meta_attn.items()},
            "gates": gates.cpu().numpy().tolist()[0],
            "alpha": alpha.item()
        }