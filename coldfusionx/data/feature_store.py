import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

class DemoFeatureStore:
    """
    Loads the static 'Demo Subset' data (pre-calculated tensors)
    to allow for instant playback of specific case studies.
    """
    def __init__(self, artifact_dir: str):
        self.data_dir = os.path.join(artifact_dir, "data")
        
        # 1. Load Metadata (Pandas)
        df_path = os.path.join(self.data_dir, "demo_rows.parquet")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Demo Data not found at {df_path}. Did artifacts unzip correctly?")
        
        self.demo_df = pd.read_parquet(df_path).reset_index(drop=True)
        
        # 2. Load Tensors (CPU)
        # We load these entirely into memory because they are small (only ~395 rows)
        self.text_emb = np.load(os.path.join(self.data_dir, "demo_text_embeddings.npy"))
        self.time_feats = torch.load(os.path.join(self.data_dir, "demo_time_features.pt"), map_location="cpu")
        self.image_emb = torch.load(os.path.join(self.data_dir, "demo_image_embeddings.pt"), map_location="cpu")
        
        # 3. Load Indices to map Global Row ID -> Local Demo Index
        # This helps if we look up by original dataset index
        self.global_indices = np.load(os.path.join(self.data_dir, "demo_indices.npy"))
        self.global_to_local = {gid: loc for loc, gid in enumerate(self.global_indices)}


    def list_cases(self):
        """Returns a summary dataframe for the UI dropdown"""
        # Return cols: index, user_id, item_id, scene_id, timestamp
        return self.demo_df[["user_id", "item_id", "scene_id", "timestamp"]].copy()


    def get_case_by_index(self, local_index: int, device: str = "cpu") -> Dict[str, Any]:
        """
        Retrieves the exact tensors for a specific row in the demo dataset.
        Ready for Model Injection.
        """
        if local_index >= len(self.demo_df):
            raise ValueError(f"Index {local_index} out of bounds.")

        row = self.demo_df.iloc[local_index]
        
        # Prepare Batch (Add Batch Dimension [1, ...])
        batch = {
            "user_idx": torch.tensor([row["user_id_idx"]], dtype=torch.long, device=device),
            "item_idx": torch.tensor([row["item_id_idx"]], dtype=torch.long, device=device),
            "scene_idx": torch.tensor([row["scene_idx"]], dtype=torch.long, device=device),
            
            # Features
            "text_vec": torch.tensor(self.text_emb[local_index], dtype=torch.float32, device=device).unsqueeze(0),
            "image_vec": self.image_emb[local_index].unsqueeze(0).to(device),
            "time_vec": self.time_feats[local_index].unsqueeze(0).to(device),
            
            # Metadata for logic
            "item_count": torch.tensor([0], dtype=torch.long, device=device), # Default to 0 for demo (treat as Cold/Low info) or load from DF if available
            "cold_mask": torch.tensor([True], dtype=torch.bool, device=device) 
        }
        
        # Logic check: If item_train_count exists in DF, use it
        if "item_train_count" in row:
             batch["item_count"] = torch.tensor([row["item_train_count"]], dtype=torch.long, device=device)
        
        return batch


    def get_raw_metadata(self, local_index: int):
        """Returns the human-readable strings (Headline, Image Path) for the UI"""
        return self.demo_df.iloc[local_index].to_dict()