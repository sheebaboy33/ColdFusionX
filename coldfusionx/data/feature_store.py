import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

class DemoFeatureStore:
    """
    Loads the highly engineered 'Demo Subset' data and pre-calculated tensors.
    """
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self.data_dir = os.path.join(artifact_dir, "data")
        
        # 1. Load Metadata (Pandas)
        df_path = os.path.join(self.data_dir, "demo_rows.parquet")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Demo Data not found at {df_path}.")
        
        self.demo_df = pd.read_parquet(df_path).reset_index(drop=True)
        print(f"[FeatureStore] Loaded {len(self.demo_df)} highly-engineered rows.")
        
        # 2. Load Tensors (CPU)
        self.text_emb = np.load(os.path.join(self.data_dir, "demo_text_embeddings.npy"), mmap_mode='c')
        self.time_feats = torch.load(os.path.join(self.data_dir, "demo_time_features.pt"), map_location="cpu")
        self.image_emb = torch.load(os.path.join(self.data_dir, "demo_image_embeddings.pt"), map_location="cpu")
        
        # 3. Load Indices
        self.global_indices = np.load(os.path.join(self.data_dir, "demo_indices.npy"))
        self.global_to_local = {gid: loc for loc, gid in enumerate(self.global_indices)}

    def get_image_path(self, item_id: str) -> str:
        """Returns the absolute path to the raw image file."""
        sid = str(item_id).strip()
        subfolder = sid.zfill(3) if len(sid) < 3 else sid[:3]
        return os.path.join(self.artifact_dir, "images", subfolder, f"{sid}.png")

    def list_cases(self) -> pd.DataFrame:
        """Returns the fully engineered dataframe for the UI."""
        return self.demo_df.copy()

    def get_raw_metadata(self, local_index: int) -> Dict[str, Any]:
        """Returns a single row as a dictionary for UI rendering."""
        return self.demo_df.iloc[local_index].to_dict()

    def get_case_by_index(self, local_index: int, device: str = "cpu") -> Dict[str, Any]:
        """
        Retrieves the exact tensors for a specific row.
        Ready for Model Injection.
        """
        if local_index >= len(self.demo_df):
            raise ValueError(f"Index {local_index} out of bounds.")

        row = self.demo_df.iloc[local_index]

        # Use the explicit encoded indices
        u_idx = row.get("user_id_idx", 0)
        i_idx = row.get("item_id_idx", 0)
        
        # Robust scene fallback
        if "scene" in row: s_idx = row["scene"]
        elif "scene_idx" in row: s_idx = row["scene_idx"]
        else: s_idx = 0
        
        # Extract the historical click count we calculated in Colab
        hist_clicks = row.get("historical_click_count", 0)
        
        # Prepare Batch (Add Batch Dimension [1, ...])
        batch = {
            "user_idx": torch.tensor([u_idx], dtype=torch.long, device=device),
            "item_idx": torch.tensor([i_idx], dtype=torch.long, device=device),
            "scene_idx": torch.tensor([s_idx], dtype=torch.long, device=device),
            
            # Features
            "text_vec": torch.tensor(self.text_emb[local_index], dtype=torch.float32, device=device).unsqueeze(0),
            "image_vec": self.image_emb[local_index].unsqueeze(0).to(device),
            "time_vec": self.time_feats[local_index].unsqueeze(0).to(device),
            
            # Metadata for logic
            "item_count": torch.tensor([hist_clicks], dtype=torch.long, device=device)
        }
        
        return batch