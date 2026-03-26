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
        # Save this path so we can construct absolute file paths later
        self.artifact_dir = artifact_dir

        self.data_dir = os.path.join(artifact_dir, "data")
        
        # 1. Load Metadata (Pandas)
        df_path = os.path.join(self.data_dir, "demo_rows.parquet")
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Demo Data not found at {df_path}. Did artifacts unzip correctly?")
        
        self.demo_df = pd.read_parquet(df_path).reset_index(drop=True)
        print(f"[FeatureStore] Loaded {len(self.demo_df)} rows.")
        
        # 2. Load Tensors (CPU)
        # We load these entirely into memory because they are small (only ~395 rows)
        self.text_emb = np.load(os.path.join(self.data_dir, "demo_text_embeddings.npy"))
        self.time_feats = torch.load(os.path.join(self.data_dir, "demo_time_features.pt"), map_location="cpu")
        self.image_emb = torch.load(os.path.join(self.data_dir, "demo_image_embeddings.pt"), map_location="cpu")
        
        # 3. Load Indices to map Global Row ID -> Local Demo Index
        # This helps if we look up by original dataset index
        self.global_indices = np.load(os.path.join(self.data_dir, "demo_indices.npy"))
        self.global_to_local = {gid: loc for loc, gid in enumerate(self.global_indices)}

    def get_image_path(self, item_id: str) -> str:
        """
        Returns the absolute path to the raw image file.
        Logic: artifacts/release_demo/images/{subfolder}/{item_id}.png
        """
        sid = str(item_id).strip()
        
        # Logic matches your artifact script: first 3 chars define subfolder
        if len(sid) < 3:
            subfolder = sid.zfill(3)
        else:
            subfolder = sid[:3]
            
        # Construct absolute path
        return os.path.join(self.artifact_dir, "images", subfolder, f"{sid}.png")

    def list_cases(self):
        """
        Returns summary DF. Auto-creates 'scene_id' string from numeric 'scene_idx'.
        """
        df = self.demo_df.copy()
        
        # 1. Ensure User/Item IDs exist as strings
        if "user_id" not in df.columns and "user_id_idx" in df.columns:
            df["user_id"] = df["user_id_idx"].astype(str)
        if "item_id" not in df.columns and "item_id_idx" in df.columns:
            df["item_id"] = df["item_id_idx"].astype(str)

        # 2. Ensure Scene ID (String) and Scene IDX (Int) exist
        # we turn them into "Scenario 0", "Scenario 1"...
        if "scene_idx" in df.columns:
            # We have the index, ensure we have a string Label
            if "scene_id" not in df.columns:
                df["scene_id"] = "Scenario " + df["scene_idx"].astype(str)
        elif "scene_id_idx" in df.columns:
            # Handle alternative naming
            df["scene_idx"] = df["scene_id_idx"]
            df["scene_id"] = "Scenario " + df["scene_id_idx"].astype(str)
        else:
            # Fallback
            df["scene_idx"] = 0
            df["scene_id"] = "Unknown"

        # 3. Ensure Timestamp
        if "timestamp" not in df.columns:
             df["timestamp"] = "2025-01-01"

        # Return ALL columns so app can find indices
        return df

    def get_case_by_index(self, local_index: int, device: str = "cpu") -> Dict[str, Any]:
        """
        Retrieves the exact tensors for a specific row in the demo dataset.
        Ready for Model Injection.
        """
        if local_index >= len(self.demo_df):
            raise ValueError(f"Index {local_index} out of bounds.")

        # Use the raw DF for tensor extraction to match original indices
        row = self.demo_df.iloc[local_index]

        # Safe extraction of indices
        u_idx = row.get("user_id_idx", 0)
        i_idx = row.get("item_id_idx", 0)

        # Handle Scene Index Robustly
        if "scene_idx" in row: s_idx = row["scene_idx"]
        elif "scene_id_idx" in row: s_idx = row["scene_id_idx"]
        else: s_idx = 0
        
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
        """Returns the CLEANED dict (with scene_id string) for UI"""
        df_clean = self.list_cases()
        return df_clean.iloc[local_index].to_dict()