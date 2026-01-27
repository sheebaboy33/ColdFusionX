import os
import json
from typing import Tuple, Dict

class ConfigLoader:
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        self.config_dir = os.path.join(artifact_dir, "config")
        self.vocab_dir = os.path.join(artifact_dir, "vocab")
        

    def load_model_configs(self) -> Tuple[Dict, Dict]:
        """Loads fusion_config.json and meta_config.json"""
        
        fusion_path = os.path.join(self.config_dir, "fusion_config.json")
        meta_path = os.path.join(self.config_dir, "meta_config.json")
        
        if not os.path.exists(fusion_path):
            raise FileNotFoundError(f"Missing fusion config at {fusion_path}")
            
        with open(fusion_path, 'r') as f:
            fusion_config = json.load(f)
            
        with open(meta_path, 'r') as f:
            meta_config = json.load(f)
            
        return fusion_config, meta_config


    def load_vocab_mappings(self) -> Tuple[Dict, Dict, Dict]:
        """Loads user, item, and scene ID mappings"""
        
        u_path = os.path.join(self.vocab_dir, "user_id_2idx.json")
        i_path = os.path.join(self.vocab_dir, "item_id_2idx.json")
        s_path = os.path.join(self.vocab_dir, "scene_2idx.json")
        
        with open(u_path, 'r') as f: user_map = json.load(f)
        with open(i_path, 'r') as f: item_map = json.load(f)
        with open(s_path, 'r') as f: scene_map = json.load(f)
            
        return user_map, item_map, scene_map