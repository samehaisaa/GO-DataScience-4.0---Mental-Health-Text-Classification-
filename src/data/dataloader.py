import pandas as pd
from typing import Tuple, Dict, List
import os
from ..config import DataConfig

class DataLoader:
    """Data loading and preparation class."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        return train_df, test_df
    
    def prepare_labels(self, df: pd.DataFrame, target_col: str = 'target') -> None:
        """Prepare label mappings."""
        labels = df[target_col].unique().tolist()
        self.label2id = {label: idx for idx, label in enumerate(labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def save_label_mappings(self) -> None:
        """Save label mappings to processed directory."""
        os.makedirs(self.config.processed_dir, exist_ok=True)
        pd.DataFrame([self.label2id]).to_json(
            os.path.join(self.config.processed_dir, 'label_mappings.json')
        )
