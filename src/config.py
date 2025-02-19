from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    model_name: str = 'roberta-base'
    num_labels: int = 3
    max_length: int = 512
    hidden_dropout_prob: float = 0.3
    attention_probs_dropout_prob: float = 0.3
    
@dataclass
class TrainingConfig:
    output_dir: str = './results'
    num_train_epochs: int = 4
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    evaluation_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    load_best_model_at_end: bool = True
    metric_for_best_model: str = 'eval_accuracy'
    logging_dir: str = './logs'

@dataclass
class DataConfig:
    train_path: str = 'data/raw/train.csv'
    test_path: str = 'data/raw/test.csv'
    cache_dir: str = 'data/cache'
    processed_dir: str = 'data/processed'
    validation_split: float = 0.2
    random_seed: int = 42
