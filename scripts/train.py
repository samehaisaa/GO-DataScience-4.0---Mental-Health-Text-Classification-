import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    set_seed
)
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from src.config import ModelConfig, TrainingConfig, DataConfig
from src.data.dataset import MentalHealthDataset
from src.models.model import WeightedTrainer
from src.utils.metrics import compute_metrics
from src.utils.preprocessing import preprocess_data
from src.data.dataloader import DataLoader
from src.utils.augmentation import augment_dataset

def train_model():
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Set random seed for reproducibility
    set_seed(data_config.random_seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader
    data_loader = DataLoader(data_config)
    train_df, test_df = data_loader.load_data()
    
    # Preprocess data
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Prepare labels
    data_loader.prepare_labels(train_df)
    data_loader.save_label_mappings()
    
    # Data augmentation
    train_df = augment_dataset(train_df, 'text', 'target')
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['text'],
        train_df['target'].map(data_loader.label2id),
        test_size=data_config.validation_split,
        stratify=train_df['target'],
        random_state=data_config.random_seed
    )
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(
        model_config.model_name,
        max_length=model_config.max_length,
        truncation=True,
        padding='max_length'
    )
    
    # Create datasets
    train_dataset = MentalHealthDataset(
        X_train,
        y_train,
        cache_dir=os.path.join(data_config.cache_dir, 'train'),
        tokenizer=tokenizer
    )
    val_dataset = MentalHealthDataset(
        X_val,
        y_val,
        cache_dir=os.path.join(data_config.cache_dir, 'val'),
        tokenizer=tokenizer
    )
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        model_config.model_name,
        num_labels=model_config.num_labels,
        id2label=data_loader.id2label,
        label2id=data_loader.label2id,
        hidden_dropout_prob=model_config.hidden_dropout_prob,
        attention_probs_dropout_prob=model_config.attention_probs_dropout_prob
    ).to(device)
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    trainer.train()
    
    return trainer, model, tokenizer

if __name__ == "__main__":
    trainer, model, tokenizer = train_model()
    trainer.save_model()