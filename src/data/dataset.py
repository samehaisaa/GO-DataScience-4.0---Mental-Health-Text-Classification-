import os
import torch
from torch.utils.data import Dataset

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels=None, cache_dir='./cache', tokenizer=None):
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if isinstance(labels, pd.Series) else labels
        self.cache_dir = cache_dir
        self.tokenizer = tokenizer

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, 'encodings.pt')

        if os.path.exists(cache_path):
            print("Loading cached encodings...")
            self.encodings = torch.load(cache_path)
        else:
            print("Tokenizing texts...")
            self.encodings = self.tokenizer.batch_encode_plus(
                self.texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            torch.save(self.encodings, cache_path)
            print("Encodings cached.")

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
