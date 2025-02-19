from textattack.augmentation import EasyDataAugmenter
import torch
import numpy as np
def augment_dataset(df, text_column, label_column):
    """Augment dataset using EDA"""
    eda = EasyDataAugmenter()
    df['aug_text'] = df[text_column].apply(lambda x: eda.augment(x)[0])
    return pd.concat([
        df,
        pd.DataFrame({
            text_column: df['aug_text'],
            label_column: df[label_column]
        })
    ], ignore_index=True)

def tta_prediction(text, model, tokenizer, device, num_augments=5):
    """Test-time augmentation prediction"""
    eda = EasyDataAugmenter()
    variations = [eda.augment(text)[0] for _ in range(num_augments)]
    variations.append(text)

    inputs = tokenizer(
        variations,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    
    return np.bincount(preds).argmax()
