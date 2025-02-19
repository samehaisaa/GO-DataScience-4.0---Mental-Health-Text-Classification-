import pandas as pd
from typing import Tuple, Optional
import os

def preprocess_data(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    text_cols: Tuple[str, str] = ('title', 'content'),
    target_col: str = 'target'
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Preprocess the input dataframes for sentiment analysis.
    
    Args:
        train_df: Training dataframe
        test_df: Optional test dataframe
        text_cols: Tuple of column names containing text data
        target_col: Name of the target column
        
    Returns:
        Preprocessed train and test dataframes
    """
    def process_df(df: pd.DataFrame) -> pd.DataFrame:
        df['text'] = df[text_cols[0]].fillna('') + ' ' + df[text_cols[1]].fillna('')
        df['text'] = df['text'].fillna('').astype(str)
        
        return df
    
    train_df = process_df(train_df)
    if test_df is not None:
        test_df = process_df(test_df)
        
    return train_df, test_df
