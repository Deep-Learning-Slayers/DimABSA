"""
Data Loading and Preprocessing Module for DimABSA
Handles JSONL dataset loading, preprocessing, and PyTorch Dataset creation.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_jsonl(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load a JSONL file and return list of dictionaries.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parse error at line {line_num}: {e}")
                continue
    return data


def parse_va_string(va_str: str) -> Tuple[float, float]:
    """
    Parse VA string format "V#A" into float tuple.

    Args:
        va_str: String in format "7.50#6.25"

    Returns:
        Tuple of (valence, arousal) as floats
    """
    parts = va_str.split('#')
    if len(parts) != 2:
        raise ValueError(f"Invalid VA format: {va_str}")
    return float(parts[0]), float(parts[1])


def jsonl_to_dataframe(data: List[Dict], task: str = "task1") -> pd.DataFrame:
    """
    Convert JSONL data to pandas DataFrame for Subtask 1.

    Handles multiple data formats:
    - Quadruplet (training data with full annotations)
    - Triplet (alternative format)
    - Aspect_VA (evaluation format)
    - Aspect only (prediction format without labels)

    Args:
        data: List of JSON objects from JSONL file
        task: Task identifier ("task1", "task2", "task3")

    Returns:
        DataFrame with columns: ID, Text, Aspect, Valence, Arousal
    """
    if not data:
        return pd.DataFrame(columns=['ID', 'Text', 'Aspect', 'Valence', 'Arousal'])

    rows = []

    for entry in data:
        text_id = entry.get('ID', '')
        text = entry.get('Text', '')

        # Handle different data formats
        if 'Quadruplet' in entry:
            # Training data with full quadruplet annotations
            for quad in entry['Quadruplet']:
                aspect = quad.get('Aspect', 'NULL')
                va_str = quad.get('VA', '5.00#5.00')
                valence, arousal = parse_va_string(va_str)
                rows.append({
                    'ID': text_id,
                    'Text': text,
                    'Aspect': aspect,
                    'Valence': valence,
                    'Arousal': arousal
                })

        elif 'Triplet' in entry:
            # Triplet format
            for triplet in entry['Triplet']:
                aspect = triplet.get('Aspect', 'NULL')
                va_str = triplet.get('VA', '5.00#5.00')
                valence, arousal = parse_va_string(va_str)
                rows.append({
                    'ID': text_id,
                    'Text': text,
                    'Aspect': aspect,
                    'Valence': valence,
                    'Arousal': arousal
                })

        elif 'Aspect_VA' in entry:
            # Aspect_VA format (evaluation data)
            for item in entry['Aspect_VA']:
                aspect = item.get('Aspect', '')
                va_str = item.get('VA', '5.00#5.00')
                valence, arousal = parse_va_string(va_str)
                rows.append({
                    'ID': text_id,
                    'Text': text,
                    'Aspect': aspect,
                    'Valence': valence,
                    'Arousal': arousal
                })

        elif 'Aspect' in entry:
            # Prediction format (no VA labels)
            aspects = entry['Aspect']
            if isinstance(aspects, list):
                for aspect in aspects:
                    rows.append({
                        'ID': text_id,
                        'Text': text,
                        'Aspect': aspect,
                        'Valence': 5.0,  # Neutral default
                        'Arousal': 5.0
                    })
            else:
                rows.append({
                    'ID': text_id,
                    'Text': text,
                    'Aspect': aspects,
                    'Valence': 5.0,
                    'Arousal': 5.0
                })

    df = pd.DataFrame(rows)

    # Remove duplicates (same ID + Aspect)
    df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')

    return df


def load_dimabsa_dataset(
    data_dir: Union[str, Path],
    lang: str = "eng",
    domain: str = "restaurant",
    split_dev: bool = True,
    dev_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Load DimABSA dataset for a specific language and domain.

    Args:
        data_dir: Path to DimABSA2026/task-dataset/track_a/subtask_1/
        lang: Language code ("eng", "jpn", "rus", etc.)
        domain: Domain ("restaurant", "laptop", "hotel", "finance")
        split_dev: Whether to split training data for validation
        dev_size: Fraction of training data for validation if split_dev=True
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'dev', and optionally 'test' DataFrames
    """
    data_dir = Path(data_dir)
    lang_dir = data_dir / lang

    result = {}

    # Load training data
    train_file = lang_dir / f"{lang}_{domain}_train_alltasks.jsonl"
    if train_file.exists():
        train_data = load_jsonl(train_file)
        train_df = jsonl_to_dataframe(train_data)

        if split_dev:
            train_df, dev_df = train_test_split(
                train_df,
                test_size=dev_size,
                random_state=random_state
            )
            result['train'] = train_df.reset_index(drop=True)
            result['dev'] = dev_df.reset_index(drop=True)
        else:
            result['train'] = train_df

    # Load dev/test data (task1 format - for prediction)
    dev_file = lang_dir / f"{lang}_{domain}_dev_task1.jsonl"
    if dev_file.exists():
        dev_data = load_jsonl(dev_file)
        result['test'] = jsonl_to_dataframe(dev_data)

    return result


class DimABSADataset(Dataset):
    """
    PyTorch Dataset for DimABSA task.

    Tokenizes input as "[CLS] sentence [SEP] aspect [SEP]" format
    for aspect-aware sentiment prediction.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        include_labels: bool = True
    ):
        """
        Initialize dataset.

        Args:
            dataframe: DataFrame with columns [ID, Text, Aspect, Valence, Arousal]
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            include_labels: Whether to include VA labels (False for prediction)
        """
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

        # Pre-tokenize all examples
        self.encodings = self._tokenize_all()

    def _tokenize_all(self):
        """Tokenize all examples in the dataset."""
        texts = self.df['Text'].tolist()
        aspects = self.df['Aspect'].tolist()

        # Format: text [SEP] aspect
        # The tokenizer will add [CLS] and final [SEP] automatically
        encodings = self.tokenizer(
            texts,
            aspects,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }

        # Add token_type_ids if available (for BERT-style models)
        if 'token_type_ids' in self.encodings:
            item['token_type_ids'] = self.encodings['token_type_ids'][idx]

        if self.include_labels:
            item['labels'] = torch.tensor(
                [self.df.loc[idx, 'Valence'], self.df.loc[idx, 'Arousal']],
                dtype=torch.float
            )

        return item

    def get_aspect_token_positions(self, idx: int) -> Tuple[int, int]:
        """
        Get start and end positions of aspect tokens in the input.
        Useful for aspect-aware attention mechanisms.

        Args:
            idx: Index of the example

        Returns:
            Tuple of (start_pos, end_pos) for aspect tokens
        """
        input_ids = self.encodings['input_ids'][idx]

        # Find SEP tokens
        sep_token_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) >= 2:
            # Aspect is between first and second SEP
            start = sep_positions[0].item() + 1
            end = sep_positions[1].item()
            return start, end
        else:
            # Fallback: return last few tokens before padding
            return len(input_ids) - 2, len(input_ids) - 1


class AspectAwareDataset(Dataset):
    """
    Enhanced dataset that provides aspect position information
    for attention-weighted aspect representation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        include_labels: bool = True
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = row['Text']
        aspect = row['Aspect']

        # Tokenize with aspect as second segment
        encoding = self.tokenizer(
            text,
            aspect,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=False
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
            # Aspect tokens have token_type_id = 1
            aspect_mask = item['token_type_ids'] == 1
        else:
            # For models without token_type_ids, find aspect by locating SEP tokens
            sep_id = self.tokenizer.sep_token_id
            input_ids = item['input_ids']
            sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]

            aspect_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if len(sep_positions) >= 2:
                start = sep_positions[0].item() + 1
                end = sep_positions[1].item()
                aspect_mask[start:end] = True

        item['aspect_mask'] = aspect_mask.float()

        if self.include_labels:
            item['labels'] = torch.tensor(
                [row['Valence'], row['Arousal']],
                dtype=torch.float
            )

        # Store raw text and aspect for lexicon lookup
        item['text'] = text
        item['aspect'] = aspect

        return item


def create_dataloaders(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
    use_aspect_aware: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        train_df: Training DataFrame
        dev_df: Validation DataFrame
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        use_aspect_aware: Use AspectAwareDataset for attention mechanisms

    Returns:
        Tuple of (train_loader, dev_loader)
    """
    DatasetClass = AspectAwareDataset if use_aspect_aware else DimABSADataset

    train_dataset = DatasetClass(train_df, tokenizer, max_length)
    dev_dataset = DatasetClass(dev_df, tokenizer, max_length)

    # Custom collate function to handle string fields
    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            elif isinstance(batch[0][key], str):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = [item[key] for item in batch]
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn if use_aspect_aware else None
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn if use_aspect_aware else None
    )

    return train_loader, dev_loader


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistics for a DimABSA dataset.

    Args:
        df: DataFrame with DimABSA data

    Returns:
        Dictionary with statistics
    """
    return {
        'num_samples': len(df),
        'num_unique_texts': df['ID'].nunique(),
        'avg_aspects_per_text': len(df) / df['ID'].nunique() if df['ID'].nunique() > 0 else 0,
        'valence_mean': df['Valence'].mean(),
        'valence_std': df['Valence'].std(),
        'valence_min': df['Valence'].min(),
        'valence_max': df['Valence'].max(),
        'arousal_mean': df['Arousal'].mean(),
        'arousal_std': df['Arousal'].std(),
        'arousal_min': df['Arousal'].min(),
        'arousal_max': df['Arousal'].max(),
        'avg_text_length': df['Text'].str.len().mean(),
        'avg_aspect_length': df['Aspect'].str.len().mean(),
    }


if __name__ == "__main__":
    # Test data loading
    import sys

    data_dir = Path("DimABSA2026/task-dataset/track_a/subtask_1")

    if data_dir.exists():
        print("Loading English Restaurant dataset...")
        datasets = load_dimabsa_dataset(data_dir, lang="eng", domain="restaurant")

        for split, df in datasets.items():
            print(f"\n{split.upper()} set:")
            stats = get_dataset_statistics(df)
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            print(f"  Sample:\n{df.head(2)}")
    else:
        print(f"Data directory not found: {data_dir}")
