import argparse
import logging
import os
import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSampler:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        train_ratio: float,
        dev_ratio: float,
        test_ratio: float,
        pos_neg_ratio: float,
        seed: int = 42
    ):
        """
        Initialize the DataSampler.

        Args:
            input_path: Path to the standard_dataset.tsv file
            output_path: Directory to save the split datasets
            train_ratio: Ratio of data to use for training
            dev_ratio: Ratio of data to use for development
            test_ratio: Ratio of data to use for testing
            pos_neg_ratio: Desired ratio of positive to negative samples
            seed: Random seed for reproducibility
        """
        self.input_path = input_path
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.pos_neg_ratio = pos_neg_ratio
        self.seed = seed

        # Validate ratios
        total = train_ratio + dev_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def load_data(self) -> pd.DataFrame:
        """Load the standard dataset."""
        logger.info(f"Loading data from {self.input_path}")
        df = pd.read_csv(self.input_path, sep='\t')
        logger.info(f"Loaded {len(df)} total samples")
        logger.info(f"Positive samples: {len(df[df['label'] == 1])}")
        logger.info(f"Negative samples: {len(df[df['label'] == 0])}")
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, dev, and test sets while maintaining the desired
        positive to negative ratio in each split.
        """
        # Separate positive and negative samples
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]

        # Calculate number of negative samples needed based on pos_neg_ratio
        total_pos = len(pos_df)
        total_neg_needed = int(total_pos / self.pos_neg_ratio)

        # If we don't have enough negative samples, use all we have
        if total_neg_needed > len(neg_df):
            logger.warning(f"Not enough negative samples. Using all {len(neg_df)} available")
            total_neg_needed = len(neg_df)

        # Sample negative samples
        neg_df = neg_df.sample(n=total_neg_needed, random_state=self.seed)

        # Split positive samples
        pos_train, pos_temp = train_test_split(
            pos_df,
            test_size=1-self.train_ratio,
            random_state=self.seed
        )
        pos_dev, pos_test = train_test_split(
            pos_temp,
            test_size=self.test_ratio/(self.dev_ratio + self.test_ratio),
            random_state=self.seed
        )

        # Split negative samples
        neg_train, neg_temp = train_test_split(
            neg_df,
            test_size=1-self.train_ratio,
            random_state=self.seed
        )
        neg_dev, neg_test = train_test_split(
            neg_temp,
            test_size=self.test_ratio/(self.dev_ratio + self.test_ratio),
            random_state=self.seed
        )

        # Combine positive and negative samples for each split
        train_df = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=self.seed)
        dev_df = pd.concat([pos_dev, neg_dev]).sample(frac=1, random_state=self.seed)
        test_df = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=self.seed)

        return train_df, dev_df, test_df

    def save_splits(self, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save the split datasets to TSV files."""
        os.makedirs(self.output_path, exist_ok=True)

        train_path = os.path.join(self.output_path, "train.tsv")
        dev_path = os.path.join(self.output_path, "dev.tsv")
        test_path = os.path.join(self.output_path, "test.tsv")

        train_df.to_csv(train_path, sep='\t', index=False)
        dev_df.to_csv(dev_path, sep='\t', index=False)
        test_df.to_csv(test_path, sep='\t', index=False)

        logger.info(f"Saved train set ({len(train_df)} samples) to {train_path}")
        logger.info(f"Saved dev set ({len(dev_df)} samples) to {dev_path}")
        logger.info(f"Saved test set ({len(test_df)} samples) to {test_path}")

        # Log class distribution
        for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            pos_count = len(df[df['label'] == 1])
            neg_count = len(df[df['label'] == 0])
            logger.info(f"{name} set: {pos_count} positive, {neg_count} negative samples")

    def run(self):
        """Run the complete data sampling process."""
        df = self.load_data()
        train_df, dev_df, test_df = self.split_data(df)
        self.save_splits(train_df, dev_df, test_df)

def main():
    parser = argparse.ArgumentParser(description='Sample and split dataset with controlled ratios')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to the standard_dataset.tsv file')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Directory to save the split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                      help='Ratio of data to use for training (default: 0.6)')
    parser.add_argument('--dev_ratio', type=float, default=0.2,
                      help='Ratio of data to use for development (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                      help='Ratio of data to use for testing (default: 0.2)')
    parser.add_argument('--pos_neg_ratio', type=float, default=1.0,
                      help='Desired ratio of positive to negative samples (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    sampler = DataSampler(
        input_path=args.input_path,
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        pos_neg_ratio=args.pos_neg_ratio,
        seed=args.seed
    )

    sampler.run()

if __name__ == "__main__":
    main()
