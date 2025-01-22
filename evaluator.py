#!/usr/bin/env python3
"""
Compare a submission.csv file against the real targets in professors-test-metadata.csv

Usage:
    python evaluate_submission.py --submission path_to_submission.csv --ground_truth professors-test-metadata.csv

Example:
    python evaluate_submission.py --submission submission.csv --ground_truth professors-test-metadata.csv
"""

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_submission(submission_file: str, ground_truth_file: str):
    """
    Reads a submission file and the ground truth, then computes basic metrics.
    
    - Merges both on 'isic_id'.
    - Computes AUC (if possible) and prints simple classification accuracy at a 0.5 threshold.

    Args:
        submission_file (str): Path to the CSV with columns [isic_id, target].
        ground_truth_file (str): Path to the CSV with columns [isic_id, target].
    """
    # Read data
    sub_df = pd.read_csv(submission_file)
    gt_df = pd.read_csv(ground_truth_file)

    # Merge on isic_id to align predictions and real targets
    merged_df = pd.merge(sub_df, gt_df, on='isic_id', how='inner', suffixes=('_pred', '_true'))

    # Ensure the columns exist
    if 'target_pred' not in merged_df.columns or 'target_true' not in merged_df.columns:
        raise ValueError("Merged data does not have the expected 'target_pred' or 'target_true' columns.")

    # Extract predictions and ground truth
    y_pred = merged_df['target_pred'].values
    y_true = merged_df['target_true'].values

    # Compute AUC
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = None
        print("[WARNING] AUC could not be computed. Make sure y_true contains both 0 and 1 classes.")

    # Compute accuracy at a fixed threshold of 0.5
    y_pred_class = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_class)

    # Print results
    print("Number of samples evaluated:", len(merged_df))
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print(f"Accuracy (threshold=0.5): {acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission CSV against ground truth.")
    parser.add_argument("--submission", required=True, help="Path to submission CSV file.")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth CSV file (professors-test-metadata.csv).")

    args = parser.parse_args()

    evaluate_submission(args.submission, args.ground_truth)

if __name__ == "__main__":
    main()
