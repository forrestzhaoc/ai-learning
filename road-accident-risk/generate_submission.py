"""
Generate submission file for Kaggle
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def generate_submission(model_path='models/ensemble_model.pkl', 
                       output_dir='predictions',
                       output_filename=None):
    """
    Generate submission file using trained model
    
    Args:
        model_path: Path to saved model
        output_dir: Directory to save submission file
        output_filename: Custom filename for submission (optional)
    """
    print("\n" + "="*60)
    print("Generating Kaggle Submission")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python -m src.train")
        sys.exit(1)
    
    # Import here to avoid circular dependency
    from src.data_processing import DataProcessor
    from src.train import EnsembleModel
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = EnsembleModel.load(model_path)
    
    # Load and process data
    print("\nProcessing data...")
    processor = DataProcessor(data_dir='data')
    train_df, test_df = processor.load_data()
    
    # Prepare features
    X_train, X_test, y_train, test_ids = processor.prepare_features(train_df, test_df)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_ids,
        'accident_risk': predictions
    })
    
    # Ensure predictions are non-negative (if applicable)
    submission_df['accident_risk'] = submission_df['accident_risk'].clip(lower=0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    if output_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'submission_{timestamp}.csv'
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Rows: {len(submission_df):,}")
    print(f"  Columns: {list(submission_df.columns)}")
    print(f"\nPrediction statistics:")
    print(submission_df['accident_risk'].describe())
    
    print(f"\nFirst few predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("Submission generation complete!")
    print("="*60)
    print(f"\nYou can now submit {output_path} to Kaggle:")
    print(f"  kaggle competitions submit -c playground-series-s5e10 -f {output_path} -m \"Ensemble model submission\"")
    
    return submission_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Kaggle submission file')
    parser.add_argument('--model', type=str, default='models/ensemble_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Output directory for submission file')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom filename for submission')
    
    args = parser.parse_args()
    
    generate_submission(
        model_path=args.model,
        output_dir=args.output_dir,
        output_filename=args.output_name
    )
