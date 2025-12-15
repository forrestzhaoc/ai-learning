"""
Quick Start Script - Run complete pipeline from data loading to submission generation
"""

import os
import sys
from datetime import datetime

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_data_exists():
    """Check if data files exist"""
    data_dir = 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    test_file = os.path.join(data_dir, 'test.csv')
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        return False
    return True

def run_pipeline(skip_eda=False, n_folds=5):
    """
    Run the complete machine learning pipeline
    
    Args:
        skip_eda: Skip exploratory data analysis
        n_folds: Number of folds for cross-validation
    """
    print_header("ROAD ACCIDENT RISK PREDICTION - QUICK START")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check and download data if needed
    print_header("Step 1: Data Preparation")
    
    if not check_data_exists():
        print("Data files not found. Downloading from Kaggle...")
        try:
            import download_data
            download_data.download_kaggle_dataset()
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("\nPlease download data manually:")
            print("  python download_data.py")
            sys.exit(1)
    else:
        print("✓ Data files found")
    
    # Step 2: Exploratory Data Analysis (optional)
    if not skip_eda:
        print_header("Step 2: Exploratory Data Analysis")
        try:
            from src import eda
            eda.generate_eda_report()
        except Exception as e:
            print(f"Warning: EDA failed: {e}")
            print("Continuing with training...")
    else:
        print("\nSkipping EDA (use --eda flag to include)")
    
    # Step 3: Load and process data
    print_header("Step 3: Data Processing")
    
    try:
        from src.data_processing import load_and_process_data
        X_train, X_val, X_test, y_train, y_val, test_ids, processor = load_and_process_data()
        
        print("\n✓ Data processing complete")
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)
    
    # Step 4: Train model
    print_header("Step 4: Model Training")
    
    try:
        from src.train import train_model
        import numpy as np
        
        # Use all available training data
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"Training with {n_folds}-fold cross-validation...")
        model = train_model(
            X_train_full, 
            y_train_full, 
            n_folds=n_folds,
            save_path='models/ensemble_model.pkl'
        )
        
        print("\n✓ Model training complete")
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Generate submission
    print_header("Step 5: Generate Submission")
    
    try:
        from generate_submission import generate_submission
        submission_df = generate_submission()
        
        print("\n✓ Submission generation complete")
    except Exception as e:
        print(f"Error generating submission: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print_header("PIPELINE COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - models/ensemble_model.pkl (trained model)")
    print("  - predictions/submission_*.csv (submission file)")
    
    if not skip_eda:
        print("  - data/target_distribution.png")
        print("  - data/feature_correlations.png")
    
    print("\nNext steps:")
    print("  1. Review the submission file in predictions/")
    print("  2. Submit to Kaggle:")
    print("     kaggle competitions submit -c playground-series-s5e10 \\")
    print("       -f predictions/submission_*.csv -m \"Your message\"")
    print("\n" + "="*70)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Quick start script for Road Accident Risk Prediction'
    )
    parser.add_argument('--eda', action='store_true',
                       help='Run exploratory data analysis')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    
    args = parser.parse_args()
    
    try:
        run_pipeline(skip_eda=not args.eda, n_folds=args.folds)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
