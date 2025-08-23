import pandas as pd
import numpy as np
import sys
import yaml
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Initialize scaler
scaler = MinMaxScaler()

try:
    params = yaml.safe_load(open("params.yaml"))["preprocess"]
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading params: {e}")
    sys.exit(1)

def preprocessing(input_path, output_path):
    """
    Preprocess the data by selecting features and applying MinMaxScaler.
    
    Parameters:
    input_path (str): Path to input CSV file
    output_path (str): Path to save processed data
    """
    try:
        # Load the data
        print(f"Loading data from: {input_path}")
        data = pd.read_csv(input_path)
        print(f"Original data shape: {data.shape}")
        
        # Select highly correlated columns + target
        selected_columns = ['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 
                           'V16', 'V17', 'V18', 'V21', 'V22', 'Class']
        
        # Check if all columns exist
        missing_cols = [col for col in selected_columns if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns in data: {missing_cols}")
            selected_columns = [col for col in selected_columns if col in data.columns]
        
        data = data[selected_columns]
        print(f"Selected {len(selected_columns)} columns")
        
        # Separate features and target
        feature_columns = [col for col in selected_columns if col != 'Class']
        
        if 'Class' in data.columns:
            X = data[feature_columns]
            y = data['Class']
            
            print(f"Features shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            # Display original feature ranges
            print("\nOriginal feature ranges:")
            print("-" * 40)
            for col in feature_columns:
                print(f"{col}: [{X[col].min():.3f}, {X[col].max():.3f}]")
            
            # Apply MinMaxScaler to features only
            print("\nApplying MinMaxScaler...")
            X_scaled = scaler.fit_transform(X)
            
            # Convert back to DataFrame
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns, index=data.index)
            
            # Combine scaled features with target
            processed_data = pd.concat([X_scaled_df, y], axis=1)
            
            # Verify scaling
            print("\nScaled feature ranges (should be [0, 1]):")
            print("-" * 40)
            for col in feature_columns:
                print(f"{col}: [{processed_data[col].min():.3f}, {processed_data[col].max():.3f}]")
            
            # Display class distribution
            print(f"\nClass distribution:")
            print(processed_data['Class'].value_counts())
            
        else:
            # If no Class column, scale all selected features
            print("No 'Class' column found. Scaling all selected features.")
            processed_data = pd.DataFrame(
                scaler.fit_transform(data), 
                columns=data.columns, 
                index=data.index
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save processed data
        processed_data.to_csv(output_path, index=False)
        print(f"\nPreprocessing completed. Data saved to: {output_path}")
        print(f"Processed data shape: {processed_data.shape}")
        
        # Save the scaler for future use
        scaler_path = output_path.replace('.csv', '_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        return processed_data
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run preprocessing
    result = preprocessing(params["input_path"], params["output_path"])
    
    # Create train-validation-test split if specified in params
    if "create_split" in params and params["create_split"]:
        print("\nCreating train-validation-test split...")
        
        # Separate features and target
        X = result.iloc[:, :-1]  # All columns except last
        y = result.iloc[:, -1]   # Last column (Class)
        
        # Get split ratios from params
        train_size = params.get("train_size", 0.7)
        val_size = params.get("val_size", 0.15)
        test_size = params.get("test_size", 0.15)
        random_state = params.get("random_state", 42)
        
        # Validate split ratios
        if abs(train_size + val_size + test_size - 1.0) > 0.001:
            print(f"Warning: Split ratios don't sum to 1.0. Got {train_size + val_size + test_size}")
            print("Adjusting ratios proportionally...")
            total = train_size + val_size + test_size
            train_size /= total
            val_size /= total
            test_size /= total
        
        print(f"Split ratios - Train: {train_size:.2f}, Val: {val_size:.2f}, Test: {test_size:.2f}")
        
        # First split: separate train+val from test
        temp_size = train_size + val_size  # Combined train+val size
        stratify_param = y if len(y.unique()) > 1 else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        # Second split: separate train from validation
        # Calculate validation size relative to temp set
        val_size_relative = val_size / temp_size
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_relative, 
            random_state=random_state + 1,  # Different seed for second split
            stratify=y_temp if len(y_temp.unique()) > 1 else None
        )
        
        # Combine features and targets for each split
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        # Generate file paths
        base_path = params["output_path"].replace('.csv', '')
        train_path = f"{base_path}_train.csv"
        val_path = f"{base_path}_val.csv"
        test_path = f"{base_path}_test.csv"
        
        # Save all splits
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Display split information
        print(f"\nData Split Summary:")
        print("=" * 50)
        print(f"Train data saved to: {train_path}")
        print(f"  Shape: {train_data.shape} ({len(train_data)/len(result)*100:.1f}%)")
        print(f"  Class distribution: {dict(y_train.value_counts())}")
        
        print(f"\nValidation data saved to: {val_path}")
        print(f"  Shape: {val_data.shape} ({len(val_data)/len(result)*100:.1f}%)")
        print(f"  Class distribution: {dict(y_val.value_counts())}")
        
        print(f"\nTest data saved to: {test_path}")
        print(f"  Shape: {test_data.shape} ({len(test_data)/len(result)*100:.1f}%)")
        print(f"  Class distribution: {dict(y_test.value_counts())}")
        
        # Verify total samples
        total_samples = len(train_data) + len(val_data) + len(test_data)
        print(f"\nTotal samples: {total_samples} (Original: {len(result)})")
        
        print("\n" + "="*50)
        print("PREPROCESSING AND SPLITTING COMPLETED!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETED!")
        print("="*50)






# import pandas as pd
# import numpy as np
# import sys
# import yaml
# import os
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

# scaler = MinMaxScaler()

# try:
#     params = yaml.safe_load(open("params.yaml"))["preprocess"]
# except (FileNotFoundError, KeyError) as e:
#     print(f"Error loading params: {e}")
#     sys.exit(1)

# def preprocessing(input_path, output_path):
#     data = pd.read_csv(input_path)
#     data = data[['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V21', 'V22','Class']]
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     data.to_csv(output_path, index=False)
#     print("Preprocessing completed. Data saved to %s" % output_path)

# if __name__ == "__main__":
#     preprocessing(params["input_path"], params["output_path"])