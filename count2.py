import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Read the dataset parts and merge them
def merge_csv_files(parts):
    """
    Merge multiple CSV parts into a single DataFrame.
    """
    dataframes = [pd.read_csv(part) for part in parts]
    return pd.concat(dataframes, ignore_index=True)

def print_distribution(y, title="Distribution"):
    """
    Helper function to print class distribution
    """
    counts = pd.Series(y).value_counts()
    percentages = pd.Series(y).value_counts(normalize=True) * 100
    
    print(f"\n{title}:")
    print("Counts:")
    print(counts)
    print("\nPercentages:")
    print(percentages.round(2), "%")
    print("\nTotal samples:", len(y))
    
    # Calculate imbalance ratio
    majority_class = counts.max()
    minority_class = counts.min()
    imbalance_ratio = majority_class / minority_class
    print(f"Imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")

# Specify the paths to your CSV files
encoded_parts = ['output_chunks/clean_df_3_single_weapon_part1.csv', 
                'output_chunks/clean_df_3_single_weapon_part2.csv']

# Merge the CSV files
df_encoded = merge_csv_files(encoded_parts)

# Get original distribution
print_distribution(df_encoded['Weapon Category'], "Original Distribution")

# Prepare data for SMOTE
X = df_encoded.drop(columns=['Weapon Category'])
y = df_encoded['Weapon Category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Print training set distribution before SMOTE
print_distribution(y_train, "Training Set Distribution (Before SMOTE)")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Print distribution after SMOTE
print_distribution(y_train_resampled, "Training Set Distribution (After SMOTE)")

# Print test set distribution (unchanged)
print_distribution(y_test, "Test Set Distribution (Unchanged)")
