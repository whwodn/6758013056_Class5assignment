import pandas as pd
from sklearn.model_selection import train_test_split

# Function to perform the 90/10 split and save the entire dataset including all columns
def split_and_save_data(file_path, output_train_file, output_test_file):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Split the dataset into features (X) and target (y)
    X = df.drop(columns=['ProdTaken'])  # Features (excluding the target)
    y = df['ProdTaken']  # Target variable

    # Perform the 90/10 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Combine the features and target back into dataframes
    train_data = pd.concat([X_train, y_train], axis=1)  # Combine training features and target
    test_data = pd.concat([X_test, y_test], axis=1)  # Combine testing features and target

    # Save the split datasets to CSV files
    train_data.to_csv(output_train_file, index=False)
    test_data.to_csv(output_test_file, index=False)

    print(f"Data has been split and saved:\n"
          f"Training Data: {output_train_file}\n"
          f"Testing Data: {output_test_file}")

# Example usage
file_path = 'data/class5/c5aclean.csv'  # Input file path
output_train_file = 'data/class5/train_data.csv'
output_test_file = 'data/class5/test_data.csv'

split_and_save_data(file_path, output_train_file, output_test_file)