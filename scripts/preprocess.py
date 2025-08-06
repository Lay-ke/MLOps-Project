import pandas as pd
import os


def preprocess_data():
    """
    Preprocess the raw Iris dataset:
    1. Remove Id column
    2. Rename columns to lowercase with underscores
    3. Clean species names (remove 'Iris-' prefix)
    4. Save cleaned data to data/iris.csv
    """
    # Define input and output paths
    raw_data_path = os.environ.get('RAW_DATA_PATH', '/opt/airflow/dags/repo/raw/raw_Iris.csv')
    output_data_path = os.environ.get('OUTPUT_DATA_PATH', '/opt/airflow/dags/repo/data/iris.csv')

    print(f"Loading raw data from: {raw_data_path}")
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Remove Id column
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("Removed 'Id' column")
    
    # Rename columns to match the cleaned format
    column_mapping = {
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm': 'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm': 'petal_width',
        'Species': 'species'
    }
    
    df = df.rename(columns=column_mapping)
    print(f"Renamed columns: {list(df.columns)}")
    
    # Clean species names - remove 'Iris-' prefix
    df['species'] = df['species'].str.replace('Iris-', '', regex=False)
    
    print(f"Unique species after cleaning: {df['species'].unique()}")
    print(f"Cleaned data shape: {df.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_data_path, index=False)
    print(f"Cleaned data saved to: {output_data_path}")
    
    # Display first few rows for verification
    print("\nFirst 5 rows of cleaned data:")
    print(df.head())
    
    return output_data_path


if __name__ == "__main__":
    preprocess_data()
