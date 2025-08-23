from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame) -> None:
        """
        Inspect the data and print basic statistics.
        Parameters:
        data (pd.DataFrame): The DataFrame to inspect.
        """
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame) -> None:
        """
        Inspect the data and print basic statistics.
        Parameters:
        data (pd.DataFrame): The DataFrame to inspect.
        """
        print("Data Types and Null Values:")
        print(data.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame) -> None:
        """
        Inspect the data and print basic statistics.
        Parameters:
        data (pd.DataFrame): The DataFrame to inspect.
        """
        # Check for numerical columns
        numerical_cols = data.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 0:
            print("Summary Numerical Statistics:")
            print(data[numerical_cols].describe())
        else:
            print("No numerical columns found in the dataset.")
        
        print("\n" + "="*50 + "\n")
        
        # Check for categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print("Summary Categorical Statistics:")
            print(data[categorical_cols].describe())
        else:
            print("No categorical columns found in the dataset.")

class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataInspectionStrategy):
        self._strategy = strategy
    
    def execute_inspection(self, data: pd.DataFrame) -> None:
        """
        Execute the inspection using the current strategy.
        Parameters:
        data (pd.DataFrame): The DataFrame to inspect.
        """
        if data is None:
            print("Error: DataFrame is None")
            return
        
        if data.empty:
            print("Warning: DataFrame is empty")
            return
            
        self._strategy.inspect(data)

# Additional utility class for comprehensive data inspection
class ComprehensiveInspectionStrategy(DataInspectionStrategy):
    def inspect(self, data: pd.DataFrame) -> None:
        """
        Comprehensive inspection including data types, statistics, and column info.
        """
        print("="*60)
        print("COMPREHENSIVE DATA INSPECTION")
        print("="*60)
        
        # Basic info
        print(f"\nDataset Shape: {data.shape}")
        print(f"Total Missing Values: {data.isnull().sum().sum()}")
        
        # Data types summary
        print(f"\nData Types Summary:")
        print(data.dtypes.value_counts())
        
        # Column categories
        numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(include=['datetime']).columns.tolist()
        
        print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
        print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
        print(f"Datetime columns ({len(datetime_cols)}): {datetime_cols}")
        
        # Numerical statistics
        if len(numerical_cols) > 0:
            print("\n" + "="*40)
            print("NUMERICAL STATISTICS")
            print("="*40)
            print(data[numerical_cols].describe())
            
            # Missing values in numerical columns
            num_missing = data[numerical_cols].isnull().sum()
            if num_missing.sum() > 0:
                print(f"\nMissing values in numerical columns:")
                print(num_missing[num_missing > 0])
        else:
            print("\nNo numerical columns found - skipping numerical statistics.")
        
        # Categorical statistics
        if len(categorical_cols) > 0:
            print("\n" + "="*40)
            print("CATEGORICAL STATISTICS")
            print("="*40)
            print(data[categorical_cols].describe())
            
            # Missing values in categorical columns
            cat_missing = data[categorical_cols].isnull().sum()
            if cat_missing.sum() > 0:
                print(f"\nMissing values in categorical columns:")
                print(cat_missing[cat_missing > 0])
                
            # Unique values count for each categorical column
            print(f"\nUnique values per categorical column:")
            for col in categorical_cols:
                unique_count = data[col].nunique()
                print(f"{col}: {unique_count} unique values")
        else:
            print("\nNo categorical columns found - skipping categorical statistics.")

if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.
    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')
    
    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)
    
    # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    
    # Use comprehensive inspection
    # inspector.set_strategy(ComprehensiveInspectionStrategy())
    # inspector.execute_inspection(df)
    pass