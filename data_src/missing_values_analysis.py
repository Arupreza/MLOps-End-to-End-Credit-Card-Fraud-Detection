from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, data: pd.DataFrame) -> None:
        """
        Analyze missing values in the DataFrame.
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.
        """
        self.identify_missing_values(data)
        self.visualize_missing_values(data)

    @abstractmethod
    def identify_missing_values(self, data: pd.DataFrame) -> None:
        """
        Identify missing values in the DataFrame.
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, data: pd.DataFrame) -> None:
        """
        Visualize missing values in the DataFrame.
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.
        """
        pass

class MissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, data: pd.DataFrame) -> None:
        """
        Identify missing values in the DataFrame.
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.
        """
        print("Missing Values Count by Columns:")
        missing_info = data.isnull().sum()
        print(missing_info[missing_info > 0])

    def visualize_missing_values(self, data: pd.DataFrame) -> None:
        """
        Visualize missing values in the DataFrame.
        Parameters:
        data (pd.DataFrame): The DataFrame to analyze.
        """
        print
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()

if __name__ == "__main__":
    # Example usage of the MissingValuesAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Missing Values Analysis
    # missing_values_analyzer = MissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass