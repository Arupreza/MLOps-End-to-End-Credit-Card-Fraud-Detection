from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

class MultiVeriateAnalysisTemplate(ABC):
    def analyze(self, data: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.
        Parameters:
        data (pd.DataFrame): The dataframe containing the data to be analyzed.
        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        # Check if data has numerical columns for correlation analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            print("Warning: Need at least 2 numerical columns for multivariate analysis.")
            return
            
        self.generate_correlation_heatmap(data)
        self.generate_pairplot(data)
    
    @abstractmethod
    def generate_correlation_heatmap(self, data: pd.DataFrame):
        pass
    
    @abstractmethod
    def generate_pairplot(self, data: pd.DataFrame):
        pass

class MultiVeriateAnalysis(MultiVeriateAnalysisTemplate):
    def generate_correlation_heatmap(self, data: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between features.
        Parameters:
        data (pd.DataFrame): The dataframe containing the data to be analyzed.
        Returns:
        None: This method generates and displays a correlation heatmap.
        """
        # Select only numerical columns for correlation
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            print("No numerical columns found for correlation analysis.")
            return
            
        # Calculate correlation matrix
        correlation_matrix = numerical_data.corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Print highly correlated pairs
        self._print_high_correlations(correlation_matrix)
    
    def generate_pairplot(self, data: pd.DataFrame):
        """
        Generate and display a pair plot of the selected features.
        Parameters:
        data (pd.DataFrame): The dataframe containing the data to be analyzed.
        Returns:
        None: This method generates and displays a pair plot.
        """
        # Select only numerical columns for pair plot
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            print("No numerical columns found for pair plot.")
            return
        
        # Limit to first 10 columns if too many (for performance)
        if len(numerical_data.columns) > 10:
            print(f"Too many columns ({len(numerical_data.columns)}). Using first 10 for pair plot.")
            numerical_data = numerical_data.iloc[:, :10]
        
        plt.figure(figsize=(12, 10))
        sns.pairplot(numerical_data)
        plt.suptitle('Pair Plot', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def _print_high_correlations(self, corr_matrix, threshold=0.7):
        """
        Print pairs of features with high correlation.
        """
        print(f"\nHighly Correlated Features (|correlation| >= {threshold}):")
        print("-" * 60)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })
        
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            for _, row in corr_df.iterrows():
                print(f"{row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.3f}")
        else:
            print(f"No correlations found above threshold {threshold}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame):
        """
        Alternative method name for backward compatibility.
        """
        self.generate_correlation_heatmap(data)

# Enhanced version with more analysis options
class AdvancedMultiVeriateAnalysis(MultiVeriateAnalysisTemplate):
    def __init__(self, correlation_threshold=0.7, max_features_pairplot=8):
        self.correlation_threshold = correlation_threshold
        self.max_features_pairplot = max_features_pairplot
    
    def generate_correlation_heatmap(self, data: pd.DataFrame):
        """Enhanced correlation heatmap with statistical significance"""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            print("No numerical columns found for correlation analysis.")
            return
        
        # Calculate correlation and p-values
        from scipy.stats import pearsonr
        
        corr_matrix = numerical_data.corr()
        
        # Create subplots for correlation and significance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Correlation heatmap
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                   center=0, ax=ax1)
        ax1.set_title('Correlation Matrix')
        
        # Significance heatmap
        p_values = np.zeros_like(corr_matrix)
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    _, p_val = pearsonr(numerical_data.iloc[:, i], numerical_data.iloc[:, j])
                    p_values[i, j] = p_val
        
        # Mask non-significant correlations
        mask = p_values > 0.05
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                   cmap='coolwarm', center=0, ax=ax2)
        ax2.set_title('Significant Correlations (p < 0.05)')
        
        plt.tight_layout()
        plt.show()
        
        self._print_high_correlations(corr_matrix, self.correlation_threshold)
    
    def generate_pairplot(self, data: pd.DataFrame):
        """Enhanced pair plot with correlation coefficients"""
        numerical_data = data.select_dtypes(include=[np.number])
        
        if numerical_data.empty:
            print("No numerical columns found for pair plot.")
            return
        
        # Limit features for performance
        if len(numerical_data.columns) > self.max_features_pairplot:
            print(f"Using top {self.max_features_pairplot} features by variance for pair plot.")
            # Select features with highest variance
            variances = numerical_data.var().sort_values(ascending=False)
            top_features = variances.head(self.max_features_pairplot).index
            numerical_data = numerical_data[top_features]
        
        # Create pair plot with custom diagonal
        g = sns.PairGrid(numerical_data)
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.scatterplot, alpha=0.6)
        g.map_diag(sns.histplot, kde=True)
        
        plt.suptitle('Enhanced Pair Plot', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def _print_high_correlations(self, corr_matrix, threshold):
        """Enhanced correlation summary with categories"""
        print(f"\nCorrelation Analysis Summary (|correlation| >= {threshold}):")
        print("=" * 70)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    # Categorize correlation strength
                    if abs(corr_val) >= 0.9:
                        strength = "Very Strong"
                    elif abs(corr_val) >= 0.7:
                        strength = "Strong"
                    elif abs(corr_val) >= 0.5:
                        strength = "Moderate"
                    else:
                        strength = "Weak"
                    
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                        'Strength': strength,
                        'Direction': 'Positive' if corr_val > 0 else 'Negative'
                    })
        
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
            
            for _, row in corr_df.iterrows():
                print(f"{row['Feature 1']} ↔ {row['Feature 2']}")
                print(f"  Correlation: {row['Correlation']:.3f} ({row['Strength']}, {row['Direction']})")
                print()
        else:
            print(f"No correlations found above threshold {threshold}")

if __name__ == "__main__":
    # Example usage of the MultiVeriateAnalysis class.
    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')
    
    # Basic multivariate analysis
    # multivariate_analyzer = MultiVeriateAnalysis()
    # selected_features = df[['V1', 'V2', 'V3', 'V4', 'V5']]  # Select relevant columns
    # multivariate_analyzer.analyze(selected_features)
    
    # Advanced analysis with custom settings
    # advanced_analyzer = AdvancedMultiVeriateAnalysis(correlation_threshold=0.6, max_features_pairplot=6)
    # advanced_analyzer.analyze(selected_features)
    
    pass