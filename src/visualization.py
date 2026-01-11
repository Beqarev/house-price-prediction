"""
Visualization module for housing price prediction.
Contains functions to generate the 5 required visualization types.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def save_plot(fig, filename, output_dir='reports/figures'):
    """
    Save a matplotlib figure to the reports/figures directory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        Name of the output file.
    output_dir : str
        Directory to save the plot.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {filepath}")


def plot_price_distribution(data, target_col='MedHouseVal', output_dir='reports/figures'):
    """
    Create a distribution plot: Histogram with KDE for the target variable (Price).
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the target variable.
    target_col : str
        Name of the target column (default: 'MedHouseVal').
    output_dir : str
        Directory to save the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data[target_col], kde=True, ax=ax, bins=50)
    ax.set_title('Distribution of House Prices (with KDE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('House Price (Median House Value)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, '01_price_distribution.png', output_dir)
    return fig


def plot_correlation_heatmap(data, output_dir='reports/figures'):
    """
    Create a correlation heatmap showing correlations between all numerical features.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing numerical features.
    output_dir : str
        Directory to save the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    numerical_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_plot(fig, '02_correlation_heatmap.png', output_dir)
    return fig


def plot_price_vs_income(data, price_col='MedHouseVal', income_col='MedInc', 
                         output_dir='reports/figures'):
    """
    Create a scatter plot: Price vs. Median Income with a regression trend line.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing price and income columns.
    price_col : str
        Name of the price column (default: 'MedHouseVal').
    income_col : str
        Name of the income column (default: 'MedInc').
    output_dir : str
        Directory to save the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(data=data, x=income_col, y=price_col, alpha=0.5, ax=ax)
    sns.regplot(data=data, x=income_col, y=price_col, scatter=False, 
                color='red', line_kws={'linewidth': 2}, ax=ax)
    
    ax.set_title('House Price vs. Median Income (with Regression Trend)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Median Income', fontsize=12)
    ax.set_ylabel('House Price (Median House Value)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, '03_price_vs_income.png', output_dir)
    return fig


def plot_price_by_age_group(data, price_col='MedHouseVal', age_col='HouseAge',
                            output_dir='reports/figures'):
    """
    Create a box plot: Price distribution grouped by House Age Group.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing price and age columns.
    price_col : str
        Name of the price column (default: 'MedHouseVal').
    age_col : str
        Name of the age column (default: 'HouseAge').
    output_dir : str
        Directory to save the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    data_copy = data.copy()
    
    # Create age groups
    data_copy['HouseAgeGroup'] = pd.cut(
        data_copy[age_col],
        bins=[0, 10, 20, 30, 40, 50, 100],
        labels=['0-10', '11-20', '21-30', '31-40', '41-50', '50+']
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=data_copy, x='HouseAgeGroup', y=price_col, ax=ax)
    ax.set_title('Price Distribution by House Age Group', fontsize=14, fontweight='bold')
    ax.set_xlabel('House Age Group (years)', fontsize=12)
    ax.set_ylabel('House Price (Median House Value)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot(fig, '04_price_by_age_group.png', output_dir)
    return fig


def plot_pair_plot(data, columns=None, output_dir='reports/figures'):
    """
    Create a pair plot: A subset of variables to show multivariate relationships.
    Default variables: Price, Income, AveRooms, HouseAge.
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables.
    columns : list, optional
        List of column names to include in the pair plot.
        If None, uses ['MedHouseVal', 'MedInc', 'AveRooms', 'HouseAge'].
    output_dir : str
        Directory to save the plot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if columns is None:
        columns = ['MedHouseVal', 'MedInc', 'AveRooms', 'HouseAge']
    
    # Filter to only include columns that exist in the data
    available_columns = [col for col in columns if col in data.columns]
    
    if len(available_columns) < 2:
        print("Warning: Not enough columns available for pair plot")
        return None
    
    data_subset = data[available_columns]
    
    fig = sns.pairplot(data_subset, diag_kind='kde', plot_kws={'alpha': 0.6})
    fig.fig.suptitle('Pair Plot: Multivariate Relationships', 
                     fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_plot(fig.fig, '05_pair_plot.png', output_dir)
    return fig.fig
