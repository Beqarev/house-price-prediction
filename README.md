# Housing Price Prediction Project

## Project Overview

This project implements a production-grade machine learning system for predicting house prices in California using the California Housing dataset from scikit-learn. The project follows best practices in data science workflow, including data exploration, preprocessing, visualization, and model training.

## Objective

Predict house prices (regression task) based on features like square footage, location, and amenities using the `sklearn.datasets.fetch_california_housing` dataset.

## Project Structure

```
project-housing-prediction/
├── data/
│   ├── raw/                 # Original fetched data
│   └── processed/           # Cleaned/transformed data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualization.ipynb
│   └── 04_machine_learning.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Data cleaning & feature engineering
│   ├── visualization.py     # Visualization functions
│   └── models.py            # Model training and evaluation
├── reports/
│   ├── figures/             # Generated plots (.png)
│   └── results/             # Model metrics (txt/csv)
├── README.md
├── CONTRIBUTIONS.md
└── requirements.txt
```

## Installation

1. Clone this repository or navigate to the project directory.

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebooks

The project is organized into four sequential notebooks that should be run in order:

1. **01_data_exploration.ipynb**: Fetches and explores the California Housing dataset, saves raw data.

2. **02_data_preprocessing.ipynb**: Performs data cleaning, outlier removal, and feature engineering using the `HousingDataProcessor` class.

3. **03_eda_visualization.ipynb**: Generates five required visualizations:
   - Distribution plot (Histogram with KDE) for house prices
   - Correlation heatmap of numerical features
   - Scatter plot: Price vs. Median Income with regression trend line
   - Box plot: Price distribution by House Age Group
   - Pair plot: Multivariate relationships (Price, Income, AveRooms, HouseAge)

4. **04_machine_learning.ipynb**: Trains and evaluates three regression models:
   - Linear Regression (Baseline)
   - Decision Tree Regressor (Comparison)
   - Random Forest Regressor (Advanced/Bonus)

To run the notebooks:
```bash
jupyter notebook notebooks/
```

Or using JupyterLab:
```bash
jupyter lab notebooks/
```

### Using the Source Modules

The core functionality is implemented in the `src/` directory and can be imported and used programmatically:

```python
from src.data_processing import HousingDataProcessor
from src.visualization import plot_price_distribution, plot_correlation_heatmap
from src.models import HousingModelTrainer

# Example usage
processor = HousingDataProcessor(data)
processed_data = processor.process()

trainer = HousingModelTrainer(X, y)
trainer.train_all_models()
results = trainer.evaluate_all_models()
```

## Features

### Data Processing
- Missing value imputation using median
- Duplicate removal
- Outlier detection and removal using IQR method
- Feature engineering: RoomsPerBedroom (Total Rooms / Total Bedrooms)

### Visualizations
- Five comprehensive visualizations for exploratory data analysis
- All plots saved to `reports/figures/` directory

### Machine Learning Models
- Three distinct regression models for comparison
- Comprehensive evaluation metrics (R2 Score, MSE)
- Model comparison DataFrame for easy analysis

## Results Summary

### Final Model Performance

After running the complete pipeline, the final model results are:

| Model | R² Score (Test) | MSE (Test) |
|-------|----------------|------------|
| Linear Regression | 0.7342 | 0.3390 |
| Decision Tree | 0.6607 | 0.4327 |
| **Random Forest** | **0.8216** | **0.2275** |

**Best Model**: Random Forest with R² = 0.8216 (82.16% variance explained)

Model performance metrics are saved to `reports/results/` after running the machine learning notebook. The results include:

- R2 Score (Train and Test)
- Mean Squared Error (Train and Test)
- Model comparison table

The best performing model is automatically identified based on R2 Score on the test set.

For detailed results, see `FINAL_RESULTS.md`.

## Dataset

This project uses the California Housing dataset from scikit-learn (`sklearn.datasets.fetch_california_housing`), which contains:

- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target**: MedHouseVal (Median House Value)

## Dependencies

See `requirements.txt` for the complete list of dependencies. Main packages include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

## Code Quality

- PEP 8 compliant code
- NumPy-style docstrings for all classes and methods
- Modular design with separation of concerns
- Reproducible results (random_state=42)

## License

This project is for academic purposes.

## Contact

For questions or issues, please refer to the project repository.
