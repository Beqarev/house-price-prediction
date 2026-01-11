"""
Machine learning models module for housing price prediction.
Contains HousingModelTrainer class for training and evaluating models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path


class HousingModelTrainer:
    """
    A class for training and evaluating multiple regression models
    for housing price prediction.
    
    Attributes
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Test target.
    models : dict
        Dictionary of trained models.
    results : pd.DataFrame
        DataFrame containing model comparison metrics.
    """
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the HousingModelTrainer and split the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or array-like
            Target variable.
        test_size : float
            Proportion of dataset to include in the test split (default: 0.2).
        random_state : int
            Random state for reproducibility (default: 42).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.models = {}
        self.scalers = {}
        self.poly_transformers = {}
        self.results = None
    
    def train_linear_regression(self, use_scaling=True, use_polynomial=True, degree=2, use_ridge=False, alpha=1.0):
        """
        Train a Linear Regression model (Baseline model).
        
        Parameters
        ----------
        use_scaling : bool
            Whether to apply StandardScaler to features (default: True).
        use_polynomial : bool
            Whether to use polynomial features (default: True).
        degree : int
            Degree of polynomial features (default: 2).
        use_ridge : bool
            Whether to use Ridge regression instead of Linear Regression (default: False).
        alpha : float
            Regularization strength for Ridge (default: 1.0).
        
        Returns
        -------
        LinearRegression or Ridge
            Trained Linear Regression model.
        """
        X_train_processed = self.X_train
        X_test_processed = self.X_test
        
        # Apply polynomial features
        if use_polynomial:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_processed = poly.fit_transform(X_train_processed)
            X_test_processed = poly.transform(X_test_processed)
            self.poly_transformers['Linear Regression'] = poly
            print(f"  Applied polynomial features (degree={degree}), shape: {X_train_processed.shape}")
        
        # Apply scaling
        if use_scaling:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_processed)
            X_test_processed = scaler.transform(X_test_processed)
            self.scalers['Linear Regression'] = scaler
        
        # Choose model
        if use_ridge:
            model = Ridge(alpha=alpha, max_iter=1000)
        else:
            model = LinearRegression()
        
        model.fit(X_train_processed, self.y_train)
        self.models['Linear Regression'] = model
        return model
    
    def train_decision_tree(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Train a Decision Tree Regressor model (Comparison model).
        
        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree (default: 15 to prevent overfitting).
        min_samples_split : int
            Minimum number of samples required to split a node (default: 10).
        min_samples_leaf : int
            Minimum number of samples required at a leaf node (default: 5).
        random_state : int
            Random state for reproducibility (default: 42).
        
        Returns
        -------
        DecisionTreeRegressor
            Trained Decision Tree Regressor model.
        """
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            criterion='squared_error'
        )
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        return model
    
    def train_random_forest(self, n_estimators=300, max_depth=None, min_samples_split=2, 
                           min_samples_leaf=1, max_features='sqrt', random_state=42):
        """
        Train a Random Forest Regressor model (Advanced/Bonus model).
        
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest (default: 200).
        max_depth : int
            Maximum depth of the trees (default: 20).
        min_samples_split : int
            Minimum number of samples required to split a node (default: 5).
        min_samples_leaf : int
            Minimum number of samples required at a leaf node (default: 2).
        max_features : str or int
            Number of features to consider for best split (default: 'sqrt').
        random_state : int
            Random state for reproducibility (default: 42).
        
        Returns
        -------
        RandomForestRegressor
            Trained Random Forest Regressor model.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        return model
    
    def train_all_models(self):
        """
        Train all three models: Linear Regression, Decision Tree, and Random Forest.
        
        Returns
        -------
        dict
            Dictionary of trained models.
        """
        print("Training Linear Regression...")
        self.train_linear_regression(use_polynomial=True, degree=2, use_ridge=True, alpha=0.1)
        
        print("Training Decision Tree...")
        self.train_decision_tree(max_depth=None, min_samples_split=2, min_samples_leaf=1)
        
        print("Training Random Forest...")
        self.train_random_forest(n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1)
        
        print("All models trained successfully!")
        return self.models
    
    def evaluate_model(self, model, model_name):
        """
        Evaluate a single model and return metrics.
        
        Parameters
        ----------
        model : sklearn model
            Trained model to evaluate.
        model_name : str
            Name of the model.
        
        Returns
        -------
        dict
            Dictionary containing R2 score and MSE.
        """
        # Apply polynomial features if used for this model
        if model_name in self.poly_transformers:
            X_train_eval = self.poly_transformers[model_name].transform(self.X_train)
            X_test_eval = self.poly_transformers[model_name].transform(self.X_test)
        else:
            X_train_eval = self.X_train
            X_test_eval = self.X_test
        
        # Apply scaling if used for this model
        if model_name in self.scalers:
            X_train_eval = self.scalers[model_name].transform(X_train_eval)
            X_test_eval = self.scalers[model_name].transform(X_test_eval)
        
        y_pred_train = model.predict(X_train_eval)
        y_pred_test = model.predict(X_test_eval)
        
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        
        metrics = {
            'Model': model_name,
            'R2_Score_Train': r2_train,
            'R2_Score_Test': r2_test,
            'MSE_Train': mse_train,
            'MSE_Test': mse_test
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  R2 Score (Train): {r2_train:.4f}")
        print(f"  R2 Score (Test): {r2_test:.4f}")
        print(f"  MSE (Train): {mse_train:.4f}")
        print(f"  MSE (Test): {mse_test:.4f}")
        
        return metrics
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models and create a comparison DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame comparing metrics of all models.
        """
        results_list = []
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, model_name)
            results_list.append(metrics)
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def save_results(self, output_dir='reports/results'):
        """
        Save model comparison results to CSV and text files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the results.
        """
        if self.results is None:
            print("No results to save. Run evaluate_all_models() first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / 'model_comparison.csv'
        self.results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Save as text file
        txt_path = output_path / 'model_comparison.txt'
        with open(txt_path, 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(self.results.to_string())
            f.write("\n\n")
            f.write("Best Model (by R2 Score on Test Set):\n")
            best_model = self.results.loc[self.results['R2_Score_Test'].idxmax()]
            f.write(f"  {best_model['Model']}: R2 = {best_model['R2_Score_Test']:.4f}\n")
        
        print(f"Results saved to {txt_path}")
