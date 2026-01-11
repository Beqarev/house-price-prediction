"""
Data processing module for housing price prediction.
Contains HousingDataProcessor class for cleaning and feature engineering.
"""

import pandas as pd
import numpy as np


class HousingDataProcessor:
    """
    A class for processing housing data including cleaning, outlier removal,
    and feature engineering.
    
    Attributes
    ----------
    data : pd.DataFrame
        The housing dataset to be processed.
    """
    
    def __init__(self, data):
        """
        Initialize the HousingDataProcessor.
        
        Parameters
        ----------
        data : pd.DataFrame
            The raw housing dataset.
        """
        self.data = data.copy()
    
    def handle_missing_values(self):
        """
        Handle missing values by imputing with median for numerical columns.
        
        Returns
        -------
        pd.DataFrame
            Dataset with missing values imputed.
        """
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                median_value = self.data[col].median()
                self.data[col].fillna(median_value, inplace=True)
        return self.data
    
    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        
        Returns
        -------
        pd.DataFrame
            Dataset with duplicates removed.
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed = initial_rows - len(self.data)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        return self.data
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5, exclude_target=True):
        """
        Remove outliers using the Interquartile Range (IQR) method.
        
        Parameters
        ----------
        columns : list, optional
            List of column names to check for outliers.
            If None, checks all numerical columns.
        multiplier : float
            IQR multiplier (default: 1.5). Use 3.0 for less aggressive removal.
        exclude_target : bool
            Whether to exclude target variable from outlier removal (default: True).
        
        Returns
        -------
        pd.DataFrame
            Dataset with outliers removed.
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
            if exclude_target and 'MedHouseVal' in columns:
                columns = columns.drop('MedHouseVal')
        
        initial_rows = len(self.data)
        
        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            self.data = self.data[
                (self.data[col] >= lower_bound) & 
                (self.data[col] <= upper_bound)
            ]
        
        removed = initial_rows - len(self.data)
        if removed > 0:
            print(f"Removed {removed} outlier rows using IQR method (multiplier={multiplier})")
        
        return self.data
    
    def create_rooms_per_bedroom(self):
        """
        Create a new feature: RoomsPerBedroom (Total Rooms / Total Bedrooms).
        This satisfies the Feature Engineering bonus requirement.
        
        Returns
        -------
        pd.DataFrame
            Dataset with the new RoomsPerBedroom feature added.
        """
        if 'AveRooms' in self.data.columns and 'AveBedrms' in self.data.columns:
            self.data['RoomsPerBedroom'] = (
                self.data['AveRooms'] / 
                (self.data['AveBedrms'] + 1e-6)  # Avoid division by zero
            )
            print("Created RoomsPerBedroom feature")
        else:
            print("Warning: Required columns (AveRooms, AveBedrms) not found")
        
        return self.data
    
    def create_additional_features(self):
        """
        Create additional engineered features to improve model performance.
        
        Returns
        -------
        pd.DataFrame
            Dataset with additional engineered features.
        """
        # Population per room
        if 'Population' in self.data.columns and 'AveRooms' in self.data.columns:
            self.data['PopulationPerRoom'] = (
                self.data['Population'] / (self.data['AveRooms'] + 1e-6)
            )
            print("Created PopulationPerRoom feature")
        
        # Occupancy rate
        if 'AveOccup' in self.data.columns and 'AveRooms' in self.data.columns:
            self.data['OccupancyRate'] = (
                self.data['AveOccup'] / (self.data['AveRooms'] + 1e-6)
            )
            print("Created OccupancyRate feature")
        
        # Income per person
        if 'MedInc' in self.data.columns and 'Population' in self.data.columns:
            self.data['IncomePerPerson'] = (
                self.data['MedInc'] * 10000 / (self.data['Population'] + 1e-6)
            )
            print("Created IncomePerPerson feature")
        
        # Distance from center (using latitude and longitude)
        if 'Latitude' in self.data.columns and 'Longitude' in self.data.columns:
            # California approximate center
            ca_center_lat = 36.7783
            ca_center_lon = -119.4179
            self.data['DistanceFromCenter'] = np.sqrt(
                (self.data['Latitude'] - ca_center_lat)**2 + 
                (self.data['Longitude'] - ca_center_lon)**2
            )
            print("Created DistanceFromCenter feature")
        
        return self.data
    
    def process(self, create_additional_features=True, outlier_multiplier=3.0):
        """
        Execute the complete data processing pipeline.
        
        Parameters
        ----------
        create_additional_features : bool
            Whether to create additional engineered features (default: True).
        outlier_multiplier : float
            IQR multiplier for outlier removal. 1.5 = aggressive, 3.0 = less aggressive (default: 3.0).
            Set to None to skip outlier removal.
        
        Returns
        -------
        pd.DataFrame
            Fully processed dataset ready for modeling.
        """
        self.handle_missing_values()
        self.remove_duplicates()
        if outlier_multiplier is not None:
            self.remove_outliers_iqr(multiplier=outlier_multiplier)
        self.create_rooms_per_bedroom()
        if create_additional_features:
            self.create_additional_features()
        return self.data
