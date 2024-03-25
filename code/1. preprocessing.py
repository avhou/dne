import pandas as pd
import numpy as np
from configparser import ConfigParser
from dataloader import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Read the filepath from the config ini file
config = ConfigParser()
config.read('config.ini')
filepath = config.get('Data', 'SolarFilePath')

# Create an instance of the DataLoader class
data_loader = DataLoader(filepath)

# Load the data
df = data_loader.read_excel(aggregation_type="daily")

# Define the steps for the pipeline
steps = [
    ('scaler', StandardScaler()),
]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline on the data
transformed_data = pipeline.fit_transform(df)

