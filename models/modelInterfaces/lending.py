# Sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import secrets
import random

# Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')

# Reading the dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('lending.csv')
print('Shape before deleting duplicate values:', dataset.shape)

# Removing duplicate rows if any
dataset=dataset.drop_duplicates()
print('Shape After deleting duplicate values:', dataset.shape)

# Printing sample data
# Start observing the Quantitative/Categorical/Qualitative variables
dataset.head(10)

target = dataset["loan_status"]
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('loan_status', axis=1)
x_test = test_dataset.drop('loan_status', axis=1)

numerical = ['emp_length', 'annual_inc', 'open_acc',  'credit_years']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)