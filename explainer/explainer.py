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
from testConfig import *
from testOutcome import *
from candidates import *
from testModels import *
from setval import *
from invodeDeltas import *
# import DiCE
import dice_ml
from dice_ml.utils import helpers # helper functions

results = pd.DataFrame(columns=columns)
numberOfCFs = 1
instance = x_test[0:1]
example = Our_exp(instance)
z = example.generate_counterfactuals(numberOfCFs)
results = results.append(instance)
results = results.append(z)
print(results)