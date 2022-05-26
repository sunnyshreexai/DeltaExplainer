from testConfig import *
from testOutcome import *
from candidates import *
from testModels import *
from deltaDriver import *
import dice_ml #for puturbations
from dice_ml.utils import helpers # helper functions

# Class for setting values
class Our_exp():
      
  # Retrieves instance variable    
  def __init__(self, query_instance, total_CFs = 1, desired_class = "opposite", features = []): 
    self.query_instance = query_instance
    self.total_CFs = total_CFs
    self.desired_class = desired_class
    self.features = features

  def getInstance(self):
    return self.query_instance

  def setFeatures(self, x):
    self.features = x

  def getFeatures(self, x):
    return self.features

  def generate_counterfactuals(self, x):
    counterfactual = pd.DataFrame(columns=['age', 'workclass', 'education', 'marital_status', 'occupation', 'race', 'gender', 'hours_per_week', 'income'])
    i = 0
    while(i < 1):
      print("*************************************** Finding counterfactual number : ", i+1, " ****************************************************************")
      cf = DC()
      counterfactual = counterfactual.append(cf, ignore_index=True)
      i = i+1
    return counterfactual