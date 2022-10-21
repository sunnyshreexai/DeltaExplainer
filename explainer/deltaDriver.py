from cacheConfig import OutcomeCache
from cacheConfigTest import oc_test
from dd import DD
import pandas as pd  

#local Iports
from data import Data
from model import Model
from dice import Dice


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

  def generate_counterfactuals(self, x, dataset, m):
    counterfactual = pd.DataFrame(columns=['age', 'workclass', 'education', 'marital_status', 'occupation', 'race', 'gender', 'hours_per_week', 'income'])
    i = 0
    global train_dataset
    train_dataset = dataset
    global model
    model = m
    while(i < 1):
      print("*************************************** Finding counterfactual number : ", i+1, " ****************************************************************")
      cf = DC()
      counterfactual = counterfactual.append(cf, ignore_index=True)
      i = i+1
    return counterfactual

#to get results in each iteration
def deltaOutcome(deltas):

  d = Data(dataframe=train_dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
  m = Model(model=model, backend="sklearn")
  exp = Dice(d, m, method="random")
  newfeatures = []
  x = []
  #print('Deltas:1 ', deltas)
  def perturbation(deltas):
    instance = example.getInstance().copy()
    #print(" ins_per - ", instance)
    #print('Deltas:2 ', deltas)
    features = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'race', 'gender', 'hours_per_week', 'income']
    for i in deltas:
      x = features[i]
      newfeatures.append(x)
    print(newfeatures)
    d = Data(dataframe=train_dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    m = Model(model=model, backend="sklearn")
    exp = Dice(d, m, method="random")
    e1 = exp.generate_counterfactuals(instance, total_CFs=1, desired_class="opposite", features_to_vary=newfeatures)
    results = e1.cf_examples_list[0].final_cfs_df
    return(results)

  org_instance = example.getInstance().copy()
  org_pred = clf.predict(org_instance)

  new_instance = perturbation(deltas)

  #print("old instance - ", org_instance)
  #print(" -- new Instance - ", new_instance)
  #print(org_pred[0], " VS ",  new_pred)
  NoneType = type(None)
  if type(new_instance) == NoneType:
    regUn = new_instance
    return 3, regUn
  else:  
    new_pred = int(new_instance.income)
    if org_pred[0] !=  new_pred:
      cf = new_instance
      x.append(newfeatures)
      print("old instance - ", org_instance)
      print(" -- new Instance - ", new_instance)
      print(org_pred[0], " VS ",  new_pred)
      return 0, cf
    elif org_pred[0] ==  new_pred:
      #new_instance["income"] = new_pred[0]
      reg = new_instance
      print("old instance - ", org_instance)
      print(" -- new Instance - ", new_instance)
      print(org_pred[0], " VS ",  new_pred)
      return 1, reg
    else: 
      #new_instance["income"] = new_pred[0]
      regUn = new_instance
      return 3, regUn



#To make call
def DC():
    global counterfactual
    counterfactual = []
    # Test the outcome cache
    oc_test()
    # Define our own DD class, with its own test method
    class MyDD(DD):
        def _test_b(self, c):
            print("------DELTAS", c)
            status, cf = deltaOutcome(c)
            print("pass/fail : ", status)
            #print("CF : ", cf)
            if status == 0:
                global counterfactual
                counterfactual = cf
                return self.FAIL
            elif status == 1:
                return self.PASS
            else: 
                return self.UNRESOLVED


        def __init__(self):
            self._test = self._test_b
            DD.__init__(self)

    mydd = MyDD()
    #mydd.debug_test     = 1			# Enable debugging output
    #mydd.debug_dd       = 1			# Enable debugging output
    #mydd.debug_split    = 1			# Enable debugging output
    #mydd.debug_resolve  = 1			# Enable debugging output

    # mydd.cache_outcomes = 0
    # mydd.monotony = 0
    #deltas = []
    #for i in range(0,8):
    #  deltas.append(i)
 
    print("Computing the failure-inducing difference...")
    (c, c1, c2) = mydd.dd([0, 7, 1, 4, 6, 3, 2, 5])  # Invoke DD
    print("The 1-minimal failure-inducing difference is", c)
    print(c1, "passes,", c2, "fails")
    print(counterfactual)
    return counterfactual

