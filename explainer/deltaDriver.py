from testConfig import *
from testOutcome import *
from candidates import *
from setval import *
import dice_ml #for puturbations
from dice_ml.utils import helpers # helper functions
from testModels import model
def driver(deltas):

  d = dice_ml.Data(dataframe=train_dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
  m = dice_ml.Model(model=model, backend=backend)
  exp = dice_ml.Dice(d, m, method="random")
  newfeatures = []
  #print('Deltas:1 ', deltas)
  def perturbation(deltas, features, cont_features, outcome):
    instance = example.getInstance().copy()
    #print(" ins_per - ", instance)
    #print('Deltas:2 ', deltas)
    features = features
    for i in deltas:
      x = features[i]
      newfeatures.append(x)
    print(newfeatures)
    d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_features, outcome_name=outcome)
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(d, m, method="random")
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
      #print("old instance - ", org_instance)
      #print(" -- new Instance - ", new_instance)
      #print(org_pred[0], " VS ",  new_pred)
      return 0, cf
    elif org_pred[0] ==  new_pred:
      #new_instance["income"] = new_pred[0]
      reg = new_instance
      #print("old instance - ", org_instance)
      #print(" -- new Instance - ", new_instance)
      #print(org_pred[0], " VS ",  new_pred)
      return 1, reg
    else: 
      #new_instance["income"] = new_pred[0]
      regUn = new_instance
      return 3, regUn
