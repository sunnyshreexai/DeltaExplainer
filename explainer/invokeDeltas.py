from testConfig import *
from testOutcome import *
from candidates import *
from testModels import *
from setval import *
import dice_ml #for puturbations
from dice_ml.utils import helpers # helper functions


def DC():
    global counterfactual
    counterfactual = []
    # Test the outcome cache
    oc_test()
    # Define our own DD class, with its own test method
    class MyDD(DD):
        def _test_b(self, c):
            print("------DELTAS", c)
            status, cf = driver(c)
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
    (c, c1, c2) = mydd.dd(deltas)  # Invoke DD
    print("The 1-minimal failure-inducing difference is", c)
    print(c1, "passes,", c2, "fails")
    print(counterfactual)
    return counterfactual