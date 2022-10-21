# Test the outcome cache
from cacheConfig import OutcomeCache

def oc_test():
    oc = OutcomeCache()

    assert oc.lookup([1, 2, 3]) == None
    oc.add([1, 2, 3], 4)
    assert oc.lookup([1, 2, 3]) == 4
    assert oc.lookup([1, 2, 3, 4]) == None

    assert oc.lookup([5, 6, 7]) == None
    oc.add([5, 6, 7], 8)
    assert oc.lookup([5, 6, 7]) == 8

    assert oc.lookup([]) == None
    oc.add([], 0)
    assert oc.lookup([]) == 0

    assert oc.lookup([1, 2]) == None
    oc.add([1, 2], 3)
    assert oc.lookup([1, 2]) == 3
    assert oc.lookup([1, 2, 3]) == 4

    assert oc.lookup_superset([1]) == 3 or oc.lookup_superset([1]) == 4
    assert oc.lookup_superset([1, 2]) == 3 or oc.lookup_superset([1, 2]) == 4
    assert oc.lookup_superset([5]) == 8
    assert oc.lookup_superset([5, 6]) == 8
    assert oc.lookup_superset([6, 7]) == 8
    assert oc.lookup_superset([7]) == 8
    assert oc.lookup_superset([]) != None

    assert oc.lookup_superset([9]) == None
    assert oc.lookup_superset([7, 9]) == None
    assert oc.lookup_superset([-5, 1]) == None
    assert oc.lookup_superset([1, 2, 3, 9]) == None
    assert oc.lookup_superset([4, 5, 6, 7]) == None

    assert oc.lookup_subset([]) == 0
    assert oc.lookup_subset([1, 2, 3]) == 4
    assert oc.lookup_subset([1, 2, 3, 4]) == 4
    assert oc.lookup_subset([1, 3]) == None
    assert oc.lookup_subset([1, 2]) == 3

    assert oc.lookup_subset([-5, 1]) == None
    assert oc.lookup_subset([-5, 1, 2]) == 3
    assert oc.lookup_subset([-5]) == 0

