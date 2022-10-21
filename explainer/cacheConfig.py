class OutcomeCache:
    # This class holds test outcomes for configurations.  This avoids
    # running the same test twice.

    # The outcome cache is implemented as a tree.  Each node points
    # to the outcome of the remaining list.
    #
    # Example: ([1, 2, 3], PASS), ([1, 2], FAIL), ([1, 4, 5], FAIL):
    #
    #      (2, FAIL)--(3, PASS)
    #     /
    # (1, None)
    #     \
    #      (4, None)--(5, FAIL)

    def __init__(self):
        self.tail = {}  # Points to outcome of tail
        self.result = None  # Result so far

    def add(self, c, result):
        """Add (C, RESULT) to the cache.  C must be a list of scalars."""
        cs = c[:]
        cs.sort()

        p = self
        for start in range(len(c)):
            if c[start] not in p.tail:
                p.tail[c[start]] = OutcomeCache()
            p = p.tail[c[start]]

        p.result = result

    def lookup(self, c):
        """Return RESULT if (C, RESULT) is in the cache; None, otherwise."""
        p = self
        for start in range(len(c)):
            if c[start] not in p.tail:
                return None
            p = p.tail[c[start]]

        return p.result

    def lookup_superset(self, c, start=0):
        """Return RESULT if there is some (C', RESULT) in the cache with
        C' being a superset of C or equal to C.  Otherwise, return None."""

        # FIXME: Make this non-recursive!
        if start >= len(c):
            if self.result:
                return self.result
            elif self.tail != {}:
                # Select some superset SUNNY 1.0
                superset = self.tail[list(self.tail.keys())[0]]
                return superset.lookup_superset(c, start + 1)
            else:
                return None

        if c[start] in self.tail:
            return self.tail[c[start]].lookup_superset(c, start + 1)

        # Let K0 be the largest element in TAIL such that K0 <= C[START]
        k0 = None
        for k in self.tail.keys():
            if (k0 == None or k > k0) and k <= c[start]:
                k0 = k

        if k0 != None:
            return self.tail[k0].lookup_superset(c, start)

        return None

    def lookup_subset(self, c):
        """Return RESULT if there is some (C', RESULT) in the cache with
        C' being a subset of C or equal to C.  Otherwise, return None."""
        p = self
        for start in range(len(c)):
            if c[start] in p.tail:
                p = p.tail[c[start]]

        return p.result
