from testConfig import *
from testOutcome import *
# Main Delta Debugging algorithm.
class DD:

    # Test outcomes.
    PASS = "PASS"
    FAIL = "FAIL"
    UNRESOLVED = "UNRESOLVED"

    # Resolving directions.
    ADD = "ADD"  # Add deltas to resolve
    REMOVE = "REMOVE"  # Remove deltas to resolve

    # Debugging output (set to 1 to enable)
    debug_test = 0
    debug_dd = 0
    debug_split = 0
    debug_resolve = 0

    def __init__(self):
        self.__resolving = 0
        self.__last_reported_length = 0
        self.monotony = 0
        self.outcome_cache = OutcomeCache()
        self.cache_outcomes = 1
        self.minimize = 1
        self.maximize = 1
        self.assume_axioms_hold = 1

    # Helpers
    def __listminus(self, c1, c2):
        """Return a list of all elements of C1 that are not in C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        c = []
        for delta in c1:
            if delta not in s2:
                c.append(delta)

        return c

    def __listintersect(self, c1, c2):
        """Return the common elements of C1 and C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        c = []
        for delta in c1:
            if delta in s2:
                c.append(delta)

        return c

    def __listunion(self, c1, c2):
        """Return the union of C1 and C2."""
        s1 = {}
        for delta in c1:
            s1[delta] = 1

        c = c1[:]
        for delta in c2:
            if delta not in s1:
                c.append(delta)

        return c

    def __listsubseteq(self, c1, c2):
        """Return 1 if C1 is a subset or equal to C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        for delta in c1:
            if delta not in s2:
                return 0

        return 1

    # Output
    def coerce(self, c):
        """Return the configuration C as a compact string"""
        # Default: use printable representation
        return c

    def pretty(self, c):
        """Like coerce(), but sort beforehand"""
        sorted_c = c[:]
        sorted_c.sort()
        return self.coerce(sorted_c)

    # Testing
    def test(self, c):
        """Test the configuration C.  Return PASS, FAIL, or UNRESOLVED"""
        c.sort()

        # If we had this test before, return its result
        if self.cache_outcomes:
            cached_result = self.outcome_cache.lookup(c)
            if cached_result != None:
                return cached_result

        if self.monotony:
            # Check whether we had a passing superset of this test before
            cached_result = self.outcome_cache.lookup_superset(c)
            if cached_result == self.PASS:
                return self.PASS

            cached_result = self.outcome_cache.lookup_subset(c)
            if cached_result == self.FAIL:
                return self.FAIL

        if self.debug_test:
            print("test(" , self.coerce(c) , ")...")

        outcome = self._test(c)

        if self.debug_test:
            print("test(" , self.coerce(c) , ") = " , outcome)

        if self.cache_outcomes:
            self.outcome_cache.add(c, outcome)

        return outcome

    def _test(self, c):
        """Stub to overload in subclasses"""
        return self.UNRESOLVED  # Placeholder

    # Splitting
    def split(self, c, n):
        """Split C into [C_1, C_2, ..., C_n]."""
        if self.debug_split:
            print("split(" , self.coerce(c) , ", " , n , ")...")

        outcome = self._split(c, n)

        if self.debug_split:
            print("split(" , self.coerce(c) , ", " , n , ") = " , outcome)

        return outcome

    def _split(self, c, n):
        """Stub to overload in subclasses"""
        subsets = []
        start = 0
        for i in range(n):
            subset = c[start:start + int((len(c) - start) / (n - i))]
            subsets.append(subset)
            start = start + len(subset)
        return subsets

    # Resolving
    def resolve(self, csub, c, direction):
        """If direction == ADD, resolve inconsistency by adding deltas
             to CSUB.  Otherwise, resolve by removing deltas from CSUB."""

        if self.debug_resolve:
            print("resolve(" , csub , ", " , self.coerce(c) , ", " , direction , ")...")

        outcome = self._resolve(csub, c, direction)

        if self.debug_resolve:
            print("resolve(" , csub , ", " , self.coerce(c) , ", " , direction , ") = " , outcome)

        return outcome

    def _resolve(self, csub, c, direction):
        """Stub to overload in subclasses."""
        # By default, no way to resolve
        return None

    # Test with fixes
    def test_and_resolve(self, csub, r, c, direction):
        """Repeat testing CSUB + R while unresolved."""

        initial_csub = csub[:]
        c2 = self.__listunion(r, c)

        csubr = self.__listunion(csub, r)
        t = self.test(csubr)

        # necessary to use more resolving mechanisms which can reverse each
        # other, can (but needn't) be used in subclasses
        self._resolve_type = 0

        while t == self.UNRESOLVED:
            self.__resolving = 1
            csubr = self.resolve(csubr, c, direction)

            if csubr == None:
                # Nothing left to resolve
                break

            if len(csubr) >= len(c2):
                # Added everything: csub == c2. ("Upper" Baseline)
                # This has already been tested.
                csubr = None
                break

            if len(csubr) <= len(r):
                # Removed everything: csub == r. (Baseline)
                # This has already been tested.
                csubr = None
                break

            t = self.test(csubr)

        self.__resolving = 0
        if csubr == None:
            return self.UNRESOLVED, initial_csub

        # assert t == self.PASS or t == self.FAIL
        csub = self.__listminus(csubr, r)
        return t, csub

    # Inquiries
    def resolving(self):
        """Return 1 while resolving."""
        return self.__resolving

    # Logging
    def report_progress(self, c, title):
        if len(c) != self.__last_reported_length:
            print()
            print(title + ": ", len(c),  " deltas left:", self.coerce(c))
            print()
            self.__last_reported_length = len(c)

   

    def test_mix(self, csub, c, direction):
        if self.minimize:
            (t, csub) = self.test_and_resolve(csub, [], c, direction)
            if t == self.FAIL:
                return (t, csub)

        if self.maximize:
            csubbar = self.__listminus(self.CC, csub)
            cbar = self.__listminus(self.CC, c)
            if direction == self.ADD:
                directionbar = self.REMOVE
            else:
                directionbar = self.ADD

            (tbar, csubbar) = self.test_and_resolve(csubbar, [], cbar,
                                                    directionbar)

            csub = self.__listminus(self.CC, csubbar)

            if tbar == self.PASS:
                t = self.FAIL
            elif tbar == self.FAIL:
                t = self.PASS
            else:
                t = self.UNRESOLVED

        return (t, csub)

    # Delta Debugging (new ISSTA version)
    def ddgen(self, c, minimize, maximize):
        """Return a 1-minimal failing subset of C"""

        self.minimize = minimize
        self.maximize = maximize

        n = 2
        self.CC = c

        if self.debug_dd:
            print("dd(" , self.pretty(c) , ", " , n , ")...")

        outcome = self._dd(c, n)

        if self.debug_dd:
            print("dd(" , self.pretty(c) , ", " , n , ") = " , outcome)

        return outcome

    def _dd(self, c, n):
        """Stub to overload in subclasses"""

        assert self.test([]) == self.PASS

        run = 1
        cbar_offset = 0

        # We replace the tail recursion from the paper by a loop
        while 1:
            tc = self.test(c)
            assert tc == self.FAIL or tc == self.UNRESOLVED

            if n > len(c):
                # No further minimizing
                print("dd: done")
                return c

            self.report_progress(c, "dd")

            cs = self.split(c, n)
            print()
            print("dd (run #" , run , "): trying ", end='')
            for i in range(n):
                if i > 0:
                    print("+", end='')
                print(len(cs[i]), end='')
            print()

            c_failed = 0
            cbar_failed = 0

            next_c = c[:]
            next_n = n

            # Check subsets
            for i in range(n):
                if self.debug_dd:
                    print("dd: trying", self.pretty(cs[i]))

                (t, cs[i]) = self.test_mix(cs[i], c, self.REMOVE)

                if t == self.FAIL:
                    # Found
                    if self.debug_dd:
                        print("dd: found", len(cs[i]), "deltas:",)
                        print(self.pretty(cs[i]))

                    c_failed = 1
                    next_c = cs[i]
                    next_n = 2
                    cbar_offset = 0
                    self.report_progress(next_c, "dd")
                    break

            if not c_failed:
                # Check complements
                cbars = n * [self.UNRESOLVED]

                # print "cbar_offset =", cbar_offset

                for j in range(n):
                    i = int((j + cbar_offset) % n)
                    cbars[i] = self.__listminus(c, cs[i])
                    t, cbars[i] = self.test_mix(cbars[i], c, self.ADD)

                    doubled = self.__listintersect(cbars[i], cs[i])
                    if doubled != []:
                        cs[i] = self.__listminus(cs[i], doubled)

                    if t == self.FAIL:
                        if self.debug_dd:
                            print("dd: reduced to ", len(cbars[i]),)
                            print(" deltas:",)
                            print(self.pretty(cbars[i]))

                        cbar_failed = 1
                        next_c = self.__listintersect(next_c, cbars[i])
                        next_n = next_n - 1
                        self.report_progress(next_c, "dd")

                        # In next run, start removing the following subset
                        cbar_offset = i
                        break

            if not c_failed and not cbar_failed:
                if n >= len(c):
                    # No further minimizing
                    print()
                    print("dd: done")
                    print()
                    return c

                next_n = min(len(c), n * 2)
                print()
                print("dd: increase granularity to", next_n)
                print()
                cbar_offset = (cbar_offset * next_n) / n

            c = next_c
            n = next_n
            run = run + 1

    def ddmin(self, c):
        return self.ddgen(c, 1, 0)

    def ddmax(self, c):
        return self.ddgen(c, 0, 1)

    def ddmix(self, c):
        return self.ddgen(c, 1, 1)

    # General delta debugging (new TSE version)
    def dddiff(self, c):
        n = 2

        if self.debug_dd:
            print("dddiff(" , self.pretty(c) , ", " , n , ")...")

        outcome = self._dddiff([], c, n)

        if self.debug_dd:
            print("dddiff(" , self.pretty(c) , ", " , n , ") = " , outcome)

        return outcome

    def _dddiff(self, c1, c2, n):
        run = 1
        cbar_offset = 0

        # We replace the tail recursion from the paper by a loop
        while 1:
            if self.debug_dd:
                print("dd: c1 =", self.pretty(c1))
                print("dd: c2 =", self.pretty(c2))

            if self.assume_axioms_hold:
                t1 = self.PASS
                t2 = self.FAIL
            else:
                t1 = self.test(c1)
                t2 = self.test(c2)

            assert t1 == self.PASS
            assert t2 == self.FAIL
            assert self.__listsubseteq(c1, c2)

            c = self.__listminus(c2, c1)

            if self.debug_dd:
                print("dd: c2 - c1 =", self.pretty(c))

            if n > len(c):
                # No further minimizing
                print("dd: done")
                return (c, c1, c2)
            print()
            self.report_progress(c, "dd")
            print()

            cs = self.split(c, n)

            print()
            print("dd (run # ", run, "): trying ", end='')
            for i in range(n):
                if i > 0:
                    print("+", end='')
                print(len(cs[i]), end='')
            print()

            progress = 0

            next_c1 = c1[:]
            next_c2 = c2[:]
            next_n = n

            # Check subsets
            for j in range(n):
                i = int((j + cbar_offset) % n)

                if self.debug_dd:
                    print("dd: trying", self.pretty(cs[i]))

                (t, csub) = self.test_and_resolve(cs[i], c1, c, self.REMOVE)
                csub = self.__listunion(c1, csub)

                if t == self.FAIL and t1 == self.PASS:
                    # Found
                    progress = 1
                    next_c2 = csub
                    next_n = 2
                    cbar_offset = 0

                    if self.debug_dd:
                        print("dd: reduce c2 to", len(next_c2), "deltas:",)
                        print()
                        self.pretty(next_c2)
                    break

                if t == self.PASS and t2 == self.FAIL:
                    # Reduce to complement
                    progress = 1
                    next_c1 = csub
                    next_n = max(next_n - 1, 2)
                    cbar_offset = i

                    if self.debug_dd:
                        print("dd: increase c1 to", len(next_c1), "deltas:",)
                        print()
                        self.pretty(next_c1)
                    break

                csub = self.__listminus(c, cs[i])
                (t, csub) = self.test_and_resolve(csub, c1, c, self.ADD)
                csub = self.__listunion(c1, csub)

                if t == self.PASS and t2 == self.FAIL:
                    # Found
                    progress = 1
                    next_c1 = csub
                    next_n = 2
                    cbar_offset = 0

                    if self.debug_dd:
                        print("dd: increase c1 to", len(next_c1), "deltas:",)
                        print()
                        self.pretty(next_c1)
                    break

                if t == self.FAIL and t1 == self.PASS:
                    # Increase
                    progress = 1
                    next_c2 = csub
                    next_n = max(next_n - 1, 2)
                    cbar_offset = i

                    if self.debug_dd:
                        print("dd: reduce c2 to", len(next_c2), "deltas:",)
                        print()
                        self.pretty(next_c2)
                    break

            if progress:
                self.report_progress(self.__listminus(next_c2, next_c1), "dd")
            else:
                if n >= len(c):
                    # No further minimizing
                    print("dd: done")
                    return (c, c1, c2)

                next_n = min(len(c), n * 2)
                print("dd: increase granularity to", next_n)
                cbar_offset = (cbar_offset * next_n) / n

            c1 = next_c1
            c2 = next_c2
            n = next_n
            run = run + 1

    def dd(self, c):
        return self.dddiff(c)  # Backwards compatibility
