from functools import reduce
from operator import mul

def check_full(dots, pat):
    parts = split(dots)
    return list(map(len, parts)) == pat

def split(dots):
    con = []
    part = []
    for x in dots:
        if x:
            part.append(x)
        else:
            if part:
                con.append(part)
                part = []
    if part:
        con.append(part)
    return con

def inc(lst):
    for i in range(len(lst)):
        if lst[i]:
            lst[i] = 0
        else:
            lst[i] = 1
            return

def brute_gen(size):
    v = [0] * size
    yield tuple(v)
    for r in range(2**size-1):
        inc(v)
        yield tuple(v)

from enum import Enum
class DotState(Enum):
    NO = 0
    YES = 1
    MAYBE = 2
    UNCHECKED = 3

    def __add__(self, state):
        if self == self.UNCHECKED:
            return state
        if self == self.MAYBE:
            return self
        if self == state:
            return self
        return self.MAYBE
    
    def __repr__(self):
        if self == self.YES:
            return "1"
        if self == self.NO:
            return "0"
        return "?"

    def is_certain(self):
        return self == self.YES or self == self.NO
       

class PicrossRow:
    def __init__(self, pat, size):
        self.pat = pat
        self.size = size
        self._vals = None
        self._sum_vals = None

    def __getitem__(self, i):
        return self.sum_vals[i]

    def get_values(self):
        if self._vals is None:
            self._vals = list(self.gen())
        return [
            list(map(lambda ds: ds.value, val))
            for val in self._vals
        ]

    @property
    def sum_vals(self):
        if self._vals is None:
            self._vals = list(self.gen())
        if self._sum_vals is None:
            self._sum_vals = self._dsum(self._vals)
        return self._sum_vals

    def _dsum(self, lst):
        if len(lst) == 0:
            return [DotState.UNCHECKED] * self.size
        elif len(lst) == 1:
            return lst[0]
        else:
            return self._dadd(lst[0], self._dsum(lst[1:]))

    def __repr__(self):
        return repr(self.sum_vals)

    def is_certain(self):
        return all(p.is_certain() for p in self.sum_vals)

    def set_certain(self, i, value):
        if self[i].is_certain():
            if value != self[i]:
                raise ValueError("Attempted to set an already certain value to something else")
            else:
                return 0
        new_vals = [
            v for v in self._vals if v[i] == value
        ]
        if new_vals:
            changes = len(self._vals) - len(new_vals)
            self._vals = new_vals
            self._sum_vals = None #self._dsum(self._vals)
            return changes
        else:
            raise ValueError("That value at that position is impossible")

    def __setitem__(self, i, value):
        self._set(i, value)

    @staticmethod
    def _dadd(a, b):
        if len(a) != len(b):
            raise ValueError("Must be same length")
        else:
            return [a[i] + b[i] for i in range(len(a))]

    def _pos(self, pat, free):
        if len(pat) == 0:
            yield [DotState.NO] * free
            return

        p0 = [DotState.YES] * pat[0]
        if len(pat) == 1:
            if free:
                for n in range(free+1):
                    yield [DotState.NO] * n + p0 + [DotState.NO] * (free-n)
            else:
                yield p0
        else:
            if free:
                for n in range(free + 1):
                    for tail in self._pos(pat[1:], free - n):
                        yield [DotState.NO] * n + p0 + [DotState.NO] + tail
            else:
                for tail in self._pos(pat[1:], free):
                    yield p0 + [DotState.NO] + tail

    def gen(self):
        free = self.size - sum(self.pat) - (len(self.pat)-1)
        yield from self._pos(self.pat, free)


class Picross:
    def __init__(self, row_patterns, column_patterns):
        n_cols = len(column_patterns)
        n_rows = len(row_patterns)

        self.rows = [PicrossRow(row, n_cols) for row in row_patterns]
        self.cols = [PicrossRow(col, n_rows) for col in column_patterns]

    def row_prune(self):
        changes = 0
        for i, c in enumerate(self.cols):
            for j, r in enumerate(self.rows):
                if c[j].is_certain():
                    changes += r.set_certain(i, c[j])
        return changes

    def col_prune(self):
        changes = 0
        for j, r in enumerate(self.rows):
            for i, c in enumerate(self.cols):
                if r[i].is_certain():
                    changes += c.set_certain(j, r[i])
        return changes

    def certain(self):
        return all(row.is_certain() for row in self.rows) or all(col.is_certain() for col in self.cols)

    # def check_solution(self):
    #     sol_T = sol.transpose()
    #     return all(check_full(sol[x], rows[x]) for x in range(len(cols))) and \
    #             all(check_full(sol_T[x], cols[x]) for x in range(len(rows)))

    def solve(self, max_tries=0):
        tries = 0
        while True:
            tries += 1
            if max_tries and tries > max_tries:
                break
            changes = self.row_prune()
            changes += self.col_prune()
            if not changes:
                break
        return (tries, self.certain())


class Matrix:
    def __init__(self, rows=None, size=0):
        if size:
            self.m = [[0] * size for n in range(size)]
        elif rows:
            self.m = rows[:]
        else:
            raise ValueError("Must specify either size or rows")

    def __getitem__(self, key):
        if type(key) == int:
            return self.m[key]
        else:
            return self.m[key[0]][key[1]]

    def __setitem__(self, key, val):
        if type(key) == int:
            m[key] = val
        else:
            m[key[0]][key[1]] = val

    def __add__(self, b):
        return Matrix([
            [self.m[i][j] + b.m[i][j] for i in range(self.w)]
            for j in range(self.h)
        ])

    def __str__(self):
        return "\n".join(map(str, self.m))

    def __eq__(self, b):
        try:
            return all(
                all(
                    self.m[i][j] == b.m[i][j] for j in range(self.w)
                ) for i in range(self.h)
            )
        except AttributeError:
            return False
    
    def to_picross(self):
        return [list(map(len, split(row))) for row in self.m], \
                [list(map(len, split(col))) for col in self.transpose().m]

    @property
    def w(self):
        return len(self.m[0])
    
    @property
    def h(self):
        return len(self.m)

    def transpose(self):
        return Matrix([
            [self.m[j][i] for j in range(len(self.m))]
            for i in range(len(self.m[0]))
        ])

def common(lsts):
    return (int(all(lst[n] for lst in lsts)) for n in range(len(lsts[0])))

class SuperMatrix:
    def __init__(self, basis):
        # a vector where the first element is all possible first rows, etc
        # shape is: [ [ (a,b,c,...),(...), ] ]
        # list of "row possibilities" for each row
        # row possibility: a list of possible rows [(...), (...), ...]
        self.m = []
        try:
            self.width = len(basis[0][0])
        except:
            self.null = True
            return
        for row_basis in basis:
            self.m.append([tuple(row) for row in row_basis])
            if len(self.m[-1]) == 0:
                self.null = True
                return
            if any(len(row) != self.width for row in self.m[-1]):
                raise ValueError("Not a matrix basis! (bad length of row)")
        self.height = len(self.m)
        self.null = False
    
    @property
    def size(self):
        if self.null:
            return 0
        else:
            return reduce(mul, map(len, self.m), 1)

    def get_all(self, transposed=False):
        def _get(upper, lower):
            if len(lower) == 0:
                if transposed:
                    yield Matrix(upper).transpose()
                else:
                    yield Matrix(upper)
            else:
                for top in lower[0]:
                    yield from _get(upper + [top], lower[1:])
        yield from _get([], self.m)

    def transpose(self):
        yield from self.get_all(True)

    def single_column_prune(self, column, at):
        return SuperMatrix([
            [irow for irow in row if irow[at] == column[i]]
            for i, row in enumerate(self.m)
        ])

    def column_prune(self, col_sm):
        return SuperMatrix.lstintersection([
            SuperMatrix.lstunion([
                self.single_column_prune(ch, i) for ch in col_sm.m[i]
            ])
            for i in range(len(self.m[0][0]))
        ])

    def union(self, other):
        if other.size == 0:
            return SuperMatrix(self.m)
        if self.size == 0:
            return SuperMatrix(other.m)

        new_m = []
        for j in range(len(self.m)):
            new_m.append(set())
            for p in self.m[j]:
                new_m[j].add(tuple(p))
            for p in other.m[j]:
                new_m[j].add(tuple(p))
            new_m[j] = list(new_m[j])
        return SuperMatrix(new_m)

    def funion(self, lst):
        if len(lst) == 0:
            return self
        elif len(lst) == 1:
            return self.union(lst[0])
        else:
            return self.union(lst[0].funion(lst[1:]))
    
    @staticmethod
    def lstunion(lst):
        return lst[0].funion(lst[1:])

    @staticmethod
    def lstintersection(lst):
        return lst[0].fintersection(lst[1:])

    def fintersection(self, lst):
        if len(lst) == 0:
            return self
        elif len(lst) == 1:
            return self.intersection(lst[0])
        else:
            return self.intersection(lst[0].fintersection(lst[1:]))

    def intersection(self, other):
        if other.size == 0:
            return SuperMatrix([])
        new_m = []
        for j in range(len(self.m)):
            new_m.append([])
            for p in self.m[j]:
                if p in other.m[j]:
                    new_m[j].append(p) 
        return SuperMatrix(new_m)


# rows = [[1,1], [2,1], [1], [2,1], [1,2]]
# cols = [[1,1], [1,1], [1,1], [1,1], [4]]

# solution = Matrix([
#     [1, 0, 0, 1, 0],
#     [0, 1, 1, 0, 1],
#     [0, 0, 0, 0, 1],
#     [0, 1, 1, 0, 1],
#     [1, 0, 0, 1, 1]
# ])

def check_sol(sol, rows, cols):
    sol_T = sol.transpose()
    return all(check_full(sol[x], rows[x]) for x in range(len(cols))) and \
            all(check_full(sol_T[x], cols[x]) for x in range(len(rows)))


def find_sol(rows, cols):
    print("Finding solution to the picross with:")
    print(f"ROW patterns: {rows}")
    print(f"COL patterns: {cols}")
    print(f"Possible picrosses of this size ({len(rows)}x{len(cols)}) = {2**(len(rows)*len(cols))}")
    row_can = [[] for i in range(len(rows))]
    col_can = [[] for i in range(len(cols))]
    for b in brute_gen(len(rows)): # square for now.
        for i, r in enumerate(rows):
            if check_full(b, r):
                row_can[i].append(b)
        for i, c in enumerate(cols):
            if check_full(b, c):
                col_can[i].append(b)
    row_sm, col_sm = SuperMatrix(row_can), SuperMatrix(col_can)
    print(f"All possible picrosses that match just the ROWS: {row_sm.size} posibilities")
    print(f"All possible that match just the COLS: {col_sm.size} posibilities")
    print(f"Which gives a worst case # of comparisons: {row_sm.size * col_sm.size}")
    pruned = row_sm.column_prune(col_sm)
    print(f"But by selecting only possibilities that exist in both, we pruned that down to {pruned.size} !")
    comps = 0
    for p in pruned.get_all():
        comps += 1
        if check_sol(p, rows, cols):
            print(f"And, after {comps} checks we found a solution!")
            return p
    print(f"But, after trying all {comps} candidates we found nothing! :(")
    #return find_first(col_sm.get_all(True), row_sm.get_all())

def find_first(smA, smB):
    comps = 0
    cpy = list(smB)
    len_smb = len(cpy)
    for a in smA:
        comps += len_smb
        if a in cpy:
            print(f"found this after {comps} comps")
            return a
    print(f"found nothing after {comps} comps")


def square_common(rows, cols):
    cols_com = Matrix([list(common(c)) for c in cols])
    rows_com = Matrix([list(common(r)) for r in rows])
    # add transpose

    return rows_com + cols_com.transpose()

test_picross = Matrix([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
])
rows, cols = test_picross.to_picross()
p = Picross(rows, cols)

# if __name__ == "__main__":
#     s = find_sol(rows, cols)
#     print(s)
    # rs = r.column_prune(c)
