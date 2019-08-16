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
])
rows, cols = test_picross.to_picross()

if __name__ == "__main__":
    s = find_sol(rows, cols)
    print(s)
    # rs = r.column_prune(c)
