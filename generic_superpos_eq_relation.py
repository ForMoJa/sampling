"""
Seraches for minimal embedding universal strucutres for E * E (the generic superposition
of the homogeneouos equivalence relation with itself).
The program saves finite substructures of E * E as matrices of non-negative integers
Confirmed results:
n = 1 : minimal N = 1
n = 2 : minimal N = 4
n = 3 : minimal N = 8
n = 4 : minimal N = 14
n = 5 : minimal N = 20
n = 6 : minimal N = 29
n = 7 : minimal N = 37
n = 8 : minimal 43 <= N

Conjectured results: OEIS A397992
"""

from __future__ import annotations
import concurrent.futures
import multiprocessing
from multiprocessing.pool import Pool
import time
from collections import deque

"""
Check whether canonical matrix A can be embedded into canonical matrix B.

embed(A, B) returns True iff there exist a row permutation sigma and a column
permutation tau such that A[sigma[i]][tau[j]] <= B[i][j] for all i, j.

Key theorem (exploiting canonical form)
----------------------------------------
When both A and B are in canonical form (rows sorted descending by row_key,
columns sorted descending by column sum), if any valid (sigma, tau) exists then
sigma = identity also works. Therefore we only need to search for a column
permutation tau — a pure bipartite matching problem.

Proof sketch:
  Given a valid (sigma, tau), summing A[sigma[i]][tau[j]] <= B[i][j] over j gives
  rowsum(A[sigma[i]]) <= rowsum(B[i]) for each i.  Since A's rows are sorted
  descending by row sum and B's rows are sorted descending by row sum, this forces
  rowsum(A[i]) <= rowsum(B[i]) for all i (the canonical ordering already aligns
  rows by decreasing weight, so re-ordering can only hurt).  An analogous argument
  applies to columns.  The column permutation tau then witnesses that A (with the
  identity row order) fits under B after re-ordering columns appropriately.

Algorithm
---------
1. O(n)     - row-sum dominance check: rowsum(A[i]) <= rowsum(B[i]) for all i.
2. O(n)     - col-sum dominance check: colsum(A[:,j]) <= colsum(B[:,j]) for all j.
3. O(n^3)   - build bipartite graph: A-column k can be matched to B-column j iff
             A[i][k] <= B[i][j] for every row i.
4. O(n^2.5) - Hopcroft-Karp maximum bipartite matching on the n x n graph.
             Return True iff a perfect matching exists.
"""

def _hopcroft_karp(graph: list[list[int]], n: int) -> list[int]:
    """
    Maximum bipartite matching via Hopcroft-Karp.

    Parameters
    ----------
    graph : adjacency list; graph[u] lists all v that u can be matched to.
            Both u and v are integers in 0..n-1.
    n     : number of nodes on each side.

    Returns
    -------
    match_u : list of length n; match_u[u] = v if u is matched to v, else -1.
    """
    INF = float("inf")
    match_u = [-1] * n   # match for left nodes
    match_v = [-1] * n   # match for right nodes

    def bfs() -> dict[int, float]:
        dist: dict[int, float] = {}
        queue: deque[int] = deque()
        for u in range(n):
            if match_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                w = match_v[v]
                if w == -1:
                    found = True
                elif dist.get(w, INF) == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return dist if found else {}

    def dfs(u: int, dist: dict[int, float]) -> bool:
        for v in graph[u]:
            w = match_v[v]
            if w == -1 or (dist.get(w, INF) == dist[u] + 1 and dfs(w, dist)):
                match_u[u] = v
                match_v[v] = u
                return True
        dist[u] = INF
        return False

    while True:
        dist = bfs()
        if not dist:
            break
        for u in range(n):
            if match_u[u] == -1:
                dfs(u, dist)

    return match_u

def embed(A,B) -> bool:
    """
    Return True if some row-permutation sigma and column-permutation tau satisfy
    A[sigma[i]][tau[j]] <= B[i][j]  for all i, j.

    Both A and B must be in canonical form (as produced by generate_matrices):
      - rows sorted descending by (row sum, reverse-lexicographic order)
      - columns sorted descending by column sum

    Parameters
    ----------
    A, B : canonical n x n matrices represented as tuples of tuples.
    """
    n = len(A)

    # 1. Row-sum dominance (necessary condition; O(n))
    for i in range(n):
        if sum(A[i]) > sum(B[i]):
            return False

    # 2. Column-sum dominance (necessary condition; O(n^2))
    for j in range(n):
        col_sum_A = sum(A[i][j] for i in range(n))
        col_sum_B = sum(B[i][j] for i in range(n))
        if col_sum_A > col_sum_B:
            return False

    # 3. Build bipartite compatibility graph (O(n^3))
    #    Left nodes  = columns of A (index k)
    #    Right nodes = columns of B (index j)
    #    Edge (k, j) exists iff A[i][k] <= B[i][j] for all rows i
    graph: list[list[int]] = [[] for _ in range(n)]
    for k in range(n):
        for j in range(n):
            if all(A[i][k] <= B[i][j] for i in range(n)):
                graph[k].append(j)

    # 4. Perfect matching via Hopcroft-Karp (O(n^2.5))
    match_u = _hopcroft_karp(graph, n)
    return all(m != -1 for m in match_u)
    
def generate_matrices(n, total=None):
    """Generate all n x n non-negative-integer matrices whose entries sum to total."""
    if total is None:
        total = n
    num_entries = n * n

    def compositions(current_total, parts):
        if parts == 1:
            yield (current_total,)
            return
        for first in range(current_total + 1):
            for rest in compositions(current_total - first, parts - 1):
                yield (first,) + rest

    for flat in compositions(total, num_entries):
        yield [list(flat[i * n:(i + 1) * n]) for i in range(n)]


# def row_key(row):
#     """Sort key for descending row order: larger sum first,
#     ties broken by 'reverse lexicographic' order (compare reversed rows)."""
#     return (sum(row), tuple(reversed(row)))


def is_sorted_matrix(matrix):
    """Check matrix ordering:

    1. Column sums are non-increasing left to right.
    2. Rows are non-increasing top to bottom by row sum.
    3. Rows with equal sum are ordered by reverse lexicographic order.
    """
    n = len(matrix)
    if n == 0:
        return True

    row_keys = [(sum(row), tuple(reversed(row))) for row in matrix]
    if any(row_keys[i] < row_keys[i + 1] for i in range(n - 1)):
        return False

    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    if any(col_sums[j] < col_sums[j + 1] for j in range(n - 1)):
        return False

    return True


def _generate_rows_with_sum(n, target_sum, prev_row):
    prev_sum = sum(prev_row) if prev_row is not None else 0

    def helper(pos, remaining, prefix):
        if prev_row is not None:
            if sum(prefix) > prev_sum: # type: ignore
                return
            if sum(prefix) == prev_sum:
                for i in range(len(prefix)):
                    if prefix[i] > prev_row[i]:
                        break
                    if prefix[i] < prev_row[i]:
                        return

        if pos == n:
            if remaining == 0:
                yield tuple(prefix)
            return
        # max_value = min(remaining, max)
        max_value = remaining
        if prev_row is not None:
            max_value = min(max_value, prev_sum) 
        for value in range(max_value, -1, -1):
            yield from helper(pos + 1, remaining - value, prefix + [value])

    yield from helper(0, target_sum, [])


def _columns_sorted(matrix):
    if not matrix:
        return True
    n = len(matrix)
    col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
    return all(col_sums[j] >= col_sums[j + 1] for j in range(n - 1))


def _generate_sorted_matrices_for_prefix(args):
    n, total, first_row = args
    prev_row = tuple(first_row)
    matrix = [prev_row]
    remaining_sum = total - sum(first_row)
    rows_left = n - 1
    results = []

    def backtrack(prev_row, remaining_sum, rows_left):
        if rows_left == 0:
            if remaining_sum == 0 and _columns_sorted(matrix):
                results.append([list(row) for row in matrix])
            return

        max_sum = remaining_sum if prev_row is None else min(sum(prev_row), remaining_sum)
        for row_sum in range(max_sum, -1, -1):
            for row in _generate_rows_with_sum(n, row_sum, prev_row):
                matrix.append(row)
                backtrack(row, remaining_sum - row_sum, rows_left - 1)
                matrix.pop()

    backtrack(prev_row, remaining_sum, rows_left)
    return results


def generate_sorted_matrices(n, total=None, workers=None):
    """Generate canonical n x n non-negative-integer matrices summing to total.

    The generated matrices satisfy:
    1. Column sums are non-increasing left to right.
    2. Rows are non-increasing by row sum top to bottom.
    3. Rows with equal sum are reverse-lexicographically ordered.
    """
    if total is None:
        total = n
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    def sequential_generation():
        matrix = []

        def backtrack(prev_row, remaining_sum, rows_left):
            if rows_left == 0:
                if remaining_sum == 0 and _columns_sorted(matrix):
                    yield [list(row) for row in matrix]
                return

            max_sum = remaining_sum if prev_row is None else min(sum(prev_row), remaining_sum)
            for row_sum in range(max_sum, -1, -1):
                for row in _generate_rows_with_sum(n, row_sum, prev_row):
                    matrix.append(row)
                    yield from backtrack(row, remaining_sum - row_sum, rows_left - 1)
                    matrix.pop()

        yield from backtrack(None, total, n)

    if workers <= 1:
        yield from sequential_generation()
        return

    first_rows = []
    for first_sum in range(total, -1, -1):
        for first_row in _generate_rows_with_sum(n, first_sum, None):
            first_rows.append((n, total, first_row))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_generate_sorted_matrices_for_prefix, first_rows):
            for matrix in result:
                yield matrix


def _generate_candidate_matrices_for_second_row(args):
    n, total, first_row, second_row = args
    matrix = [first_row, second_row]
    remaining_sum = total - sum(first_row) - sum(second_row)
    rows_left = n - 2
    prev_row = second_row
    results = []

    def backtrack(prev_row, remaining_sum, rows_left):
        if rows_left == 0:
            if remaining_sum == 0 and _columns_sorted(matrix):
                results.append([list(row) for row in matrix])
            return

        max_sum = min(sum(prev_row), remaining_sum)
        for row_sum in range(max_sum, -1, -1):
            for row in _generate_rows_with_sum(n, row_sum, prev_row):
                # enforce that the first column equals the fixed first-column
                # values taken from the initial first_row tuple: each row k
                # must have row[0] == first_row[k]
                required_first = first_row[len(matrix)]
                if row[0] != required_first:
                    continue
                matrix.append(row)
                backtrack(row, remaining_sum - row_sum, rows_left - 1)
                matrix.pop()

    backtrack(prev_row, remaining_sum, rows_left)
    return results


def generate_candidate_matrices(n, total=None, workers=None):
    """Generate canonical candidate matrices summing to total with a fixed first row.

    The first row and column is fixed to [n // (i + 1) for i in range(n)].
    The remainder of each matrix is generated in parallel using workers.
    """
    if total is None:
        total = n
    first_row = tuple(n // (i + 1) for i in range(n))
    first_sum = sum(first_row)
    if first_sum > total:
        return
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    if n == 1:
        if total == first_sum:
            yield [list(first_row)]
        return

    remaining_total = total - first_sum

    second_row_candidates = []
    # We also require the first column to equal the same sequence as the
    # fixed first row. Therefore second-row candidates must have their
    # first element equal to the second entry of the fixed first-row,
    # and subsequent rows will be filtered during backtracking.
    max_second_sum = min(first_sum, remaining_total)
    for second_sum in range(max_second_sum, -1, -1):
        for second_row in _generate_rows_with_sum(n, second_sum, first_row):
            # second_row corresponds to matrix row index 1, so enforce
            # second_row[0] == first_row[1]
            if second_row[0] != first_row[1]:
                continue
            second_row_candidates.append(second_row)

    if workers <= 1:
        for second_row in second_row_candidates:
            for matrix in _generate_candidate_matrices_for_second_row((n, total, first_row, second_row)):
                yield matrix
        return

    tasks = [(n, total, first_row, second_row) for second_row in second_row_candidates]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_generate_candidate_matrices_for_second_row, tasks):
            for matrix in result:
                yield matrix


def is_universal_matrix(B, n, total=None, verbose=False):
    """Return True if B is n-embedding universal for homogeneous equivalence relation."""
    if total is None:
        total = n
    canonical_As = list(generate_sorted_matrices(n, total=total))
    if verbose:
        print(f"checking universal property for B with total={sum(sum(row) for row in B)}")
    return all(embed(A, B) for A in canonical_As)

def output(s, file):
    print(s)
    with open(file, "a", encoding="utf-8") as f:
        f.write(s + "\n")
        f.close()

def _init_worker(canonical_As):
    global _CANONICAL_AS
    _CANONICAL_AS = canonical_As

def _check_candidate(B):
    return B if all(embed(A, B) for A in _CANONICAL_AS) else None


def find_minimal_universal_matrix(n, min_total=None, max_total=None, verbose=False, file="results.txt",workers=None):
    """Find a minimal n-embedding universal matrix for given n.

    The function searches over canonical candidate matrices B whose total entry
    sum increases from n upward, returning the first B that embeds every
    canonical n x n matrix A with total sum n.
    """
    if max_total is None:
        max_total = n * n
    if min_total is None:
        min_total = n
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    canonical_As = list(generate_sorted_matrices(n, total=n))
    for total in range(min_total, max_total + 1):
        if verbose:
            output(f"n={n}, generating candidates with N={total}", file)
            t0 = time.time()
        candidates = list(generate_candidate_matrices(n, total=total))
        c_candidates = len(candidates)
        if not candidates:
            if verbose:
                output(f"n={n} found no candidates for N={total}", file)
            continue
        if verbose:
            dt = time.time() - t0
            output(f"n={n}, generated {c_candidates} (time {dt//60} min)", file)
        t0 = time.time()

        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(canonical_As,)
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(
                    _check_candidate,
                    candidates,
                    chunksize=1_000
                ),
                start=1
            ):
                if verbose and i % 10000 == 0:
                    print(f"Tested {i} candidates")

                if result is not None:
                    pool.terminate()
                    return result, total
        if verbose:
            dt = time.time() - t0
            output(f"no universal matrix found for total={total} (time {dt//60}  min)", file)
    return None, None

def cli():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n', type=int,
                    help='compute the minimal n-embedding universal structure')
    ap.add_argument('--min', type=int,
                    help='lower bound on cadidates')
    ap.add_argument('--max', type=int,
                    help='upper bound on candidate size to try')
    ap.add_argument('--workers', type=int,
                    help='number of OR-Tools search workers (default 1)')
    ap.add_argument('--console', action='store_true',
                        help='set whether results are printed in console (default yes)')
    ap.add_argument('--file', action='store_true',
                            help='set whether results are printed in console (default yes)')
    args = ap.parse_args()

    B, total = find_minimal_universal_matrix(args.n, min_total=args.min, max_total=args.max, 
                                  verbose=args.console, file=args.file, workers=args.workers)

    if B is not None:
        s = "\n"
        s += f"Minimal {args.n}-embedding universal structure S_{args.n}:\n"
        s += f"  |S_{args.n}| = {total}\n"
        s += f"  S_{args.n} = {B}\n"
        s += f"  (<1 = order of first coordinate, <2 = order of second coordinate)"
        output(s, args.file)

if __name__ == "__main__":
    filename = "generic_results.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Results of rooted_tree.py.\n\n")
        f.close()

    min_total = 43
    for n in range(8,20):
        t0 = time.time()
        B, min_total = find_minimal_universal_matrix(n, min_total=min_total, verbose=True) 
        dt = time.time() - t0
        output(f"  {B} in {dt/60} min", filename)
