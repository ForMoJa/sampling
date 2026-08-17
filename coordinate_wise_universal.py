#!/usr/bin/env python3
"""
Known / verified results (computed with this program)
    n = 1 : minimal N = 1
    n = 2 : minimal N = 4
    n = 3 : minimal N = 8
    n = 4 : minimal N = 14
    n = 5 : 14 <= N <= 25
    for general n: N <= n^2
"""

import multiprocessing
import itertools
import time

filename = "coordinate_wise_results.txt"


def standardize(pts):
    """Canonical form of a finite set of pairwise-distinct (x,y) integer
    pairs: replace x by its dense rank among the used x-values, and y by
    its dense rank among the used y-values."""
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))
    xr = {v: i for i, v in enumerate(xs)}
    yr = {v: i for i, v in enumerate(ys)}
    return frozenset((xr[x], yr[y]) for x, y in pts)


def enumerate_patterns(m):
    """All canonical patterns of exactly m points that embed into A."""
    seen = set()
    out = []
    for p in range(1, m + 1):
        for q in range(1, m + 1):
            if p * q < m:
                continue
            cells = [(x, y) for x in range(p) for y in range(q)]
            for combo in itertools.combinations(cells, m):
                xs = set(c[0] for c in combo)
                ys = set(c[1] for c in combo)
                if len(xs) == p and len(ys) == q:
                    fs = frozenset(combo)
                    if fs not in seen:
                        seen.add(fs)
                        out.append(fs)
    return out


def contains_pattern(T, pattern):
    """Check if T contains a subset that is order-isomorphic to 'pattern'."""
    m = len(pattern)
    if m > len(T):
        return False
    for combo in itertools.combinations(T, m):
        if standardize(combo) == pattern:
            return True
    return False


def T_contains_all(T, patterns):
    """Check if T contains all patterns in the list 'patterns'."""
    return all(contains_pattern(T, p) for p in patterns)


def brute_minimal(n, max_N=None, verbose=True, file=filename):
    """Brute-force search for a minimal n-embedding universal structure."""
    patterns = enumerate_patterns(n)
    if not max_N:
        max_N = n ** 2
    for N in range(1, max_N + 1):
        cells = [(x, y) for x in range(N) for y in range(N)]
        t0 = time.time()
        for combo in itertools.combinations(cells, N):
            if T_contains_all(combo, patterns):
                if verbose:
                    output(f"n={n}: minimal N = {N}  ({time.time()-t0:.2f}s)", file)
                return N, combo
        if verbose:
            output(f"n={n}: N={N} infeasible  ({time.time()-t0:.2f}s)", file)
    if verbose:
        output(f"n={n}: no witness found up to N={max_N}", file)
    return None, None


def _build_and_solve(N, patterns, time_limit, workers):
    from ortools.sat.python import cp_model

    for pat in patterns:
        if len(pat) > N:
            return None  # a pattern literally can't fit -> N infeasible

    model = cp_model.CpModel()
    x = [model.new_int_var(0, N - 1, f'x{i}') for i in range(N)]
    y = [model.new_int_var(0, N - 1, f'y{i}') for i in range(N)]
    combo = [model.new_int_var(0, N * N - 1, f'c{i}') for i in range(N)]
    for i in range(N):
        model.add(combo[i] == x[i] * N + y[i])
    model.add_all_different(combo)
    # symmetry breaking: kill the N! relabelings of "which slot is which point"
    for i in range(N - 1):
        model.add(combo[i] < combo[i + 1])

    for pidx, pat in enumerate(patterns):
        pts = sorted(pat)
        m = len(pts)
        assign = [[model.new_bool_var(f'a_{pidx}_{k}_{i}') for i in range(N)]
                  for k in range(m)]
        for k in range(m):
            model.add(sum(assign[k][i] for i in range(N)) == 1)
        for i in range(N):
            model.add(sum(assign[k][i] for k in range(m)) <= 1)
        for k in range(m):
            for l in range(k + 1, m):
                ak, bk = pts[k]
                al, bl = pts[l]
                xrel = 'lt' if ak < al else ('gt' if ak > al else 'eq')
                yrel = 'lt' if bk < bl else ('gt' if bk > bl else 'eq')
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        lits = [assign[k][i], assign[l][j]]
                        if xrel == 'lt':
                            model.add(x[i] < x[j]).OnlyEnforceIf(lits)
                        elif xrel == 'gt':
                            model.add(x[i] > x[j]).OnlyEnforceIf(lits)
                        else:
                            model.add(x[i] == x[j]).OnlyEnforceIf(lits)
                        if yrel == 'lt':
                            model.add(y[i] < y[j]).OnlyEnforceIf(lits)
                        elif yrel == 'gt':
                            model.add(y[i] > y[j]).OnlyEnforceIf(lits)
                        else:
                            model.add(y[i] == y[j]).OnlyEnforceIf(lits)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = max(1, workers)
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [(solver.Value(x[i]), solver.Value(y[i])) for i in range(N)]
    elif status == cp_model.INFEASIBLE:
        return None
    else:
        return 'UNKNOWN'


def cpsat_minimal(n, time_limit, max_N=None, min_N=0, workers=8, verbose=True, file=filename):
    """Compute a minimal n-embedding universal structure using CP-SAT."""
    if not max_N:
        max_N = n ** 2
    for N in range(min_N, max_N + 1):
        t0 = time.time()
        res = _build_and_solve(N, enumerate_patterns(n), time_limit=time_limit, workers=workers)
        dt = time.time() - t0
        if res == 'UNKNOWN':
            if verbose:
                output(f"n={n},N={N}: solver undecided within {time_limit}s "
                      f"(time limit reached)", file)
            return None, None
        if res is None:
            if verbose:
                output(f"n={n},N={N}: infeasible  ({dt:.2f}s)", file)
            continue
        if verbose:
            output(f"n={n}: minimal N = {N}, ({dt:.2f}s)", file)
        return N, sorted(res)
    if verbose:
        output(f"n={n}: no witness found up to N={max_N}; maximal searched size reached", file)
    return None, None


def cli():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=3,
                    help='compute the minimal n-embedding universal structure')
    ap.add_argument('--max-N', type=int, default=20,
                    help='upper bound on candidate size to try (default 20)')
    ap.add_argument('--time-limit', type=float, default=60,
                    help='CP-SAT time limit per candidate size in seconds')
    ap.add_argument('--brute-force', action='store_true',
                    help='use exact exhaustive search instead of CP-SAT')
    ap.add_argument('--workers', type=int, default=None,
                    help='number of OR-Tools search workers (default 1)')
    ap.add_argument('-f', '--file', type=str, default=None,
                    help='file for output')
    args = ap.parse_args()

    workers = max(1, multiprocessing.cpu_count() - 1) or args.workers
    if args.brute_force:
        N, T = brute_minimal(args.n, max_N=args.max_N, file=args.file)
    else:
        N, T = cpsat_minimal(args.n, args.time_limit,
                             max_N=args.max_N, workers=workers, file=args.file)

    if T is not None:
        s = "\n"
        s += f"Minimal {args.n}-embedding universal structure S_{args.n}:\n"
        s += f"  |S_{args.n}| = {N}\n"
        s += f"  S_{args.n} = {{ {', '.join(str(p) for p in sorted(T))} }}\n"
        s += f"  (<1 = order of first coordinate, <2 = order of second coordinate)"
        output(s, args.file)

def output(s, file):
    print(s)
    with open(file, "a", encoding="utf-8") as f:
        f.write(s + "\n")
        f.close()


if __name__ == '__main__':
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Results of rooted_tree.py.\n\n")
        f.close()

    hours = 20
    min_N = 14
    for n in range(5,20):
        N, T = cpsat_minimal(n, hours*60*60, min_N=min_N) # type: ignore
        output(f"n={n}: minimal N = {N}, T = {T}", filename)
        min_N = N