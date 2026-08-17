import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

GEN_CACHE_SIZE = 64
DISPLAY_CACHE_SIZE = 200_000


@lru_cache(maxsize=GEN_CACHE_SIZE)
def gen_trees(n):
    """All canonical unlabeled rooted binary tree shapes with n leaves.
    Leaf = () ; internal node = (left, right) with left <= right (canonical/unordered)."""
    if n == 1:
        return ((),)
    results = set()
    for a in range(1, n // 2 + 1):
        b = n - a
        trees_a = gen_trees(a)
        trees_b = gen_trees(b)
        if a == b:
            for i, ta in enumerate(trees_a):
                for tb in trees_a[i:]:
                    results.add((ta, tb) if ta <= tb else (tb, ta))
        else:
            for ta in trees_a:
                for tb in trees_b:
                    results.add((ta, tb) if ta <= tb else (tb, ta))
    return tuple(sorted(results))

# Iterate candidate trees lazily to avoid building the full candidate
# tuple in memory for large `m`. `gen_trees_stream` yields the same
# canonical trees as `gen_trees(m)` but one by one.
def gen_trees_stream(n):
    """Yield canonical rooted tree shapes with n leaves lazily.
    Mirrors `gen_trees(n)` ordering without materializing the full
    result set in memory.
    """
    if n == 1:
        yield ()
        return
    for a in range(1, n // 2 + 1):
        b = n - a
        trees_a = gen_trees(a)
        trees_b = gen_trees(b)
        if a == b:
            for i, ta in enumerate(trees_a):
                for tb in trees_a[i:]:
                    yield (ta, tb) if ta <= tb else (tb, ta)
        else:
            for ta in trees_a:
                for tb in trees_b:
                    yield (ta, tb) if ta <= tb else (tb, ta)

@lru_cache(maxsize=DISPLAY_CACHE_SIZE)
def displays(t1, t2):
    """True iff t1 is an induced binary subtree ('displayed by') t2."""
    if t1 == ():
        return True
    if t2 == ():
        return False
    a, b = t1
    l, r = t2
    opt1 = (displays(a, l) and displays(b, r)) or \
           (displays(a, r) and displays(b, l))
    opt2 = (not opt1) and (displays(t1, l) or displays(t1, r))
    return opt1 or opt2


def is_universal(U, n_leaf_trees):
    return all(displays(t, U) for t in n_leaf_trees)

_WORKER_TREES_N = None


def _init_worker(trees_n):
    global _WORKER_TREES_N
    _WORKER_TREES_N = trees_n


def _check_candidate(candidate):
    return candidate, is_universal(candidate, _WORKER_TREES_N)

def find_minimal_universal(n, m_min=None, m_max=None, verbose=False,
                           parallel=False, file=None, num_workers=None):
    """Brute-force search for the minimal n-universal rooted binary tree(s)."""
    if file:
        with open(file, "w", encoding="utf-8") as f:
            f.write("Results of rooted_tree.py.\n\n")
            f.close()

    trees_n = gen_trees(n)
    m = n if m_min is None else m_min
    if parallel:
        if num_workers is None:
            num_workers = max(1, os.cpu_count() or 1)
        if num_workers == 1:
            parallel = False

    while True:
        if m_max is not None and m > m_max:
            raise RuntimeError(f"No universal tree found with <= {m_max} leaves")

        t0 = time.time()
        found = []
        if parallel:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(trees_n,),
            ) as executor:
                chunksize = max(1, min(16, num_workers * 2)) # type: ignore
                for candidate, is_univ in executor.map(
                    _check_candidate,
                    gen_trees_stream(m),
                    chunksize=chunksize,
                ):
                    if is_univ:
                        found.append(candidate)
        else:
            for U in gen_trees_stream(m):
                if is_universal(U, trees_n):
                    found.append(U)
        if verbose:
            dt = time.time() - t0
            output(f"  m={m}: {len(found)} universal, time={dt:6.2f}s", file=file)
        if found:
            return m, found
        m += 1

def output(s, file=None):
    print(s)
    if file:
        with open(file, "a", encoding="utf-8") as f:
            f.write(s + "\n")
            f.close()

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

    B, total = find_minimal_universal(args.n, m_min=args.min, m_max=args.max, 
                                  verbose=args.console, file=args.file, num_workers=args.workers)

    if B is not None:
        s = "\n"
        s += f"Minimal {args.n}-embedding universal structure S_{args.n}:\n"
        s += f"  |S_{args.n}| = {total}\n"
        s += f"  S_{args.n} = {B}\n"
        s += f"  (<1 = order of first coordinate, <2 = order of second coordinate)"
        output(s, args.file)
    
if __name__ == "__main__":
    filename = "rooted_tree_results.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Results of rooted_tree.py.\n\n")
        f.close()
    m_min = None
    for n in range(1, 100): # absurdly high upper limit
        t0 = time.time()
        m, found = find_minimal_universal(
            n,
            m_min=m_min,
            verbose=True,
            parallel=True,
            num_workers=max(1, os.cpu_count() or 1),
        )
        m_min = m
        dt = time.time() - t0
        output(f"n={n:2d}  u(n)={m:3d}  #minimal={len(found):3d}  time={dt:6.2f}s", file=filename)
