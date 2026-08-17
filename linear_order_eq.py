"""
Search for minimal n-embedding universal structures for the Fraisse limit
of the class K = E * LO  (equivalence relation E freely/independently
combined with a linear order <, i.e. the "free superposition" of the
class of equivalence relations and the class of linear orders).

--------------------------------------------------------------------------
Formalisation
--------------------------------------------------------------------------
Since a finite linear order has a UNIQUE order-automorphism (the
identity), the isomorphism type of a finite structure (A, E, <) in K
with |A| = m is completely determined by which pairs of the m
<-ordered points lie in the same E-class -- i.e. by a SET PARTITION of
{1,...,m}. So:

    { iso. types of size-m structures in K }  <-->  { set partitions of [m] }

and there are exactly Bell(m) of them.

We represent a set partition of [m] canonically as a "restricted growth
string" (RGS) b = (b_0,...,b_{m-1}) with b_0 = 0 and
b_i <= max(b_0,...,b_{i-1}) + 1. This is a bijection between {0,..,m-1}
-> partitions of [m], so enumerating RGS enumerates each partition
exactly once (no isomorphic duplicates), keeping the search space as
small as possible (size Bell(m)).

A host structure U of size m (given by RGS b) EMBEDS a target pattern of
size n (given by RGS t) iff there is a strictly increasing sequence of
indices i_1 < ... < i_n in [m] such that

        b[i_p] == b[i_q]   <=>   t[p] == t[q]      for all p,q.

U is "n-embedding universal" for K iff it embeds every size-n pattern
(equivalently, by a downward-closure argument, every pattern of size
<= n too). We search increasing m and, for each m, all Bell(m)
partitions of [m], looking for one universal for all Bell(n) targets.
The first m that succeeds is the minimal size; we report a witness.

--------------------------------------------------------------------------
Complexity warning
--------------------------------------------------------------------------
This is brute force: for a given m it tries all Bell(m) partitions, and
Bell(m) grows very fast (1,1,2,5,15,52,203,877,4140,21147,115975,...).
That is fine for n = 2, 3 (and usually 4), but do not expect this to
scale to large n without replacing the outer search by a SAT/ILP
encoding (each target's embedding becomes an existential constraint over
auxiliary position variables). The embedding check itself (embeds()) is
already the "hard part" reused by such an encoding.

Confirmed results:
n = 1 : minimal N = 1
n = 2 : minimal N = 3
n = 3 : minimal N = 5
n = 4 : minimal N = 8
n = 5 : minimal N = 11
n = 6 : minimal N = 15
n = 7 : minimal 17 <= N 
"""

from itertools import count
import argparse
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


_WORKER_TARGETS = None


def gen_rgs(length):
    """Yield every restricted growth string of the given length, i.e.
    every set partition of {0,...,length-1}, exactly once."""
    if length == 0:
        yield ()
        return

    def rec(seq, max_val):
        if len(seq) == length:
            yield tuple(seq)
            return
        for v in range(max_val + 2):
            yield from rec(seq + [v], max(max_val, v))

    yield from rec([], -1)


def _is_candidate_partition(host, n):
    block_sizes = Counter(host)
    return len(block_sizes) == n and sum(size == n for size in block_sizes.values()) == 1


def _generate_candidate_partitions_from_prefix_chunk(args):
    length, n, prefixes = args
    results = []

    for prefix in prefixes:
        prefix = tuple(prefix)

        def rec(seq, max_val):
            if len(seq) == length:
                if _is_candidate_partition(tuple(seq), n):
                    results.append(tuple(seq))
                return
            for v in range(max_val + 2):
                rec(seq + [v], max(max_val, v))

        rec(list(prefix), max(prefix) if prefix else -1)

    return results


def iter_candidate_partitions(length, n, workers=None):
    """Stream candidate RGS-partitions in a memory-friendly way.

    Each host is checked immediately, so we never need to materialize the full
    search space of candidate partitions at once.
    """
    if length < n:
        return

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
    if workers <= 1:
        for host in gen_rgs(length):
            if _is_candidate_partition(host, n):
                yield host
        return

    prefixes = []

    if length <= 1:
        prefixes = [()]
    else:
        def bell_number(k):
            if k <= 0:
                return 1
            row = [1]
            for _ in range(k):
                new_row = [row[-1]]
                for x in row:
                    new_row.append(new_row[-1] + x)
                row = new_row
            return row[0]

        target_tasks = max(8, workers * 8)
        split_depth = 1
        while split_depth < length and bell_number(split_depth) < target_tasks:
            split_depth += 1
        split_depth = min(split_depth, length - 1)

        def add_prefixes(seq, max_val):
            if len(seq) == split_depth:
                prefixes.append(tuple(seq))
                return
            for v in range(max_val + 2):
                add_prefixes(seq + [v], max(max_val, v))

        add_prefixes([], -1)

    batch_size = max(1, len(prefixes) // max(1, workers * 8))
    if batch_size == 0:
        batch_size = 1

    tasks = [prefixes[i:i + batch_size] for i in range(0, len(prefixes), batch_size)]
    if not tasks:
        tasks = [()]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch in executor.map(
            _generate_candidate_partitions_from_prefix_chunk,
            [(length, n, group) for group in tasks],
            chunksize=1,
        ):
            for candidate in batch:
                yield candidate


def generate_candidate_partitions(length, n, workers=None):
    """Compatibility wrapper that materializes the candidate list when needed."""
    return list(iter_candidate_partitions(length, n, workers=workers))


def _init_worker(targets):
    global _WORKER_TARGETS
    _WORKER_TARGETS = tuple(targets)


def _check_candidate(host):
    return host, is_universal(host, _WORKER_TARGETS)


# def bell(n):
#     # simple Bell-triangle computation, only used for progress reporting
#     row = [1]
#     for _ in range(n):
#         new_row = [row[-1]]
#         for x in row:
#             new_row.append(new_row[-1] + x)
#         row = new_row
#     return row[0]


def embeds(host, target):
    """Does `host` (RGS of length m) contain `target` (RGS of length n)
    as an induced sub-pattern (order-preserving, exact E-pattern)?"""
    m, n = len(host), len(target)
    if n > m:
        return False

    def rec(host_pos, t_pos, mapping, used_host_blocks):
        if t_pos == n:
            return True
        remaining = n - t_pos
        for hp in range(host_pos, m - remaining + 1):
            hb = host[hp]
            tb = target[t_pos]
            if tb in mapping:
                if mapping[tb] != hb:
                    continue
            else:
                if hb in used_host_blocks:
                    continue  # would collapse two distinct target blocks
                mapping[tb] = hb
                used_host_blocks.add(hb)
                if rec(hp + 1, t_pos + 1, mapping, used_host_blocks):
                    return True
                used_host_blocks.discard(hb)
                del mapping[tb]
                continue
            if rec(hp + 1, t_pos + 1, mapping, used_host_blocks):
                return True
        return False

    return rec(0, 0, {}, set())


def is_universal(host, targets):
    for target in targets:
        if not embeds(host, target):
            return False
    return True


def rgs_to_partition(rgs):
    """Human-readable partition, 1-indexed positions in their <-order."""
    blocks = {}
    for idx, b in enumerate(rgs, start=1):
        blocks.setdefault(b, []).append(idx)
    return list(blocks.values())


def find_minimal_universal(n, min_m=None, max_m=None, verbose=True, file="results.txt", workers=None):
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    # The target set is the main memory consumer for larger n. Parallelism needs
    # to duplicate it in every worker, so we keep the single-process path once the
    # target enumeration becomes too large to hold comfortably in RAM.
    if n >= 10:
        workers = 1

    if workers > 1:
        targets = tuple(gen_rgs(n))
    else:
        targets = gen_rgs(n)

    if verbose:
        output(f"n = {n}:", file=file)

    if min_m is None:
        min_m = n

    for m in count(min_m):
        if max_m is not None and m > max_m:
            if verbose:
                output(f"No universal structure found up to m = {max_m}.", file=file)
            return None

        t0 = time.time()
        checked = 0
        candidate_count = 0

        if workers > 1:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(targets,),
            ) as executor:
                for host, universal in executor.map(_check_candidate, iter_candidate_partitions(m, n, workers=workers), chunksize=1):
                    candidate_count += 1
                    checked += 1
                    if verbose and checked % 100000 == 0:
                        elapsed = time.time() - t0
                        print(f"m = {m}: still running; checked {checked:,} candidates in {elapsed:.2f}s")
                    if universal:
                        dt = time.time() - t0
                        if verbose:
                            output(f"m = {m}: FOUND after checking {checked} "
                                  f"({dt:.2f}s)", file=file)
                        return m, host
        else:
            for host in iter_candidate_partitions(m, n, workers=1):
                candidate_count += 1
                checked += 1
                if verbose and checked % 10000 == 0:
                    elapsed = time.time() - t0
                    print(f"m = {m}: still running; checked {checked:,} candidates in {elapsed:.2f}s")
                if is_universal(host, targets):
                    if verbose:
                        dt = time.time() - t0
                        output(f"m = {m}: FOUND after checking {checked} "
                              f"({dt:.2f}s)", file=file)
                    return m, host

        if verbose:
            dt = time.time() - t0
            output(f"m = {m}: none of the {candidate_count:,} candidate partitions works "
                  f"({dt:.2f}s), trying m = {m + 1}", file=file)

def output(s, file=None):
    print(s)
    if file is not None and not isinstance(file, bool):
        with open(file, "a", encoding="utf-8") as f:
            f.write(s + "\n")
            f.close()


def cli():
    ap = argparse.ArgumentParser(
        description="Search minimal n-embedding universal structures for "
                     "the Fraisse limit of E*LO.")
    ap.add_argument("n", type=int, help="size of patterns to be universal for")
    ap.add_argument("--max-m", type=int, default=None,
                     help="give up if no witness is found by this size")
    ap.add_argument("--file", default="results.txt",
                    help="append results to the given file (default: results.txt)")
    args = ap.parse_args()

    result = find_minimal_universal(args.n, max_m=args.max_m, file=args.file)
    if result is None:
        return
    m, host = result
    s = f"\nMinimal n-embedding universal structure for n = {args.n}:\n"
    s += f"  size m = {m}\n"
    s += f"  RGS    = {host}\n"
    s += f"  partition (blocks, positions in <-order) = {rgs_to_partition(host)}"

    output(s, file=args.file)


if __name__ == "__main__":
    filename = "linear_order_eq-results.txt"
    with open(filename, "w", encoding="utf-8") as f:
            f.write("Results of linear_order_eq.py.\n\n")
            f.close()
    
    min_m = 17
    for n in range(7,20):
        result = find_minimal_universal(n, min_m=min_m, file=filename)
        if result is None:
            output(f"ERROR! n = {n}. No universal structures found!")
            exit()
        m, host = result
        s = f"\nMinimal n-embedding universal structure for n = {n}:\n"
        s += f"  size m = {m}\n"
        s += f"  RGS    = {host}\n"
        s += f"  partition (blocks, positions in <-order) = {rgs_to_partition(host)}\n"
    
        output(s, file=filename)

        min_m = m
