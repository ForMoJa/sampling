from __future__ import annotations
import copy
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    """Simple rooted binary tree node.

    Attributes:
        label: optional label for the node (only used in printing)
        left: left child (or None)
        right: right child (or None)
        parent: parent node (or None)
    """
    label: Optional[int] = None # only used in printing
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    parent: Optional["TreeNode"] = None
    marked: bool = False


def find_marked_leaf(root: TreeNode) -> TreeNode:
    """Return the unique marked readleave node in a 2r2-tree."""
    if root is None:
        raise ValueError("tree must be a non-empty 2r2-tree")
    marked_leaf: Optional[TreeNode] = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.marked:
            if marked_leaf is not None:
                raise ValueError("tree contains more than one marked readleave")
            if node.left is not None or node.right is not None:
                raise ValueError("marked node is not a leaf")
            marked_leaf = node
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    if marked_leaf is None:
        raise ValueError("tree contains no marked readleave")
    return marked_leaf


def root(node: TreeNode) -> TreeNode:
    """Return the root of the tree that contains node."""
    while node.parent is not None:
        node = node.parent
    return node


def is_leaf(node: TreeNode) -> bool:
    """Return True if node is a leaf (has no children)."""
    return node.left is None and node.right is None


def depth(node: TreeNode) -> int:
    """Return depth of node (root has depth 0)."""
    d = 0
    while node.parent is not None:
        node = node.parent
        d += 1
    return d


def merge(T1: TreeNode, T2: Optional[TreeNode]) -> TreeNode:
    """Merge a 2r2-tree T1 with T2 by identifying T1's readleave with T2's root.
    Notation: T1[T2]
    """
    readleave = find_marked_leaf(T1)
    readleave.marked = False
    if T2 is None:
        return root(T1)
    if readleave.parent is None:
        # T1 consists of a single marked root leaf.
        T2.parent = None
        return T2
    parent = readleave.parent
    if parent.left is readleave:
        parent.left = T2
    else:
        parent.right = T2
    T2.parent = parent
    return root(T1)


def oplus(T1: Optional[TreeNode], T2: Optional[TreeNode]) -> TreeNode:
    """Create a new root r and attach T1 and T2 as its left and right subtrees. 
    If T1 or T2 is not None their parent pointers are updated to the new root.
    """
    r = TreeNode()
    r.left = T1
    r.right = T2
    if T1 is not None:
        T1.parent = r
    if T2 is not None:
        T2.parent = r
    return r


def assign_labels(root: TreeNode) -> TreeNode:
    """Assign natural number labels to all nodes in tree; root gets label 0."""
    counter = 0
    stack = [root]
    while stack:
        node = stack.pop(0)
        node.label = counter
        counter += 1
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    return root

def assign_leaves(root: TreeNode) -> tuple[TreeNode, list[TreeNode]]:
    """Assign natural number labels to all leaf nodes in tree; leftmost leaf gets label 0."""
    leaves = []
    counter = 0
    stack = [root]
    while stack:
        node = stack.pop(0)
        if is_leaf(node):
            node.label = counter
            leaves.append(node)
            counter += 1
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)

    return root, leaves


def print_tree(tree_root: TreeNode, filename: Optional[str] = None) -> None:
    """Print tree either to console (whitespace/ASCII) or file (graph notation).

    Args:
        tree_root: root of the tree to print
        filename: if provided, write graph notation (vertices and edges) to file;
                  otherwise print ASCII art with whitespace to console
    """
    tree_root = assign_labels(tree_root)

    if filename is not None:
        # Write graph notation to file
        with open(filename, "w") as f:
            stack = [tree_root]
            while stack:
                node = stack.pop(0)
                if node.left is not None:
                    stack.append(node.left)
                    f.write(f"{node.label},{node.left.label}\n")
                if node.right is not None:
                    stack.append(node.right)
                    f.write(f"{node.label},{node.right.label}\n")
        f.close() 

    # Print ASCII art to console
    lines = []

    def format_tree(node: Optional[TreeNode], prefix: str = "", is_left: Optional[bool] = None) -> None:
        if node is None:
            return
        marked_str = "*" if node.marked else ""
        if is_left is None:
            # Root
            lines.append(f"{node.label}{marked_str}")
        else:
            connector = "├── " if is_left else "└── "
            lines.append(prefix + connector + f"{node.label}{marked_str}")
        
        if is_left is None:
            prefix_left = ""
            prefix_right = ""
        else:
            prefix_left = prefix + ("│   " if is_left else "    ")
            prefix_right = prefix + ("│   " if is_left else "    ")
        
        if node.left is not None:
            format_tree(node.left, prefix_left, is_left=True)
        if node.right is not None:
            format_tree(node.right, prefix_right, is_left=False)

    format_tree(tree_root)
    for line in lines:
        print(line)


def _lca(a: Optional[TreeNode], b: Optional[TreeNode]) -> Optional[TreeNode]:
    """Lowest common ancestor of two nodes (or None if either is None)."""
    if a is None or b is None:
        return None
    da = depth(a)
    db = depth(b)
    while da > db:
        a = a.parent # type: ignore
        da -= 1
    while db > da:
        b = b.parent # type: ignore
        db -= 1
    while a is not b:
        a = a.parent # type: ignore
        b = b.parent # type: ignore
    return a


def _lca_three(a: Optional[TreeNode], b: Optional[TreeNode], c: Optional[TreeNode]) -> Optional[TreeNode]:
    """LCA of three nodes: lca(lca(a,b), c)."""
    ab = _lca(a, b)
    return _lca(ab, c)


def _is_strict_descendant(desc: TreeNode, anc: TreeNode) -> bool:
    """Return True if desc is a strict descendant of anc (anc above desc)."""
    node = desc
    if node == anc:
        return False
    while node.parent is not None:
        node = node.parent
        if node is anc:
            return True
    return False


def compute_C_relation(root: TreeNode) -> tuple[int, set[tuple[int, int, int]]]:
    """Compute the C-relation of `root`.

    Returns a set of triples (x,y,z) where each entry is the natural-number
    label of a node (assigned by breadth-first numbering starting at 0).

    C(x,y,z) holds iff the youngest common ancestor of y and z is strictly
    below the youngest common ancestor of x, y, and z.
    """
    root, leaves = assign_leaves(root)
    C = set()
    for x in leaves:
        for y in leaves:
            for z in leaves:
                lca_yz = _lca(y, z)
                lca_xyz = _lca_three(x, y, z)
                if lca_yz is None or lca_xyz is None:
                    continue
                if _is_strict_descendant(lca_yz, lca_xyz):
                    C.add((x.label, y.label, z.label))
    return len(leaves), C

def r2r_universal(n: int):
    if n == 1:
        T = TreeNode()
        TR = TreeNode(marked=True)
        TL = TreeNode()
        T.left = TL
        T.right = TR
        TL.parent = T
        TR.parent = T
    else:
        n_T = tree_universal(n)
        half_r2r = r2r_universal(n // 2)
        T = merge(half_r2r, oplus(copy.deepcopy(half_r2r), n_T))

    return T

def tree_universal(n: int):
    if n == 1:
        T = TreeNode()
    else:
        half_T = tree_universal(n // 2)
        half_r2r = r2r_universal(n // 2)
        T = merge(half_r2r, oplus(half_T, copy.deepcopy(half_T)))
    return T 


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter,
                                    prog="c-relation")
    ap.add_argument('n', type=int,
                    help='compute the an n-embedding universal structure')
    ap.add_argument('--redleaf', action=argparse.BooleanOptionalAction, default=False,
                        help='compute the an n-embedding universal redleaf structure')
    ap.add_argument('--c-relation', action=argparse.BooleanOptionalAction, default=False,
                            help='converts outputed tree to C-relation; red leaves are treated as normal leaves; file must be set')
    ap.add_argument('-f', '--file', type=str, default=None,
                                help='file for output of the n-embedding universal structure')
    
    args = ap.parse_args()
    if args.redleaf:
        T = r2r_universal(args.n)
    else:
        T = tree_universal(args.n)

    if args.c_relation:
        if args.file is None:
            print("Error: --file must be set when using --c-relation")
        else:
            n, C = compute_C_relation(T)
            with open(args.file, "w") as f:
                f.write(f"{n}\n\n")
                for triple in C:
                    f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
            f.close()
            print(f"C-relation written to {args.file}")
    else:
        print_tree(T,filename=args.file)
    