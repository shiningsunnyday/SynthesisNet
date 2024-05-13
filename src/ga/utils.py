import random
import uuid
from typing import Dict
from collections import defaultdict

import networkx as nx
import numpy as np

from synnet.utils.data_utils import skeleton2graph


def random_boolean(p: float) -> bool:
    return np.random.random() < p


def random_bitmask(n: int, k: int) -> np.ndarray:
    mask = np.zeros([n], dtype=bool)
    mask[:k] = True
    np.random.shuffle(mask)
    return mask


def random_name() -> str:
    return str(uuid.uuid4())


# TODO: double check
# Reference
#   https://people.math.carleton.ca/~kcheung/blog/haskell/random_bal_paren.html
#   https://dl.acm.org/doi/10.1145/357084.357091
def random_parentheses_string(n: int) -> Dict[int, int]:
    pairs = {}  # key = index of (, value index of )
    stack = []

    for i in range(2 * n):
        r = len(stack)  # number of unclosed '('s
        k = (2 * n) - i  # number of characters left
        p = r * (k + r + 2) / (2 * k * (r + 1))
        if random_boolean(p):  # )
            pairs[stack.pop(-1)] = i
        else:  # (
            stack.append(i)

    return pairs


def random_binary_tree(n: int) -> nx.DiGraph:
    tree = nx.DiGraph()
    pairs = random_parentheses_string(n)

    # (L)R => L <- root -> R
    def _recurse(start, end):
        if start == end:
            return None
        root = random_name()
        tree.add_node(root)
        l = _recurse(start + 1, pairs[start])
        r = _recurse(pairs[start] + 1, end)
        if l is not None:
            tree.add_edge(root, l, left=True)
        if r is not None:
            tree.add_edge(root, r, left=False)
        return root

    _recurse(0, 2 * n)
    return tree


def random_add_leaf(tree: nx.DiGraph) -> None:
    nonfull = [v for v, d in tree.out_degree() if (d < 2)]
    parent = random.choice(nonfull)
    if tree.out_degree(parent) == 0:
        left = random_boolean(0.5)
    else:
        existing_edge = list(tree.out_edges(parent, data="left"))[0]
        left = (not existing_edge[-1])
    child = random_name()
    tree.add_node(child)
    tree.add_edge(parent, child, left=left)


def random_remove_leaf(tree: nx.DiGraph) -> None:
    if tree.number_of_nodes() == 1:  # safety, don't remove root
        return
    leaves = [v for v, d in tree.out_degree() if (d == 0)]
    victim = random.choice(leaves)
    tree.remove_node(victim)


# TODO: is there something built-in?
def descendant_counts(tree: nx.DiGraph) -> Dict[str, int]:
    counts = dict()
    for node in reversed(list(nx.topological_sort(tree))):
        counts[node] = 1 + sum(counts[child] for child in tree.neighbors(node))
    return counts


def random_graft(
    donee: nx.DiGraph,
    donor: nx.DiGraph,
    min_nodes: int,
    max_nodes: int,
) -> nx.DiGraph:
    allowed = []
    for (old, old_descs) in descendant_counts(donee).items():
        if donee.in_degree(old) == 0:  # don't replace root
            continue
        for (new, new_descs) in descendant_counts(donor).items():
            merged_size = donee.number_of_nodes() - old_descs + new_descs
            if min_nodes <= merged_size <= max_nodes:
                allowed.append((new, old))
    if not allowed:
        return donee.copy()
    new, old = random.choice(allowed)

    old_subtree = nx.descendants(donee, old) | {old}
    new_subtree = nx.descendants(donor, new) | {new}

    merged: nx.DiGraph = nx.union(donee, donor.subgraph(new_subtree))
    parent, _, left = list(merged.in_edges(old, data="left"))[0]
    merged.add_edge(parent, new, left=left)
    merged.remove_nodes_from(old_subtree)

    # Need to relabel nodes to avoid conflict in future
    nx.relabel_nodes(merged, {k: random_name() for k in merged}, copy=False)

    return merged


def skeleton_to_binary_tree(skeleton):
    bt = nx.MultiDiGraph()
    lookup = defaultdict(random_name)
    for e in skeleton.tree.edges:
        src = lookup[skeleton.nodes[e[0]]]
        dst = lookup[skeleton.nodes[e[1]]]
        left = dst["child"] == "left"
        bt.add_edge(src, dst, left=left)
    return bt
