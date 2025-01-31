import itertools
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

sys.setrecursionlimit(1 << 25)  # increase recursion limit for deep trees


@dataclass
class Example:
    features: Dict[str, int]  # feature name -> value
    is_positive: bool  # label


@dataclass
class Node:
    feature: Optional[str] = None  # None for leaf nodes
    threshold: Optional[int] = None  # split: <= threshold left, > threshold right
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    is_leaf: bool = False
    is_positive: Optional[bool] = None  # only for leaf nodes


#
# stage 1
#


def mindt(examples: List[Example], s: int) -> Optional[Node]:
    """find minimal decision tree with at most s nodes (algorithm 3)"""
    if not examples:
        return None

    global gamma
    gamma = compute_global_assignment(examples)

    support_sets = enumerate_minimal_support_sets(examples, s)
    best_tree = None
    for S in support_sets:
        tree = mindts(examples, s, S)
        if tree and (best_tree is None or count_nodes(tree) < count_nodes(best_tree)):
            best_tree = tree

    return best_tree if best_tree and count_nodes(best_tree) <= s else None


def mindts(examples: List[Example], s: int, S: Set[str]) -> Optional[Node]:
    """find minimal tree using features in S and branch with R0 (algorithm 4)"""
    # find minimal tree with features S
    current_tree = find_minimal_tree(examples, S, s)
    if current_tree is None:
        return None

    # compute branching set R0
    R0 = compute_branching_set(examples, S)

    # recursively try adding each feature in R0
    best_tree = current_tree
    for f in R0:
        new_S = S.union({f})
        subtree = mindts(examples, s, new_S)
        if subtree and count_nodes(subtree) < count_nodes(best_tree):
            best_tree = subtree
    return best_tree if count_nodes(best_tree) <= s else None


def compute_global_assignment(examples: List[Example]) -> Dict[str, int]:
    """compute global assignment gamma (lemma 13)"""
    return examples[0].features.copy() if examples else {}  # first assignment (arbitrary)


def enumerate_minimal_support_sets(examples: List[Example], s: int) -> List[Set[str]]:
    """enumeration of minimal support sets of size up to s (corollary 9)"""
    # using a backtracking approach

    # compute all delta sets (differences between positive and negative examples)
    delta_sets = []
    E_plus = [e for e in examples if e.is_positive]
    E_minus = [e for e in examples if not e.is_positive]

    for e_p in E_plus:
        for e_m in E_minus:
            delta = set()
            # collect all features present in either example
            all_features = set(e_p.features.keys()).union(e_m.features.keys())
            for f in all_features:
                val_p = e_p.features.get(f, None)
                val_m = e_m.features.get(f, None)
                if val_p != val_m:
                    delta.add(f)
            if delta:  # ensure delta is non-empty (as per CI definition)
                delta_sets.append(frozenset(delta))

    # remove duplicate delta sets
    unique_delta = list({d for d in delta_sets if d})
    if not unique_delta:
        return []  # no differences to cover

    results = set()

    def backtrack(current_set: Set[str], index: int):
        if index == len(unique_delta):
            # check if current_set is minimal
            for f in list(current_set):
                subset = current_set - {f}
                # check if subset is a hitting set
                is_hitting = True
                for d in unique_delta:
                    if not subset & d:
                        is_hitting = False
                        break
                if is_hitting:
                    return  # not minimal
            if len(current_set) <= s:
                results.add(frozenset(current_set))
            return

        current_d = unique_delta[index]
        # check if current_set already hits current_d
        if current_set & current_d:
            backtrack(current_set, index + 1)
        else:
            for f in current_d:
                new_set = current_set | {f}
                if len(new_set) > s:
                    continue
                # prune if new_set is a superset of any existing result
                if any(existing.issubset(new_set) for existing in results):
                    continue
                backtrack(new_set, index + 1)

    backtrack(set(), 0)

    # filter to ensure minimality (remove sets that have subsets in results)
    minimal_support = []
    for candidate in results:
        candidate_set = set(candidate)
        if len(candidate_set) > s:
            continue
        is_minimal = True
        # check all subsets with one fewer element
        for f in candidate_set:
            subset = candidate_set - {f}
            if subset in results:
                is_minimal = False
                break
        if is_minimal:
            minimal_support.append(candidate_set)

    # remove duplicates and sort by size and elements for deterministic order
    minimal_support = list({frozenset(s): s for s in minimal_support}.values())
    minimal_support = [set(s) for s in minimal_support]
    minimal_support.sort(key=lambda x: (len(x), sorted(x)))

    return minimal_support


def find_minimal_tree(examples: List[Example], S: Set[str], s: int) -> Optional[Node]:
    """find minimal tree using features in S (theorem 4)"""

    def generate_trees_of_size(n: int) -> List[Node]:
        if n == 0:
            return [Node(is_leaf=True)]

        trees = []
        for left_size in range(n):
            right_size = n - 1 - left_size
            left_subtrees = generate_trees_of_size(left_size)
            right_subtrees = generate_trees_of_size(right_size)

            for left in left_subtrees:
                for right in right_subtrees:
                    new_node = Node(left=left, right=right)
                    trees.append(new_node)
        return trees

    def collect_internal_nodes(tree: Node) -> List[Node]:
        nodes = []

        def traverse(node):
            if node.is_leaf or node is None:
                return
            nodes.append(node)
            traverse(node.left)
            traverse(node.right)

        traverse(tree)
        return nodes

    def check_uniform(examples: List[Example]) -> Optional[bool]:
        if not examples:
            return None
        label = examples[0].is_positive
        return label if all(e.is_positive == label for e in examples) else None

    def build_tree(node: Node, feat_map: Dict[int, str], thresholds: Dict[str, int]) -> Tuple[Node, bool]:
        if node is None:
            return None, False

        if node.is_leaf:
            return Node(is_leaf=True), True

        # get feature and threshold for current node
        feat = feat_map.get(id(node), None)
        if feat is None:
            return None, False

        threshold = thresholds.get(feat, None)
        if threshold is None:
            return None, False

        # recursively build children
        left_child, left_valid = build_tree(node.left, feat_map, thresholds)
        right_child, right_valid = build_tree(node.right, feat_map, thresholds)

        if not left_valid or not right_valid:
            return None, False

        return Node(feature=feat, threshold=threshold, left=left_child, right=right_child), True

    best_tree = None
    smallest_size = float("inf")

    # early check for uniform examples
    uniform_result = check_uniform(examples)
    if uniform_result is not None:
        return Node(is_leaf=True, is_positive=uniform_result)

    # iterate all possible tree sizes up to s
    for current_size in range(1, s + 1):
        # generate all possible tree structures with current_size internal nodes
        trees = generate_trees_of_size(current_size)

        for tree in trees:
            # collect internal nodes in this tree structure
            internal_nodes = collect_internal_nodes(tree)
            if not internal_nodes or len(internal_nodes) != current_size:
                continue  # skip invalid trees

            # generate all feature assignments for internal nodes
            features = list(S)
            for feature_comb in itertools.product(features, repeat=current_size):
                # create feature assignment dict
                feature_assignment = {id(node): feat for node, feat in zip(internal_nodes, feature_comb)}

                # check if valid thresholds exist
                thresholds = findth(examples, tree, feature_assignment)
                if thresholds is None:
                    continue

                # build actual tree structure with thresholds
                actual_tree, valid = build_tree(tree, feature_assignment, thresholds)
                if valid and count_nodes(actual_tree) <= s and count_nodes(actual_tree) < smallest_size:
                    best_tree = actual_tree
                    smallest_size = count_nodes(best_tree)

    return best_tree if best_tree else None


def compute_branching_set(examples: List[Example], S: Set[str]) -> Set[str]:
    """compute branching set R0 for support set S (lemma 14)"""
    global gamma

    # group examples by their S-feature values (partial assignments α)
    equivalence_classes = defaultdict(list)
    for e in examples:
        alpha = tuple(sorted((f, e.features[f]) for f in S))  # represents α
        equivalence_classes[alpha].append(e)

    # select one representative per non-empty equivalence class
    E_S = [next(iter(group)) for group in equivalence_classes.values()]

    # compute δ(e, γ) for each representative
    R0 = set()
    for e in E_S:
        delta = {f for f, val in e.features.items() if gamma.get(f) != val}
        R0.update(delta)

    return R0


#
# second stage
#


def findth(examples: List[Example], tree: Node, feature_assignment: Dict[str, str]) -> Optional[Dict[str, int]]:
    """find valid thresholds for tree (algorithm 1)"""
    # base case: leaf node
    if tree.is_leaf:
        if not examples:
            return {}
        is_positive = examples[0].is_positive
        is_uniform = all(e.is_positive == is_positive for e in examples)
        if not is_uniform:
            return None
        return {}

    # get feature of node from assignment
    feature = feature_assignment[id(tree)]

    # find largest valid threshold for left child
    threshold = binary_search(examples, tree, feature_assignment, feature, tree.left)

    # try right subtree first
    right_examples = [e for e in examples if e.features[feature] > threshold]
    right_assignment = findth(right_examples, tree.right, feature_assignment)
    if right_assignment is None:
        return None

    # then try left subtree
    left_examples = [e for e in examples if e.features[feature] <= threshold]
    left_assignment = findth(left_examples, tree.left, feature_assignment)
    assert left_assignment is not None

    # combine assignments
    return {**{feature: threshold}, **left_assignment, **right_assignment}


def binary_search(examples: List[Example], tree: Node, feature_assignment: Dict[str, str], feature: str, left_child: Node) -> int:
    """find largest valid threshold for left child (algorithm 2)"""
    domain_values = sorted(set(e.features[feature] for e in examples))

    left = 0
    right = len(domain_values) - 1
    best_threshold = domain_values[0] - 1  # default if no valid threshold found

    while left <= right:
        mid = (left + right) // 2
        threshold = domain_values[mid]

        # try left subtree with current threshold
        left_examples = [e for e in examples if e.features[feature] <= threshold]
        left_result = findth(left_examples, left_child, feature_assignment)

        if left_result is not None:
            # valid threshold found, try larger ones
            best_threshold = threshold
            left = mid + 1
        else:
            # try smaller thresholds
            right = mid - 1

    return best_threshold


#
# utils
#


def count_nodes(tree: Optional[Node]) -> int:
    if tree is None or tree.is_leaf:
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)
