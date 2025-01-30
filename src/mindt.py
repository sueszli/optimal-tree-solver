import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class Example:
    features: Dict[str, int]
    is_positive: bool


@dataclass
class Node:
    feature: Optional[str] = None
    threshold: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    is_leaf: bool = False
    is_positive: Optional[bool] = None


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


def mindts(examples: List[Example], s: int, S: Set[str], gamma: Dict[str, int]) -> Optional[Node]:
    """find minimal tree using features in S and branch with R0 (algorithm 4)"""
    # step 1: find minimal tree with features S
    current_tree = find_minimal_tree(examples, S, s)
    if current_tree is None:
        return None

    # step 2: compute branching set R0
    R0 = compute_branching_set(examples, S)

    # step 3: recursively try adding each feature in R0
    best_tree = current_tree
    for f in R0:
        new_S = S.union({f})
        subtree = mindts(examples, s, new_S, gamma)
        if subtree and count_nodes(subtree) < count_nodes(best_tree):
            best_tree = subtree
    return best_tree if count_nodes(best_tree) <= s else None


def compute_global_assignment(examples: List[Example]) -> Dict[str, int]:
    """compute global assignment gamma (lemma 13)"""
    return examples[0].features.copy() if examples else {}  # first assignment (arbitrary)


def enumerate_minimal_support_sets(examples: List[Example], s: int) -> List[Set[str]]:
    """enumeration of minimal support sets of size up to s using brute-force (corollary 9)"""
    if not examples:
        return []
    features = list(examples[0].features.keys())
    E_plus = [e for e in examples if e.is_positive]
    E_minus = [e for e in examples if not e.is_positive]
    if not E_plus or not E_minus:
        return []

    support_sets = []

    # generate all subsets of features up to size s
    for k in range(1, s + 1):
        for subset in itertools.combinations(features, k):
            subset_set = set(subset)
            # check if subset is a support set
            is_support = True
            for e_p in E_plus:
                for e_n in E_minus:
                    if not any(f in subset_set and e_p.features[f] != e_n.features[f] for f in subset_set):
                        is_support = False
                        break
                if not is_support:
                    break
            if is_support:
                support_sets.append(subset_set)

    # filter minimal support sets
    minimal_support = []
    for candidate in support_sets:
        is_minimal = True
        # check all proper subsets
        for size in range(1, len(candidate)):
            for smaller_subset in itertools.combinations(candidate, size):
                smaller_set = set(smaller_subset)
                if smaller_set in support_sets:
                    is_minimal = False
                    break
            if not is_minimal:
                break
        if is_minimal:
            minimal_support.append(candidate)

    return minimal_support


def find_minimal_tree(examples: List[Example], S: Set[str], s: int) -> Optional[Node]:
    """find minimal tree using features in S (theorem 4)"""
    if not S:
        # handle empty support set case (all examples same class)
        if all(e.is_positive == examples[0].is_positive for e in examples):
            return Node(is_leaf=True, is_positive=examples[0].is_positive)
        return None

    best_tree = None
    # try trees with single node first
    if s >= 1:
        for feature in S:
            # create root node with this feature
            root = Node(feature=feature)
            feature_assignment = {id(root): feature}

            # find valid threshold using binary search
            domain = sorted({e.features[feature] for e in examples})
            left, right = 0, len(domain) - 1
            best_threshold = domain[0] - 1 if domain else 0
            valid = False

            while left <= right:
                mid = (left + right) // 2
                threshold = domain[mid]
                left_examples = [e for e in examples if e.features[feature] <= threshold]
                right_examples = [e for e in examples if e.features[feature] > threshold]

                # check if left and right are uniform
                left_uniform = all(e.is_positive == left_examples[0].is_positive for e in left_examples) if left_examples else True
                right_uniform = all(e.is_positive == right_examples[0].is_positive for e in right_examples) if right_examples else True

                if left_uniform and right_uniform:
                    best_threshold = threshold
                    valid = True
                    left = mid + 1
                else:
                    right = mid - 1

            if valid:
                root.threshold = best_threshold
                root.left = Node(is_leaf=True, is_positive=any(e.is_positive for e in examples if e.features[feature] <= best_threshold))
                root.right = Node(is_leaf=True, is_positive=any(e.is_positive for e in examples if e.features[feature] > best_threshold))
                if best_tree is None or 1 < count_nodes(best_tree):
                    best_tree = root
    return best_tree


def compute_branching_set(examples: List[Example], S: Set[str]) -> Set[str]:
    """compute branching set R0 for support set S (lemma 14)."""
    global gamma

    # compute E(S): one example per equivalence class
    equivalence_classes = {}
    for e in examples:
        key = tuple((f, e.features[f]) for f in sorted(S))
        if key not in equivalence_classes:
            equivalence_classes[key] = e
    E_S = list(equivalence_classes.values())

    # collect features not in S where examples disagree with gamma
    R0 = set()
    for e in E_S:
        for f in e.features:
            if f not in S and e.features[f] != gamma.get(f, None):
                R0.add(f)
    return R0


def count_nodes(tree: Optional[Node]) -> int:
    if tree is None or tree.is_leaf:
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)


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
