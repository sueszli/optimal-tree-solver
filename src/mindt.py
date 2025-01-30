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


def minDT(examples: List[Example], s: int) -> Optional[Node]:
    """find minimal decision tree with at most s nodes (algorithm 3)"""
    gamma = compute_global_assignment(examples)
    support_sets = enumerate_minimal_support_sets(examples, s)
    best_tree = None
    for S in support_sets:
        tree = minDTS(examples, s, S)
        if tree and (best_tree is None or count_nodes(tree) < count_nodes(best_tree)):
            best_tree = tree
    return best_tree


def minDTS(examples: List[Example], s: int, S: Set[str]) -> Optional[Node]:
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
        if len(new_S) > s:
            continue
        subtree = minDTS(examples, s, new_S)
        if subtree is not None and (count_nodes(subtree) < count_nodes(best_tree)):
            best_tree = subtree
    return best_tree if count_nodes(best_tree) <= s else None


def compute_global_assignment(examples: List[Example]) -> Dict[str, int]:
    """compute global assignment gamma (lemma 13)"""
    # TODO: implement
    pass


def enumerate_minimal_support_sets(examples: List[Example], s: int) -> List[Set[str]]:
    """enumeration of minimal support sets (corollary 9)"""
    # TODO: implement
    pass


def find_minimal_tree(examples: List[Example], S: Set[str], s: int) -> Optional[Node]:
    """find minimal tree using features in S (theorem 4)"""
    # TODO: implement
    pass


def compute_branching_set(examples: List[Example], S: Set[str]) -> Set[str]:
    """compute branching set R0 for support set S (lemma 14)."""
    # TODO: implement
    pass


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
