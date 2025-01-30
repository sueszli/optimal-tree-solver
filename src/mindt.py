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


def is_support_set(examples: List[Example], features: Set[str]) -> bool:
    pos = [e for e in examples if e.is_positive]
    neg = [e for e in examples if not e.is_positive]
    return all(any(p.features[f] != n.features[f] for f in features) for p in pos for n in neg)


def generate_minimal_support_sets(examples: List[Example], max_size: int) -> List[Set[str]]:
    all_features = set(examples[0].features.keys()) if examples else set()
    for size in range(1, max_size + 1):
        for candidate in itertools.combinations(all_features, size):
            candidate_set = set(candidate)
            if is_support_set(examples, candidate_set):
                return [candidate_set]  # Simplified minimal set generation
    return []


def build_optimal_tree(examples: List[Example], features: Set[str], max_size: int) -> Optional[Node]:
    # Simplified tree building - in practice would generate all possible tree structures
    if not examples:
        return None

    # Base case: all examples same class
    if all(e.is_positive for e in examples):
        return Node(is_leaf=True, is_positive=True)
    if all(not e.is_positive for e in examples):
        return Node(is_leaf=True, is_positive=False)

    # Try features and thresholds (simplified)
    for feature in features:
        values = sorted({e.features[feature] for e in examples})
        for threshold in values:
            left = [e for e in examples if e.features[feature] <= threshold]
            right = [e for e in examples if e.features[feature] > threshold]

            left_tree = build_optimal_tree(left, features, max_size - 1)
            right_tree = build_optimal_tree(right, features, max_size - 1)

            if left_tree and right_tree:
                return Node(feature=feature, threshold=threshold, left=left_tree, right=right_tree)
    return None


def minDTS(examples: List[Example], s: int, S: Set[str], current_features: Set[str]) -> Optional[Node]:
    if len(current_features) > s:
        return None

    # Build tree with current features
    tree = build_optimal_tree(examples, current_features, s)
    if not tree:
        return None

    # Simplified branching set (paper's R0)
    remaining_features = set(examples[0].features.keys()) - current_features if examples else set()

    # Try adding each feature from branching set
    best_tree = tree
    for f in remaining_features:
        new_tree = minDTS(examples, s, S, current_features | {f})
        if new_tree and count_nodes(new_tree) < count_nodes(best_tree):
            best_tree = new_tree

    return best_tree if count_nodes(best_tree) <= s else None


def minDT(examples: List[Example], s: int) -> Optional[Node]:
    minimal_support_sets = generate_minimal_support_sets(examples, s)
    best_tree = None

    for support_set in minimal_support_sets:
        tree = minDTS(examples, s, support_set, support_set.copy())
        if tree and (not best_tree or count_nodes(tree) < count_nodes(best_tree)):
            best_tree = tree

    return best_tree


def count_nodes(tree: Optional[Node]) -> int:
    if not tree or tree.is_leaf:
        return 0
    return 1 + count_nodes(tree.left) + count_nodes(tree.right)
