from typing import Dict, List, Optional
from dataclasses import dataclass
import bisect


@dataclass
class Example:
    features: Dict[str, int]  # feature name -> value
    is_positive: bool # label


@dataclass
class Node:
    feature: Optional[str] = None  # None for leaf nodes
    threshold: Optional[int] = None # split: <= threshold left, > threshold right
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    is_leaf: bool = False
    is_positive: Optional[bool] = None  # only for leaf nodes


def findth(examples: List[Example], tree: Node, feature_assignment: Dict[str, str]) -> Optional[Dict[str, int]]:
    # base case
    if tree.is_leaf: # tree is already the root node
        # check if all examples have same classification (uniform)
        if not examples:
            return {}
        is_positive = examples[0].is_positive
        is_uniform = all(e.is_positive == is_positive for e in examples)
        if not is_uniform:
            return None
        return {} # empty assignment

    # get feature from assignment
    feature = feature_assignment[id(tree)]

    """
    simplified version of `binary_search`
    """

    # get all possible thresholds for this feature (domain value range)
    thresholds = sorted(set(e.features[feature] for e in examples), reverse=True) # largest first

    # find largest valid threshold
    for t in thresholds:
        # split examples based on threshold
        left_examples = [e for e in examples if e.features[feature] <= t]
        right_examples = [e for e in examples if e.features[feature] > t]
        
        # recursively try to find threshold assignments for subtrees
        left_assignment = findth(left_examples, tree.left, feature_assignment)
        if left_assignment is not None:
            right_assignment = findth(right_examples, tree.right, feature_assignment)
            if right_assignment is not None:
                # combine assignments
                return {**{feature: t}, **left_assignment, **right_assignment}

    return None


if __name__ == "__main__":
    examples = [
        Example({'temp': 37, 'rain': 1}, is_positive=False),
        Example({'temp': 68, 'rain': 0}, is_positive=True),
        Example({'temp': 53, 'rain': 1}, is_positive=False),
        Example({'temp': 60, 'rain': 0}, is_positive=True),
    ]

    tree = Node(
        left=Node(is_leaf=True, is_positive=False),
        right=Node(is_leaf=True, is_positive=True)
    )

    feature_assignment = {id(tree): 'temp'}  # assign 'temp' to root node
    
    threshold_assignment = findth(examples, tree, feature_assignment)
    
    if threshold_assignment:
        print("found valid threshold assignment:", threshold_assignment)
    else:
        print("no valid threshold assignment found")
