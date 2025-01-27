from typing import Dict, List, Optional
from dataclasses import dataclass


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

    threshold = binary_search(examples, tree, feature_assignment, feature, tree.left)
    
    # try right subtree first
    right_examples = [e for e in examples if e.features[feature] > threshold]
    right_assignment = findth(right_examples, tree.right, feature_assignment)
    if right_assignment is None:
        return None
        
    # then try left subtree
    left_examples = [e for e in examples if e.features[feature] <= threshold]
    left_assignment = findth(left_examples, tree.left, feature_assignment)
    if left_assignment is None:
        return None
    
    # combine assignments
    return {**{feature: threshold}, **left_assignment, **right_assignment}


def binary_search(examples: List[Example], tree: Node, feature_assignment: Dict[str, str], feature: str, left_child: Node) -> int:
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
        for e in examples:
            classification = "positive" if e.features['temp'] > threshold_assignment['temp'] else "negative"
            success: bool = classification == 'positive' if e.is_positive else classification == 'negative'
            if not success:
                print("error in classification")
                print(e.features)
    else:
        print("no valid threshold assignment found")
