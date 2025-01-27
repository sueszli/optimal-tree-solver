from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Example:
    features: Dict[str, int]  # feature name -> value
    is_positive: bool

@dataclass
class Node:
    feature: Optional[str] = None  # None for leaf nodes
    threshold: Optional[int] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    is_leaf: bool = False
    is_positive: Optional[bool] = None  # Only for leaf nodes

def findth(examples: List[Example], tree: Node, feature_assignment: Dict[str, str]) -> Optional[Dict[str, int]]:
    """
    Implementation of Algorithm 1: FINDTH
    Returns threshold assignment if possible, None otherwise
    """
    # Base case: leaf node
    if tree.is_leaf:
        # Check if all examples have same classification
        if not examples:
            return {}
        is_positive = examples[0].is_positive
        if all(e.is_positive == is_positive for e in examples):
            return {}
        return None

    # Get feature from assignment
    feature = feature_assignment[id(tree)]
    
    # Get all possible thresholds for this feature
    thresholds = sorted(set(e.features[feature] for e in examples))
    
    # Try to find valid threshold
    for t in thresholds:
        # Split examples based on threshold
        left_examples = [e for e in examples if e.features[feature] <= t]
        right_examples = [e for e in examples if e.features[feature] > t]
        
        # Recursively try to find threshold assignments for subtrees
        left_assignment = findth(left_examples, tree.left, feature_assignment)
        if left_assignment is not None:
            right_assignment = findth(right_examples, tree.right, feature_assignment)
            if right_assignment is not None:
                # Combine assignments
                return {**{feature: t}, **left_assignment, **right_assignment}
    
    return None

# Example usage
def main():
    # Create some example data
    examples = [
        Example({'temp': 37, 'rain': 1}, is_positive=False),
        Example({'temp': 68, 'rain': 0}, is_positive=True),
        Example({'temp': 53, 'rain': 1}, is_positive=False),
        Example({'temp': 60, 'rain': 0}, is_positive=True),
    ]

    # Create a simple decision tree structure
    tree = Node(
        left=Node(is_leaf=True, is_positive=False),
        right=Node(is_leaf=True, is_positive=True)
    )

    # Create feature assignment
    feature_assignment = {id(tree): 'temp'}  # Assign 'temp' to root node

    # Find thresholds
    threshold_assignment = findth(examples, tree, feature_assignment)
    
    if threshold_assignment:
        print("Found valid threshold assignment:", threshold_assignment)
    else:
        print("No valid threshold assignment found")

if __name__ == "__main__":
    main()
