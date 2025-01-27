from typing import Dict, List, Optional
from dataclasses import dataclass


# 
# training
# 


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
# eval
# 

if __name__ == "__main__":
    # 
    # dataset based on figure 2
    # 

    # examples = [
    #     Example({"temp": 37, "rain": 1, "time": 10, "day": 3}, is_positive=False),
    #     Example({"temp": 68, "rain": 0, "time": 60, "day": 2}, is_positive=True),
    #     Example({"temp": 53, "rain": 1, "time": 60, "day": 4}, is_positive=False),
    #     Example({"temp": 53, "rain": 0, "time": 15, "day": 2}, is_positive=False),
    #     Example({"temp": 60, "rain": 0, "time": 60, "day": 5}, is_positive=True),
    #     Example({"temp": 51, "rain": 0, "time": 40, "day": 3}, is_positive=True),
    #     Example({"temp": 71, "rain": 0, "time": 35, "day": 5}, is_positive=True)
    # ]

    examples = [
        Example({'temp': 68, 'rain': 0}, is_positive=True),
        Example({'temp': 60, 'rain': 0}, is_positive=True),
        Example({'temp': 53, 'rain': 1}, is_positive=False),
        Example({'temp': 37, 'rain': 1}, is_positive=False),
    ]

    tree = Node(
        left=Node(is_leaf=True, is_positive=False),
        right=Node(is_leaf=True, is_positive=True)
    )

    feature_assignment = {id(tree): 'temp'}  # assign 'temp' to root node
    

    threshold_assignment = findth(examples, tree, feature_assignment)
    if not threshold_assignment:
        print("no valid threshold assignment found")
        exit()
    
    # 
    # eval
    # 

    def init_tree(tree: Node, feature_assignment: Dict[int, str] = {}, threshold_assignment: Dict[str, int] = {}):
        if tree.is_leaf:
            return
        tree.feature = feature_assignment[id(tree)]
        tree.threshold = threshold_assignment[tree.feature]
        init_tree(tree.left, threshold_assignment)
        init_tree(tree.right, threshold_assignment)

    init_tree(tree, feature_assignment, threshold_assignment)

    for example in examples:
        actual = example.is_positive
        def get_prediction(tree: Node, example: Example):
            if tree.is_leaf:
                return tree.is_positive
            if example.features[tree.feature] <= tree.threshold:
                return get_prediction(tree.left, example)
            return get_prediction(tree.right, example)
        pred = get_prediction(tree, example)
        assert actual == pred, f"actual: {actual}, pred: {pred} - {example.features}"

    print("tree is valid")
