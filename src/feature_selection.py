from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

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

def get_domain_values(examples: List[Example], feature: str) -> List[int]:
    """Get sorted unique values for a feature across examples"""
    values = set()
    for e in examples:
        if feature in e.features:
            values.add(e.features[feature])
    return sorted(list(values))

def split_examples(examples: List[Example], feature: str, threshold: int) -> tuple[List[Example], List[Example]]:
    """Split examples into left (≤ threshold) and right (> threshold)"""
    left, right = [], []
    for e in examples:
        if feature in e.features:
            if e.features[feature] <= threshold:
                left.append(e)
            else:
                right.append(e)
    return left, right

def is_uniform(examples: List[Example]) -> Optional[bool]:
    """Check if all examples have same label. Returns None if empty."""
    if not examples:
        return None
    return all(e.is_positive == examples[0].is_positive for e in examples)

def minDT(examples: List[Example], features: Set[str], size_bound: int) -> Optional[Node]:
    """Find minimum size decision tree using given features"""
    # Base cases
    if not examples:
        return Node(is_leaf=True, is_positive=True)  # Default to positive
    
    uniform = is_uniform(examples)
    if uniform is not None:
        return Node(is_leaf=True, is_positive=uniform)
    
    if size_bound <= 0:
        return None
        
    # Try each feature and threshold
    best_tree = None
    for feature in features:
        domain_values = get_domain_values(examples, feature)
        
        for threshold in domain_values:
            left_examples, right_examples = split_examples(examples, feature, threshold)
            
            # Recursively build subtrees
            left_tree = minDT(left_examples, features, size_bound - 1)
            if left_tree is None:
                continue
                
            right_tree = minDT(right_examples, features, size_bound - 1)
            if right_tree is None:
                continue
            
            # Valid tree found
            tree = Node(
                feature=feature,
                threshold=threshold,
                left=left_tree,
                right=right_tree
            )
            best_tree = tree
            break
            
        if best_tree is not None:
            break
            
    return best_tree

def minDTS(examples: List[Example], size_bound: int) -> Optional[Node]:
    """Find minimum size decision tree"""
    # Get all features
    features = set()
    for e in examples:
        features.update(e.features.keys())
    
    # Try increasingly larger feature subsets
    for size in range(1, min(len(features), size_bound) + 1):
        # Try each feature subset of current size
        for feature_subset in get_feature_subsets(features, size):
            tree = minDT(examples, feature_subset, size_bound)
            if tree is not None:
                return tree
    
    return None

def get_feature_subsets(features: Set[str], size: int) -> List[Set[str]]:
    """Helper to get all feature subsets of given size"""
    if size == 0:
        return [set()]
    if not features:
        return []
    
    feature = next(iter(features))
    rest = features - {feature}
    
    with_feature = [{feature} | s for s in get_feature_subsets(rest, size-1)]
    without_feature = get_feature_subsets(rest, size)
    
    return with_feature + without_feature
