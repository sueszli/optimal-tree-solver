> Ordyniak, S., & Szeider, S. (2021). Parameterized Complexity of Small Decision Tree Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 35 (7), 6454-6462. https://www.ac.tuwien.ac.at/files/tr/ac-tr-21-002.pdf

# introduction & preliminaries

*motivation*

- small decision trees:
	- pros: fewer tests, easier to interpret, generalize better, faster eval
	- cons: expensive to train
- in general, deciding whether a tree with given size/depth can fit the data is np-hard and can be done using SAT solvers
- we want to study the parameterized complexity to get runtime guarantees based on the "problem parameters" (size/depth for given data)
- we want to study the scalability of algorithms to large inputs as long as the problem parameters remain small

*problems*

- decision problems, but can also output the tree in practice
- DTS - minimum decision tree size
	- decide whether there exists a decision tree with at most $s$ tests or report that no such tree exists
- DTD - minimum decision tree depth
	- decide whether there exists a decision tree with at most depth $d$ or report that no such tree exists

*problem parameters*

- reasonable for these to assume to be small
- (1) size $s$ = total number of tests (non-leaf nodes). each test determines whether a certain feature is below a certain threshold or not
- (1) depth $s$ = the number of tests on any root-to-leaf path
- (2) maximum domain size $D_{\max}$ = total number of unique values for all features
- (3) maximum difference $\delta_{\max}$ = the highest number of features two examples with a different classification can disagree on (see table 1 to see that this is small in practice)

*definition*

- decision tree for binary classification on numerical values
- features $f \in \text{feat}(E)$
	- possibly infinite, linearly ordered range of $D \in \mathbb Z$
	- value range $D_E(f)$ of a feature
	- maximum domain size $D_{\max}$ = largest value range for any feature
- examples $e \in E$
	- all examples have the same features: $\forall e_i, e_j \in E: \text{feat}(e_i) = \text{feat}(e_j)$
	- a "classification instance CI" or just "instance" $E$ is the union of all positive $E^+$ and negative examples $E^-$
	- an example can only have one label (disjoint/non-overlapping union)
	- a set of exampels is "uniform" if they all have the same labels
	- for complexity analysis we assume that $|E| = |E| \cdot (|feat(E)| + 1) \cdot \log D_{max}$
- support set $F$
	- = a set of features that can distinguish between all positive and negative examples
	- = all features that examples with different labels assign different values to: $F = \delta(e_1, e_2)$ where $e_1 \in E^+,~ e_2 \in E^-$
	- finding the smallest support set (MSS problem) is np-hard
- disagreeing examples
	- difference $\delta(e_1, e_2) = \forall f: e_1(f) \not= e_2(f)$ → set of features that two examples "disagree" on / assign different feature values to
	- maximum difference $\delta_\max(E) = \max_{e^+\in E^+\wedge e^-\in E^-}|\delta(e^+,e^-)|$ → highest number of disagreements among any two examples with different labels
- agreeing examples
	- $E[\alpha]$ = set of examples that "agree" on / assign the value $\alpha$ to a set of features $F'$
- decision tree
	- $T$ = vertices $V$, arcs/edges $A$
	- each $v \in V(T)$ has a:
		- test threshold $\lambda(v)$ where the left path is $e(\text{feat}(u)) \leq \lambda(u)$ and the right path is $e(\text{feat}(v)) > \lambda(v)$
		- feature $\text{feat}(v)$
		- subtree $E_T(v)$

*observation 1*

- 💡✨ the features of a decision tree $T$ are the "support set" of $E$
- = any decision tree's features must form a support set (seperate positive from negative examples)
- $\text{feat}(T) = \{\text{feat}(v) \mid v \in V(T)\}$ = all features used for tests
- proof by contradiction:
	- assuming two examples with different labels would agree on any of the features / assign the same value to it, then they would wrongly end up in the same leaf (which is impossible)

*fixparam complexity*

- https://en.wikipedia.org/wiki/Parameterized_complexity
- [Downey, R. G., & Fellows, M. R. (2013). Fundamentals of parameterized complexity (Vol. 4). London: springer.](https://link.springer.com/content/pdf/10.1007/978-1-4471-5559-1.pdf)
- problem instances expressed as $(I, k)$
- FTP = fixed parameter tractable means there is a function $g$ to solve the instance in time $g(k)\cdot||I||^{O(1)}$
- XP = xp-tractable means there is a function $g$ to solve the instance in time $||I||^{g(k)}$
- paraNP-hard = fixing the parameter to a constant gives an NP-hard problem
- $W[1] \subseteq W[2] \subseteq \ldots$ = weft complexity hierarchy
	- usually problems that are XP-tractable but not FTP
	- hardness for any of these is strong evidence that the problem is not FTP

# hardness results

results:

- FTP tractable = $\{sol, \delta_\max, D_\max\}$
	- authors believe that $\{sol, \delta_\max \}$ may also be FTP tractable - they say this is the only open question
- XP tractable = $\{sol\},~ \{sol, \delta_\max\},~ \{sol,  D_\max\}$
- w[2] tractable = $\{sol\},~ \{sol,  D_\max\}$
- paraNP hard = $\emptyset,~ \{\delta_\max\},~ \{D_\max\},~ \{\delta_\max, D_\max\}$
- $sol$ stands for the solution size, which are the $s, d$ parameters

*hitting set problem*

- input: 
	- $\mathcal F \subseteq U$ = collection of sets
		- $\Delta$ = maximum arity, size of the largest set $F \in \mathcal F$
	- $U$ = universe (of all possible elements)
	- $k = |H|$ = maximum size allowed for the hitting set
		- $H \subseteq U$ = hitting set 
- goal:
	- given an instance $\mathcal I = (\mathcal F, U, k)$
	- find a subset $H$ containing at most $k$ elements that interset with every set in the collection (non-empty intersections $\forall F \in \mathcal F: F \cap H = \emptyset$)
- hitting set → decision tree:
	- done with $E(\mathcal I)$
	- each row in dataset is a set in the collection $F \in \mathcal F$
	- each feature is a binary flag on whether that set contains the element $u\in U$
		- i) empty set  $\emptyset$: set all flags to 0 → label: positive
		- ii) all sets in collection: binary encode → label: negative
	- any valid decision tree must distinguish the single positive example from all negative examples. this forces it to identify features that "hit" all the negative examples (which represent the sets from $F$)
	- the features in the decision tree (support set) are equivalent to the nodes that need to be selected to hit all sets in the collection
	- the reduction preserves the parameter $k$ (hitting set size becomes tree size/depth - the size and depth are the same)

*theorem 2 (important)*

- 💡✨ DTS and DTD parameterized by $s, d$ are W2-hard even with $D_{\max}\text{=}2$
- proof:
	- since hitting set is W2-hard when parameterized by $k$, both DTS and DTD inherit this hardness, which means:
		- the problems DTS and DTD cannot be solved in FPT time unless W2 = FPT
		- the hardness holds even when restricted to boolean instances with $D_{\max}\text{=}2$
- in conclusion: finding optimal trees is really hard. just using the $s, d$ and $D_{\max}$ as problem parameters alone does not yield fixed-parameter tractability FTP, even if $D_{\max}\text{=}2$

*theorem 3*

- 💡✨ DTS and DTD parameterized by $\delta_\max(E)$ are paraNP-hard even with $\Delta{=}2$
	- $\delta_\max(E(\mathcal I)) = \Delta_\max(\mathcal I)$ = the highest number of features two examples with a different classification can disagree on, is equivalent to the size of the largest set in the collection
	- the only positive example has all zeros, each negative example has ones exactly in positions corresponding to elements of its set $F$. therefore, the number of disagreements between the only positive and any negative examples equals the size of set $F$
- 💡✨ DTS and DTD parameterized by $\text{min}_{\#}(E)$ are paraNP-hard
	- where: $\text{min}_{\#}(E) = \min\{|E^+|, |E^-|\}$
	- in our reduction $\text{min}_{\#}(E(\mathcal I)) = 1$ because we have a single positive example
- 💡✨ DTS and DTD parameterized by the number of test nodes are paraNP-hard
	- due to $\text{min}_{\#}(E(\mathcal I)) = 1$ we have 0 branching nodes, just two leafs from the root

# algorithms

for the sake of simplicity only the DTS problem is mentioned

two stage algorithm:

- i) feature selection = finding a support set $S$ (a small set of features that can distinguish between all positive and negative examples)
	- → the first stage of the algorithm is discussed last in the paper
- ii) training = determining how to organize those features into an optimal tree structure

## stage 2: training

*theorem 4*

- 💡✨ given examples $E$, a support set $S$, an integer $s$, there exists an algorithm that runs in time $2^{\mathcal{O}(s^2)}\|E\|^{1+o(1)} \log \|E\|$ that finds the optimal/smallest decision tree using just the support set for decisions and is smaller than $s$ - if such tree exists, otherwise it returns `nil`
	- the runtime fits the FPT format of $f(sol) \cdot \text{poly}(n)$, where $f(sol) = 2^{\mathcal{O}(s^2)}$ is the function dependent only on solution size and $\|E\|^{1+o(1)} \log \|E\|$ is polynomial in the input
- once you have the right set of features (the support set $S$), finding the optimal tree structure is FTP-tractable when parameterized by solution size $s$ only, even though the general problem of finding optimal decision trees is NP-hard
	- = **the hardness of decision tree learning comes primarily from feature selection, rather than training**
	- = the running time is only exponential in the size of the solution (the tree size), while remaining polynomial in the input size
	- = the runtime is bounded by $f(sol) \cdot \text{poly}(n)$, where $f$ is some arbitrary function and $\text{poly}$ is a polynomial function
	- this suggests that you might be even able to reach FTP tractability using just $\{sol, \delta_\max\}$ without needing $D_\max$
- the feature selection must make sure to choose $k = |S| \leq s$ features
- proof follows with the algorithms below

*lemma 5, 6*

- 💡✨ given examples $E$, a support set $S$, an integer $s \geq k = |S|$, then there exists an algorithm that enumerates all pseudo tree and test feature assignment $(T, \alpha)$ pairs in time $O(s^s)$
- 💡✨ given examples $E$, a pseudo tree $T$, a feature assignment $\alpha$, then there exists an algorithm that decides whether $(T, \alpha)$ can have a decision criteria to become a valid decision tree in time $O(d^{d^2 / 2} \|E\|^{1+o(1)} \log \|E\|)$ where $d \leq s$
- we can enumerate all pseudo decision trees' feature assignments efficiently, but not the number of all thresholds, since the number of possible thresholds is potentially as large as the input size (that will need a heuristic)
- definition:
	- pseudo dt = pseudo decision tree, each non-leaf node must have exactly two children (left and right). only defines structure of tree, but doesn't have features or decision criteria (but they can be added and it will become a valid decision tree)
	- $\alpha: V(T) \mapsto \text{feat}(E)$ = feature assignment function, assigns features to each test node
	- $\gamma: V(T) \mapsto D$ = decision criteria assignment function, assigns numerical threshhold values to each test node
- proof:
	- = how to recursively determine if a pseudo tree with test feature assignment $(T, \alpha)$ can be extended into a valid decision tree using threshhold values
	- starting at the root, each test-node has two children
	- the root can only be a valid tree, if the left/right child-trees can also be recursively extended to a valid tree, using all examples where the values of the root's test feature $f$ are $\leq t$ or $> t$
	- the problem can therefore be reduced to recursively finding threshold $t$ among all values in the range $D_E(f)$ for each node
	- however, because the range is very large, we have to make use of the "monotonicity property of thresholds", which allows us to use binary search instead of searching exhaustively → if a valid threshold $t$ exists on the left subtree, it remains valid for any $t' \geq t$
- algorithm:
	- i. find the largest threshold $t$ for the feature at the root at which the left subtree is valid
	- ii. verify whether the right subtree is valid with that threshold

*algorithm 1, 2*

- this is an efficient way to find threshhold values for pseudo trees with assigned test-features
- $\text{findTH}$ algorithm:
	- input: examples $E$, pseudo tree $T$, test-feature assignments $\alpha$
	- output: test-threshhold assignments $\lambda$ (such that tree is valid) or `nil`
- $\text{binarySearch}$ subroutine:
	- finds the largest value for left sub tree
	- input: examples $E$, pseudo tree $T$, test-feature assignments $\alpha$, feature of root $f$, left child of root $c_l$
	- output: largest threshold in range $D_E(f)$ such that left subtree is a valid decision tree with all examples where $f \leq t$. otherwise the smallest value in the range, subtracted by one, is returned.
- runtime proof for theorem 4:
	- i. enumerate all pseudo trees and test-feature assignemnts ($T, \alpha$)
		- see lemma 5
		- runtime: $O(s^s)$
	- ii. for each ($T, \alpha$) pair we filter by those that have a valid threshold $t$, in order for them to become a decision tree.
		- see lemma 6
		- runtime: $O(d^{d^2 / 2} \|E\|^{1+o(1)} \log \|E\|)$ where $d \leq s$
		- = number of recursive $\text{findTH}$ calls $\cdot$ runtime of each call
			- number of recursive calls = $O(\log \|E\|^d)$ because it calls it self at most $\log \|E\| + 2$ times for each decision tree of smaller depth
			- runtime of each call = $O(\|E\| \log \| E \|)$
	- iii. we return the smallest of all valid decision trees or `nil` if it doesn't exist
		- $O(s^s) + O(d^{d^2 / 2} \|E\|^{1+o(1)} \log \|E\|)$
		- the second term dominates and is the final runtime

```python
def find_threshold(E: Examples, T: Tree, alpha: FeatureSetAssignment):
    r = T.get_root()
    
    # base case
    if r.is_leaf():
        if E.get_labels().unique().count() != 1:
           return None
        return {}
    
    f = r.get_test_feature()
    c_l = T.get_left_child()
    c_r = T.get_right_child()

    # get largest threshold for left child
    t = binary_search(E, T, alpha, c_l, f)
    # check if threshold is valid for right child
    lambda_r = find_threshold(E[f > t], c_r, alpha)
    if lambda_r is None:
        return None

    # continue recursive calls
    lambda_l = find_threshold(E[f <= t], c_l, alpha)
    return {
        r: t,
        **lambda_r,
        **lambda_l
    }

def binary_search(E: Examples, T: Tree, alpha: FeatureSetAssignment, f: Feature, c_l: Tree):
    D = f.get_values().sort(ascending=True) # array of feature value range
    
    L, R = 0, = len(f.get_values()) - 1 # search pointers
    b = False # success flag
    while L <= R:
        m = math.floor((L+R)/2) # middle value
        if find_threshold(E[f >= D[m]]) is not None: # check if condition is satisfied
            L = m + 1
            b = True
        else:
            R = m - 1
            b = False
    
    # success
    if b:
        return D[m]
    
    # failure
    if (m-1) == -1:
        return D[0] - 1 # pseudocode assumes that: D[-1] = D[0] - 1
    return D[m-1]
```

```python
def find_th(E, T, alpha):
    """
    Algorithm 1: Computes threshold assignment for a pseudo DT and feature assignment
    
    Args:
        E: Classification instance (CI)
        T: Pseudo decision tree
        alpha: Feature assignment for T
    
    Returns:
        Threshold assignment or None if no valid assignment exists
    """
    r = T.root
    if r.is_leaf():
        if not E.is_uniform():
            return None
        return {}  # Empty assignment
        
    f = alpha[r]
    cl, cr = r.left_child, r.right_child
    
    t = binary_search(E, T, alpha, cl, f)
    lambda_r = find_th(E.filter(f, '>', t), cr, alpha)
    
    if lambda_r is None:
        return None
        
    lambda_l = find_th(E.filter(f, '<=', t), cl, alpha)
    if lambda_l is None:
        return None
        
    return {f: t, **lambda_l, **lambda_r}

def binary_search(E, T, alpha, cl, f):
    """
    Algorithm 2: Binary search to find largest valid threshold
    
    Args:
        E: Classification instance
        T: Pseudo decision tree 
        alpha: Feature assignment
        cl: Left child node
        f: Feature to find threshold for
    
    Returns:
        Largest valid threshold or smallest value minus 1 if none exists
    """
    D = sorted(E.domain_values(f))
    L = 0
    R = len(D) - 1
    b = 0
    
    while L <= R:
        m = (L + R) // 2
        if find_th(E.filter(f, '<=', D[m]), cl, alpha) is not None:
            L = m + 1
            b = 1
        else:
            R = m - 1
            b = 0
            
    if b == 1:
        return D[m]
    return D[m-1] if m > 0 else D[0] - 1
```

```python
from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Union
from enum import Enum
import numpy as np

class NodeType(Enum):
    INTERNAL = 0
    LEAF = 1

@dataclass
class Example:
    """Represents a single example in the classification instance"""
    features: Dict[str, int]  # Feature name -> value mapping
    label: bool  # True for positive, False for negative
    
    def __getitem__(self, feature: str) -> int:
        return self.features[feature]

class Node:
    """Represents a node in the decision tree"""
    def __init__(self, node_type: NodeType, left=None, right=None):
        self.type = node_type
        self.left = left
        self.right = right
        self.feature = None  # For internal nodes
        self.threshold = None  # For internal nodes
        self.label = None  # For leaf nodes
        
    def is_leaf(self) -> bool:
        return self.type == NodeType.LEAF
    
    @property
    def left_child(self):
        return self.left
        
    @property
    def right_child(self):
        return self.right

class PseudoDT:
    """Represents a pseudo decision tree"""
    def __init__(self, root: Node):
        self.root = root
        
    @property
    def root(self) -> Node:
        return self._root
        
    @root.setter 
    def root(self, node: Node):
        self._root = node

class ClassificationInstance:
    """Represents a classification instance (CI)"""
    def __init__(self, examples: List[Example]):
        self.examples = examples
        self._compute_feature_domains()
        
    def _compute_feature_domains(self):
        """Computes domain values for each feature"""
        self.domains = {}
        for e in self.examples:
            for f, v in e.features.items():
                if f not in self.domains:
                    self.domains[f] = set()
                self.domains[f].add(v)
                
    def is_uniform(self) -> bool:
        """Returns True if all examples have same label"""
        if not self.examples:
            return True
        return all(e.label == self.examples[0].label for e in self.examples)
    
    def domain_values(self, feature: str) -> List[int]:
        """Returns sorted domain values for given feature"""
        return sorted(list(self.domains[feature]))
        
    def filter(self, feature: str, op: str, threshold: int) -> 'ClassificationInstance':
        """Returns new CI with examples filtered by feature condition"""
        filtered = []
        for e in self.examples:
            if op == '<=' and e[feature] <= threshold:
                filtered.append(e)
            elif op == '>' and e[feature] > threshold:
                filtered.append(e)
        return ClassificationInstance(filtered)

class FeatureAssignment:
    """Represents a feature assignment for nodes"""
    def __init__(self):
        self.assignments: Dict[Node, str] = {}
        
    def __getitem__(self, node: Node) -> str:
        return self.assignments[node]
        
    def __setitem__(self, node: Node, feature: str):
        self.assignments[node] = feature

def find_th(E: ClassificationInstance, T: PseudoDT, alpha: FeatureAssignment) -> Optional[Dict[str, int]]:
    """
    Algorithm 1: Computes threshold assignment for a pseudo DT and feature assignment
    """
    r = T.root
    if r.is_leaf():
        if not E.is_uniform():
            return None
        return {}  # Empty assignment
        
    f = alpha[r]
    cl, cr = r.left_child, r.right_child
    
    t = binary_search(E, T, alpha, cl, f)
    lambda_r = find_th(E.filter(f, '>', t), cr, alpha)
    
    if lambda_r is None:
        return None
        
    lambda_l = find_th(E.filter(f, '<=', t), cl, alpha)
    if lambda_l is None:
        return None
        
    return {f: t, **lambda_l, **lambda_r}

def binary_search(E: ClassificationInstance, T: PseudoDT, 
                 alpha: FeatureAssignment, cl: Node, f: str) -> int:
    """
    Algorithm 2: Binary search to find largest valid threshold
    """
    D = E.domain_values(f)
    L = 0
    R = len(D) - 1
    b = 0
    m = 0
    
    while L <= R:
        m = (L + R) // 2
        if find_th(E.filter(f, '<=', D[m]), cl, alpha) is not None:
            L = m + 1
            b = 1
        else:
            R = m - 1
            b = 0
            
    if b == 1:
        return D[m]
    return D[m-1] if m > 0 else D[0] - 1

# Example usage:
if __name__ == "__main__":
    # Create example data from Figure 2 in the paper
    examples = [
        Example({"temp": 37, "rain": 1, "time": 10, "day": 3}, False),
        Example({"temp": 68, "rain": 0, "time": 60, "day": 2}, True),
        Example({"temp": 53, "rain": 1, "time": 60, "day": 4}, False),
        Example({"temp": 53, "rain": 0, "time": 15, "day": 2}, False),
        Example({"temp": 60, "rain": 0, "time": 60, "day": 5}, True),
        Example({"temp": 51, "rain": 0, "time": 40, "day": 3}, True),
        Example({"temp": 71, "rain": 0, "time": 35, "day": 5}, True)
    ]
    
    ci = ClassificationInstance(examples)
    
    # Create a simple pseudo decision tree
    root = Node(NodeType.INTERNAL)
    root.left = Node(NodeType.LEAF)
    root.right = Node(NodeType.LEAF)
    tree = PseudoDT(root)
    
    # Create feature assignment
    alpha = FeatureAssignment()
    alpha[root] = "temp"
    
    # Find threshold assignment
    thresholds = find_th(ci, tree, alpha)
    print(f"Found thresholds: {thresholds}")
```

## stage 1: feature selection

*theorem 7*

- 💡✨ DTS and DTD are XP-tractable parameterized by $sol$
- theorem 4 says we can efficiently $O(d^{d^2 / 2} \|E\|^{1+o(1)} \log \|E\|)$ find an optimal decision tree if we know which features to use
- the maximum number of features the decision tree can use is $s$, therefore $O(|\text{feat}(E)|^s)$ feature cobinations are possible at most
- when $s$ is constant:
	- i) enumerate all $O(|feat(E)|^s)$ possible feature subsets
	- ii) for each subset, apply theorem 4's algorithm
	- total runtime = $O(|\text{feat}(E)|^s \cdot 2^{(s^2/2)} \cdot |E|^(1+o(1)) \log|E|)$
	- $|\text{feat}(E)|^s$ becomes polynomial in input size, $2^{(s^2/2)}$ becomes a constant - therefore total runtime becomes polynomial in input size
- this makes it XP-tractable because runtime can be expressed as $|\text{input}|^{f(s)}$ where $f(s)$ is some function of $s$
- but not FPT tractable because the degree of the polynomial depends on parameter $s$

*theorem 8 (main result)*

- 💡✨ DTS and DTD are FTP-tractable parameterized by $\{sol, \delta_\max, D_\max\}$
- as seen in theorem 2: just using the $s, d$ and $D_{\max}$ as problem parameters alone does not yield fixed-parameter tractability FTP, even if all the features are booleans $D_{\max}\text{=}2$
- however, you add one more restriction the problem becomes fixed-parameter tractable:
	- limiting the size of the sets in the hitting set instance = limiting maximum difference $\delta_{\max}$
	- this approach of adding carefully chosen restrictions to make hard problems manageable is called "deconstruction of intractability"
- adding the maximum difference $\delta_{\max}$ as additional parameter to solution size $s$ and maximum domain size $D_{\max}$ renders both problems, DTS and DTD, fixed-parameter tractable
- proof is in algorithm below

*corollary 9*

- 💡✨ given examples $E$ (with maximum difference $\delta_\max$), support size limit $k$, there is an algorithm that can enumerate all minimal support sets in time $O(\delta_\max(E)^k \cdot |E|)$
- = runtime to find minimal support sets
- $k$ = maximum size of support sets we're looking for
- $\delta_\max(E)$ = the maximum number of features any two examples with different classifications can disagree on
- for each position in the support set (up to $k$ positions)
	- we have at most $\delta_\max(E)$ choices of features
	- leading to at most $\delta_\max(E)^k$ possible combinations to check
    - each check requires $O(|E|)$ time to verify if it's a valid support set

*lemma 10*

- 💡✨ there are examples $E_D$ or $E_S$ such that when used on a decision tree, it's test features are not a minimal support set for any decision tree of minimum depth / size
- given observation 1 and collarary 9:
	- any decision tree's features must form a support set (seperate positive from negative examples)
	- we can find the minimal support set in just $O(\delta_\max(E)^k \cdot |E|)$
	- one might think optimal (minimum size/depth) DTs would only use features from minimal support sets, since they use the fewest features needed for classification → this way we could (1) enumerate all support sets, (2) check each one for optimality based on the generated decision tree, (3) pick the best result
	- however counter examples show that this is not the case
- this shows **optimal decision trees may need additional features beyond minimal support sets** → we can't solve feature selection problem for decision trees by solving the minimal suport set problem
- sometimes adding "unnecessary" features (from a support set perspective) can lead to better trees, fundamentally changing how the problem needs to be approached
- proof by 1 counter example

*lemma 11*

- 💡✨ given an optimal (min size/depth) tree $T$, that contains a support set $S$ - a subset of all the features used by its test nodes $\text{feat}(T)$ - the features that are not in that subset $R = \text{feat}(T) \textbackslash S$ are useful
- intuition:
	- we need to find features used in optimal (min size/depth) decision trees
	- we can't just use minimal support sets (proven by lemma 10)
	- every optimal decision tree $T$ must contain some minimal support set $S$
	- **additional features** $R = \text{feat}(T) \textbackslash S$ **must serve some purpose**
	- these features are "useful" if they help split/differentiate examples that look identical under $S$ (equivalence classes)
- definition of usefulness:
	- there must be some equivalence class under $S$ (defined by $\alpha$) that becomes empty when you add $R$'s feature tests
	- i) $E[\alpha]$ is non-empty → meaning this is a valid equivalence class
	- ii) $E[\alpha \cup \beta]$ is empty → meaning $R$ splits this class
	- where:
		- alpha and beta represent value combinations in examples, they are parameters that decide which example to choose
		- $E[\alpha]$ = all examples matching feature values of $S$
		- $E[\beta]$ = all examples matching feature values of $R$
		- $E[\alpha \cup \beta]$ = all examples matching both feature values from both sets
		- equivalence classes = all examples that have the same value for all features in $S$
- proof by contradiction:
	- we want to show that we can build a smaller decision tree, by removing features in $R$
	- assume $R$ is not empty (otherwise there is nothing to prove)
	- assume there is some assignment $\beta$ to $R$ that leaves all equivalence classes under $S$ (defined by $\alpha$) intact, therefore $E[\alpha \cup \beta] = \emptyset$
	- i) tree transformation
		- step 1: for all nodes testing features in $R$, follow path according to $\beta$, remove unused subtrees → the result is a tree $T''$ where $R$-nodes have single-children
		- step 2: find paths containing only $R$-nodes, contract the paths (remove the $R$ nodes by connecting parent to child directly) → the result is a tree $T'$ with only $S$-nodes having two children
	- ii) contradition
		- assume $T'$ is not a valid decision tree, meaning it must some leaf $l$ with mixed classifications
		- $E[\alpha^+ \cup \beta]$ is non empty and contains at least a positive example in that leaf
		- $E[\alpha^- \cup \beta]$ is non empty and contains at least a negative example in that leaf
		- this means the original tree $T$ must have also misclassified these examples - but $T$ was valid, so this is impossible
	- in conclusion, $R$ must be useful, because no smaller valid tree can exist

*lemma 12*

- 💡✨ let $E(S)$ contain at least one arbitrary example from each non-empty $E[\alpha]$. then every useful set $R$ must contain at least one feature from the union of disagreements $\bigcup_{e \in E(S)} \delta(e, \beta)$
- branching set $R_0$ = every useful set $R$ for support set $S$ must contain at least one feature from $R_0$
- idea:
	- for every $R$ there must be some equivalence class (that can't be distinguished) under $S$ that becomes empty (distinguishable) when you add $R$'s features to test nodes
	- for every example $e \in E[\alpha]$ the set $R$ must contain at least one feature $f$ for which $e$ disagrees with $\beta$, such that the example set $E[\alpha \cup \beta]$ becomes empty
	- meaning $R \subseteq \delta(e, \beta)$

*lemma 13*

- 💡✨ given examples $E$ there is an assignment $\gamma$ (computable in polynomial time) such that every example disagres with at most $2 \cdot \delta_\max(E)$ features from examples in that assignment.
- the union of disagreements $\bigcup_{e \in E(S)} \delta(e, \beta)$ can get large, so we have to bound it
- global assignment $\gamma$ = subset of examples that disagree with at most $2 \cdot \delta_\max(E)$ features, efficient to compute
- works by just copying
- if you pick any example $e_1$ and use its values for $\gamma$, then any other example $e_2$ can differ from $e_1$ in at most $2 \delta_\max(E)$ features
	- this is by the definition of $\delta_\max(E)$ being the maximum number of features where a positive and negative example can differ
	- because both $e_1$ and $e_2$ can only differ from $e$ at most in $\delta_\max(E)$ features

*lemma 14*

- 💡✨ there is a polynomial time algorithm that given a support set $S$ computes the branching set $R_0$ for $S$ of size at most $D_\max^{|S|} \cdot 2 \delta_\max(E)$
- second factor:
	- the union of disagreements $\bigcup_{e \in E(S)} \delta(e, \beta)$ contains at most $2 \delta_\max(E)$ features for all examples in $E(S)$
- first factor:
	- $|E(S)| \leq D^{|S|}_\max$ because for every feature in $S$ we can have at most $D_\max$ values from that domain

*algorithm 3, 4*

- $\text{minDT}$ algorithm:
	- finds minimum size decision tree
	- input: dataset $E$, integer $s$
	- output: tree or `nil`
- $\text{minDTS}$ algorithm (subroutine):
	- finds minimum size decision tree, using at least the features in a given support set $S$
	- input: dataset $E$, support set $S$ for $E$ with $|S| \leq s$
	- output: tree using $S$ for test nodes or `nil`

```python
from typing import List, Set, Dict, Optional
from dataclasses import dataclass

@dataclass
class Example:
    features: Dict[str, int]  # feature name -> value
    label: bool  # True for positive, False for negative

@dataclass 
class DecisionTree:
    feature: Optional[str]  # None for leaf nodes
    threshold: Optional[int] # None for leaf nodes
    left: Optional['DecisionTree']  # None for leaf nodes  
    right: Optional['DecisionTree'] # None for leaf nodes
    is_leaf: bool
    leaf_label: Optional[bool] # Only for leaf nodes

def min_decision_tree_size(examples: List[Example], max_size: int) -> Optional[DecisionTree]:
    """
    Main algorithm for DTS (Minimum Decision Tree Size) problem
    Returns a decision tree with minimum size or None if impossible
    """
    # Get parameters
    features = get_features(examples)
    delta_max = get_max_difference(examples)
    d_max = get_max_domain_size(examples)
    
    # Step 1: Feature Selection
    candidate_feature_sets = enumerate_feature_sets(examples, max_size, delta_max, d_max)
    
    # Step 2: For each feature set, try to build optimal tree
    best_tree = None
    for feature_set in candidate_feature_sets:
        # Try to build tree using only these features
        tree = build_tree_with_features(examples, feature_set, max_size)
        if tree and (best_tree is None or tree_size(tree) < tree_size(best_tree)):
            best_tree = tree
            
    return best_tree

def min_decision_tree_depth(examples: List[Example], max_depth: int) -> Optional[DecisionTree]:
    """
    Main algorithm for DTD (Minimum Decision Tree Depth) problem
    Returns a decision tree with minimum depth or None if impossible
    """
    # Get parameters
    features = get_features(examples)
    delta_max = get_max_difference(examples)
    d_max = get_max_domain_size(examples) 
    
    # Step 1: Feature Selection
    candidate_feature_sets = enumerate_feature_sets(examples, max_depth, delta_max, d_max)
    
    # Step 2: For each feature set, try to build optimal tree
    best_tree = None
    for feature_set in candidate_feature_sets:
        # Try to build tree using only these features
        tree = build_tree_with_features_depth(examples, feature_set, max_depth)
        if tree and (best_tree is None or tree_depth(tree) < tree_depth(best_tree)):
            best_tree = tree
            
    return best_tree

# Helper functions (not implemented)
def get_features(examples: List[Example]) -> Set[str]:
    """Returns set of all features"""
    pass

def get_max_difference(examples: List[Example]) -> int:
    """Returns maximum difference (δmax) between any positive and negative example"""
    pass

def get_max_domain_size(examples: List[Example]) -> int:
    """Returns maximum domain size (Dmax) over all features"""
    pass

def enumerate_feature_sets(examples: List[Example], size: int, delta_max: int, d_max: int) -> List[Set[str]]:
    """Enumerates candidate feature sets based on parameters"""
    pass

def build_tree_with_features(examples: List[Example], features: Set[str], max_size: int) -> Optional[DecisionTree]:
    """Builds optimal decision tree using only given features with size constraint"""
    pass

def build_tree_with_features_depth(examples: List[Example], features: Set[str], max_depth: int) -> Optional[DecisionTree]:
    """Builds optimal decision tree using only given features with depth constraint"""
    pass

def tree_size(tree: DecisionTree) -> int:
    """Returns size (number of internal nodes) of decision tree"""
    pass

def tree_depth(tree: DecisionTree) -> int:
    """Returns depth of decision tree"""
    pass

```



```python
def mindt(E, s, dmax, delta_max):
    """
    Algorithm MinDT: Find minimal decision tree
    Args:
        E: Classification instance (set of examples)
        s: Maximum size of decision tree
        dmax: Maximum domain size
        delta_max: Maximum difference between examples
    Returns:
        Decision tree or None if no solution exists
    """
    # find all minimal support sets up to size s
    support_sets = enumerate_support_sets(E, s, delta_max)
    
    min_tree = None # 'B' variable in pseudocode
    min_size = float('inf')
    
    # Try each support set
    for S in support_sets:
        # Try to find decision tree using features from S
        tree = find_tree_with_features(E, S, s)
        
        if tree is not None and tree.size < min_size:
            min_tree = tree
            min_size = tree.size
            
    return min_tree

def enumerate_support_sets(E, k, delta_max):
    """
    Enumerates all minimal support sets up to size k
    Based on Corollary 9 from the paper

    Args:
        E: Classification instance
        k: Maximum size of support sets
        delta_max: Maximum difference between examples
    Returns:
        List of minimal support sets
    """
    # implementation would use hitting set enumeration
    # runtime: O(delta_max^k * |E|)
    
    # get disagreement features between positive and negative examples
    disagreements = []
    for pos in E.positive_examples:
        for neg in E.negative_examples:
            # get disagreeing features
            diff_features = {f for f in pos.features if pos[f] != neg[f]}
            if len(diff_features) <= delta_max:
                disagreements.append(diff_features)

    # use hitting set enumeration to find minimal support sets
    # each support set must hit all disagreement sets
    support_sets = []

    def is_hitting_set(features, disagreements):
        return all(any(f in dset for f in features) for dset in disagreements)
      
    def is_minimal(features, disagreements):
        return not any(is_hitting_set(features - {f}, disagreements) for f in features)

    for size in range(1, k+1):
        for feature_set in combinations(E.features, size):
            if is_hitting_set(feature_set, disagreements): # check if hits all disagreement sets
                if is_minimal(feature_set, disagreements): # check if is minimal hitting set
                    support_sets.append(feature_set)

    return support_sets




def mindts(E, s, dmax, delta_max):
    """
    Algorithm MinDTS: Find minimal decision tree with size bound
    Args:
        E: Classification instance (set of examples)
        s: Maximum size of decision tree
        dmax: Maximum domain size
        delta_max: Maximum difference between examples
    Returns:
        Decision tree or None if no solution exists
    """
    # first try MinDT
    tree = mindt(E, s, dmax, delta_max)
    
    # check if solution meets size bound
    if tree is not None and tree.size <= s:
        return tree
        
    return None
```
