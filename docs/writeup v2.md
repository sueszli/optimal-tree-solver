*motivation*

- small decision trees = fewer tests, easier to interpret, generalize better - but expensive to train (np-hard)
- parametrized complexity = input as problem parameters that provide runtime guarantees

*problem*

- DTS/DTD minimum decision tree size/depth
- decision problem, but our algorithms also return the solution tree

*preliminaries*

- $sol$ - solution size = can be $s, d$
- $s$ - size = count of non-leaf nodes
- $d$ - depth = longest root-to-leaf path
- $F,~ \text{feat}(E)$ - features = can have a domain value, finite
- $D$ - domain values = some integers, possibly infinite
- $D_E(f)$ - feature domain values = all mappings to that feature from all examples
	- $D_E(f) = \{ e(f) \mid e \in E\}$
- $D_{\max}$ - maximum domain size = maximum size of $D_E(f)$ for any feature
	- $D_\max = \max_{f \in \text{feat}(E)} |D_E(f)|$
- $E$ - examples, classification instance = mapping of domain values $D$ to features $f$
	- $E^+, E^-$ = positive or negative examples
	- all examples have the same features $\forall e_i, e_j \in E: \text{feat}(e_i) = \text{feat}(e_j)$
	- all examples have a unique label (disjoint/non-overlapping union of labels)
	- uniform examples = all have the same labels
- $E[\alpha]$ - agreeing examples = query of examples with the same values for features as in the mapping $\alpha$
	- $E[\alpha] = \{ e \mid e(f) = \alpha(f) \wedge f \in F'\}$ where $\alpha: F' \mapsto D$
- $\delta(e, e')$ - difference = set of features that two examples have different feature values for / disagree on
	- $\delta(e_1, e_2) = \forall f: e_1(f) \not= e_2(f)$
- $\delta_\max$ - maximum difference = maximum difference between any two examples with different labels
	- $\delta_\max(E) = \max_{e^+\in E^+\wedge e^-\in E^-}|\delta(e^+,e^-)|$
- $S$ - support set = set of features that can distinguish between all positive and negative examples
	- any two examples with different labels must disagree in at least one feature of $S$
	- $\forall e_1 \in E^+, e_2 \in E^-: \exists f \in S: \delta(e_1, e_2) \not = \emptyset$
- $T$ - decision tree = unbalanced binary tree $T = (V, A)$ that partitions decision space
	- has verteces/tests, arcs/edges
	- each inner node $v$ has a feature $\text{feat}(v)$ and threshold value $\lambda(v)$ assigned to it
	- correctly classifies every example
- pseudo tree = balanced binary tree, no features and thresholds, can't correctly classify examples
	- if a valid threshold $\gamma$ can be found for a pseudo tree with assigned features $(T, \alpha)$, then it can be extended to a valid decision tree
- $\alpha: \text{feat}(E) \mapsto v(T)$ - feature assignment function = defines test features $\text{feat}(T)$ in tree
- $\gamma: D_E(\alpha(v)) \mapsto v(T)$ - threshold assignment function = defines test thresholds $\lambda$ in tree

assumption for complexity analysis: $|E| = |E| \cdot (|feat(E)| + 1) \cdot \log D_{max}$

*observation 1*

- 💡 trees use some support set $S$ for their test nodes $\text{feat}(T)$
- proof: if examples with different labels wouldn't disagree on a value, they would end in the same leaf and the tree would be invalid

# hardness results

theorem 2 reduces the hitting set problem HS to DTS/DTD and shows that $\{sol,  D_\max\}$ alone do not yield FTP tractability, even if all the features are booleans.

however, when the hitting set has a bounded size, the problem becomes FTP tractable. this is equivalent to $\delta_\max$ in our original problem and is naturally small for most datasets (see table 1). this approach to finding parameters is called "deconstruction of intractability".

theorem 8 confirms this finding by showing the final algorithm.

for the sake of simplicity most proofs only mention the DTS problem.

*complexity landscape*

- we can assume the problem parameters to be small
- FTP tractable = $\{sol, \delta_\max, D_\max\}$
	- runtime is exponential in a function of the problem params, polynomial in the input size
	- runtime is bounded by $f(k) \cdot n^{O(1)}$, where $f$ is any computable function of parameter $k$ and $n$ is the input size
	- author's assume $D_\max$ might be redundant
	- $sol$ - solution size, can be $s$ or $d$
- XP tractable = $\{sol\},~ \{sol, \delta_\max\},~ \{sol,  D_\max\}$
	- the degree of the polynomial can depend on the parameter
	- runtime is bounded by $n^{f(k)}$, where $f$ is any computable function of parameter $k$ and $n$ is the input size
- w[2] tractable = $\{sol\},~ \{sol,  D_\max\}$
	- problem can be solved by a circuit with $t$ layers of gates with many inputs
	- strong evidence that a problem is not FTP
- paraNP hard = $\emptyset,~ \{\delta_\max\},~ \{D_\max\},~ \{\delta_\max, D_\max\}$
	- problem remains NP-hard even when the parameters are fixed to a constant

*threorem 2*

- 💡 $\{sol,  D_\max\}$ does not yield FTP tractability, even if all the features are booleans – because it's W[2] tractable
- proof: you can reduce the hitting set problem HS to the optimal decision tree problem DTS/DTD
- any valid decision tree must distinguish the single positive example from all negative examples with a single split. this forces it to identify all features that "hit" all the negative examples

hitting set problem:

- in some universe $U$ of elements, given a collection of sets $\mathcal F$ – find a subset $H$ containing at most $k$ elements that intersect with every single set in the collection $\forall F \in \mathcal F: F \cap H \not= \emptyset$
	- $U$ = universe of all possible elements
	- $\mathcal F \subseteq U$ = collection of sets
	- $H \subseteq U$ = hitting set of size of max size $k$
	- $\Delta$ - maximum arity = size of the largest set $F \in \mathcal F$
- reduction: hitting set → decision tree:
	- reduction in polynomial time, preserves parameter $k$ (hitting set size becomes solution size)
	- elements in the universe - are represented as features
	- collection of sets - are represented as examples, using boolean flags to encode whether an element from the universe belongs to that set
	- steps to generate $E(\mathcal I)$ from hitting set instance $\mathcal I$:
		- i. create the empty set $\emptyset$ – set all feature flags to `false`, set label to `positive`
		- ii. create the collection of sets – set feature flag based on belonging, set label to `negative`

*theorem 3*

- 💡 $\{\delta_\max(E)\}$ does not yield FTP tractability, even if all the features are booleans – because it's paraNP-hard tractable
	- proof: $\delta_\max(E(\mathcal I)) = \Delta_\max(\mathcal I)$
	- the highest number of features two examples with a different classification can disagree on, is equivalent to the size of the largest set in the collection
	- the only positive example has all zeros, each negative example has ones exactly in positions corresponding to elements of its set $F$. therefore, the number of disagreements between the only positive and any negative examples equals the size of set $F$
- 💡 $\{\text{min}_{\#}(E)\}$ does not yield FTP tractability – because it's paraNP-hard tractable
	- where: $\text{min}_{\#}(E) = \min\{|E^+|, |E^-|\}$
	- proof: in our reduction $\text{min}_{\#}(E(\mathcal I)) = 1$ because we have a single positive example
- 💡 num of inner nodes does not yield FTP tractability – because it's paraNP-hard tractable
	- proof: due to $\text{min}_{\#}(E(\mathcal I)) = 1$ we have 0 branching nodes, just two leafs







# algorithm

the hardness of decision tree learning comes primarily from feature selection, rather than training
## stage 1: feature selection










## stage 2: training

*theorem 4*

- 💡 one can compute an optimal decision tree that exclusively tests features from a given support set $S$ in time $2^{\mathcal{O}(s^2)}\|E\|^{1+o(1)} \log \|E\|$
- proof: final algorithm

*lemma 5*

- the number of trees we have to consider is defined by $k$ which is bounded by the solution size $k = |S| \leq s$
- this means we can enumerate/brute force all tree structures and feature assignments to nodes, but not all possible thresholds


---





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
- $\text{findTH}$ algorithm: (finds threshhold)
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
