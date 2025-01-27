*motivation*

- small decision trees = fewer tests, easier to interpret, generalize better - but expensive to train (np-hard)
- parametrized complexity = input as problem parameters that provide runtime guarantees

*problem*

- DTS/DTD minimum decision tree size/depth
- decision problem, but our algorithms also return the solution tree

*preliminaries*

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


assumption for complexity analysis: $|E| = |E| \cdot (|feat(E)| + 1) \cdot \log D_{max}$

- $T$ - decision tree = consists of vertices and arcs/edges $T = (V, A)$, each node has a feature and threshold assigned to it (left child gets $\leq$ values)



*observation 1*

- 💡 trees use some support set $S$ for their test nodes




# hardness results

theorem 2 reduces the hitting set problem HS to DTS/DTD and shows that $\{sol,  D_\max\}$ alone do not yield FTP tractability, even if all the features are booleans.

however, when the hitting set has a bounded size, the problem becomes FTP tractable. this is equivalent to $\delta_\max$ in our original problem and is naturally small for most datasets (see table 1). this approach to finding parameters is called "deconstruction of intractability".

theorem 8 confirms this finding by laying out the entire algorithm.

*complexity landscape*

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
- paraNP hard = $\emptyset,~ \{\delta_\max\},~ \{D_\max\},~ \{\delta_\max, D_\max\}$
	- problem remains NP-hard even when the parameters are fixed to a constant

*threorem 2*

- 💡 $\{sol,  D_\max\}$ alone don't yield FTP tractability, even if all the features are booleans.
- this is proven by reducing the hitting set HS problem to ours









---



# hardness results

this chapter just serve to explain why specifically the 3 parameters were chosen

- FTP tractable = $\{sol, \delta_\max, D_\max\}$ → author's assume $D_\max$ might be redundant
- XP tractable = $\{sol\},~ \{sol, \delta_\max\},~ \{sol,  D_\max\}$
- w[2] tractable = $\{sol\},~ \{sol,  D_\max\}$
- paraNP hard = $\emptyset,~ \{\delta_\max\},~ \{D_\max\},~ \{\delta_\max, D_\max\}$
- we can assume problem parameters to be small: $sol, D_{\max}, \delta_{\max}$ (see table 1)

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
	- finding the smallest support set (MSS problem) is np-hard

*observation 1*

- 💡✨ the features of a decision tree $T$ are the "support set" of $E$
- = any decision tree's features must form a support set (seperate positive from negative examples)
- $\text{feat}(T) = \{\text{feat}(v) \mid v \in V(T)\}$ = all features used for tests
- proof by contradiction:
	- assuming two examples with different labels would agree on any of the features / assign the same value to it, then they would wrongly end up in the same leaf (which is impossible)

*theorem 2: sol and Dmax alone do not yield ftp, even if features are boolean*

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
