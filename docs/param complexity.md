*classical complexity vs. parametrized complexity*

- lets us differentiate between $O(2^k \cdot n)$ vs. $O(n^k)$ → technically both are polynomial, but vastly different
- a "parameter" should be part of the input, most often a number, that you can expect to be small in your very specific use-case
- there is a whole ecosystem of parameterized complexity classes for NP-complete problems
	- FTP = parameter helps
	- …
	- XP = parameter helps very little
	- para-NP = parameter does not help at all

*fpt algorithm*

- FTP = fixed parameter tracable
- runtime = $f(k) \cdot \text{poly}(n)$
- where:
	- $f$ = an arbitrary (computable) function
	- $\text{poly}$ = polynomial function, independent of $k$
