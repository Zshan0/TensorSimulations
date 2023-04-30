# Order bra-ket identical to order row-col

Consider density matrix $\rho = \ket{a}\bra{b}$, we can construct this by `a[1] ^ b[1]`. Contracting this will give $\rho$. Let’s consider the following case:

$$
A = 
\begin{bmatrix} 
1 \\ 
2
\end{bmatrix} 

\begin{bmatrix} 
3 & 4
\end{bmatrix} = 

\begin{bmatrix} 
3 & 4 \\
6 & 8
\end{bmatrix}
$$

Setting the row’s index to 1 is equivalent to picking the first element in $\ket{a}$. We can generalize this to any number of qubits and we can say the orders are equivalent.
