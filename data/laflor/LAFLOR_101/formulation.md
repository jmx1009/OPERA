# Siting and whole-zone assignment

Let $y_i$ indicate that site $i$ opens. For each customer zone $c$, let $q_c$ be its daily demand. Create a binary assignment $a_{ci}$ for every site within 10 km of $c$. Require exactly three open sites, set $y_i=0$ for sites more than 30 km from the warehouse and impose $y_i+y_j\leq1$ for every pair outside the interval $[2,16]$ km.

Each zone can use at most one site. Enforce $a_{ci}\leq y_i$ and $\sum_c q_c a_{ci}\leq u_i y_i$, where $u_i$ is the site's daily handling capacity. With safety and noise scores $s_i$ and $n_i$, minimize

$$
1000\sum_c q_c\left(1-\sum_i a_{ci}\right)+10\sum_i(s_i+n_i)y_i.
$$

Five of the ten site triples meet the distance limits. The triples $\{1,2,3\}$, $\{1,2,4\}$ and $\{1,3,5\}$ each leave a customer beyond last-mile range. With $\{1,3,4\}$, C2 and C4 can use only S4. Their combined 150 packages exceed its capacity of 120. With $\{2,3,4\}$, assigning C1 through C6 to sites $(3,2,3,4,3,2)$ serves all 390 packages. The penalty is $10(0.6+0.6+0.9)=21$. Every other triple incurs at least one unserved-package penalty, proving optimality.
