# Vertiport siting with split deliveries and service commitments

Let $y_i$ indicate whether site $i$ opens and $z_c$ whether the operator accepts the entire order of zone $c$. Both are binary. Let $f_{ci}$ be the nonnegative continuous volume that site $i$ dispatches to zone $c$. Define this variable only on last-mile links of at most 10 km.

For order quantities $q_c$, handling capacities $u_i$, safety scores $s_i$ and noise scores $n_i$, minimize

$$
\sum_c q_c(1-z_c) + 80\sum_i (s_i+n_i)y_i.
$$

Open exactly three sites, $\sum_i y_i=3$, and fix $y_i=0$ for any candidate farther than 30 km from the warehouse. For each pair outside the closed interval $[2,10]$ km, impose $y_i+y_j\le 1$. Order acceptance and site capacity impose the following balances over eligible links.

$$
\sum_i f_{ci}=q_c z_c \quad(\text{each zone }c),\qquad
\sum_c f_{ci}\le u_i y_i \quad(\text{each site }i).
$$

## Optimal sites and dispatch

Three of the ten triples meet the separation rule. All volumes are daily packages and costs are penalty points.

| Open sites | Maximum volume in accepted orders | Unserved packages | Safety and noise | Minimum penalty |
| --- | ---: | ---: | ---: | ---: |
| S1, S2, S3 | 240 | 150 | 136 | **286** |
| S1, S3, S5 | 130 | 260 | 160 | 420 |
| S2, S3, S4 | 260 | 130 | 168 | 298 |

With S1, S2 and S3 open, only S3 can reach C3 and C5. Its 30 slots cannot fill either order (60 and 40 packages). The other four zones order 290 packages between them. The three sites have 260 slots, so at least one more order must be declined. Declining the smallest, C1 (50), leaves C2, C4 and C6 with a combined order of 240 packages.

S2 handles C2 (80) and C4 (70). S1 and S3 can deliver 70 and 20 packages to C6, respectively. Other optimal splits are possible. S2 has only ten slots left, so C6's 90 packages must use both S1 and S3. S3 is exactly 10 km from C6. Opening S4 in place of S1 would serve 20 more packages and add 32 points in site charges.

## Independent verification

For each legal triple $S$, enumerate all 64 subsets $Q$ of the six orders. Write $N_S(c)$ for the open sites that can serve customer $c$. An accepted set is feasible exactly when every subset $T$ of the open sites satisfies

$$
\sum_{c\in Q:\,N_S(c)\subseteq T} q_c \le \sum_{i\in T} u_i .
$$

Orders on the left can use only capacity in $T$. These inequalities cover every cut of the customer-to-site flow network. By max-flow/min-cut they also suffice to route all accepted demand. Including the empty set rules out accepting a zone that has no eligible site. This enumeration gives the best service volume for each row of the table. The executable Gurobi reference independently returns the optimum and a dispatch plan. [Reference program](reference.py)

## Effect of changing one modeling rule

Each row below optimizes the same inputs after changing one rule. The default grading rule accepts an absolute deviation of at most 0.1 penalty units from the reference optimum of 286.

| Change to the formulation | Optimal penalty | Absolute error (penalty units) |
| --- | ---: | ---: |
| Assign each accepted order to one site | 316 | 30 |
| Give credit for partially served orders | 266 | 20 |
| Remove handling capacities at open sites | 136 | 150 |
| Allow closed sites to dispatch at their stated capacities | 186 | 100 |
| Require each open site to have one neighbor within 10 km | 250 | 36 |
| Allow at most three open sites | 258 | 28 |
| Omit safety and noise costs | 130 | 156 |
| Use a strict 10 km upper bound for intersite separation | 298 | 12 |
| Omit the intersite distance upper bound | 250 | 36 |

Imposing $f_{ci}=q_c a_{ci}$ with binary $a_{ci}$ forces single-site service. One optimal plan under this restriction serves C1 through S1 and C4 and C6 through S2. It fulfills 210 packages. Allowing C6 to split frees capacity at S2 for C2. The operator can exchange C1's 50-package order for C2's 80, reducing the penalty by 30.

Relaxing order acceptance to $0\le z_c\le 1$ has the opposite effect. The 240-package dispatch above leaves ten slots at S2 and ten at S3. Sending 20 packages to C1 fills those slots and gives credit for 20 of its 50 packages, which violates the service commitment. A separate ambiguity concerns geometry. The one-neighbor rule selects S1, S2 and S4. S2 is within 10 km of both outer sites. S1 and S4 are 12.37 km apart, exceeding the pairwise limit.

Run `python tools/laflor_048.py` from the repository root to recompute the comparisons from the site and customer tables. The test suite also solves all ten formulations using SciPy/HiGHS, both with free site selection and with each candidate site set fixed. [Comparison code](../../../tools/laflor_048.py) · [HiGHS cross-check](../../../tests/test_laflor_048.py)
