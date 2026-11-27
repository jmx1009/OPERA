# Skyport selection with a corridor exclusion

Let binary variables $y_i$ indicate open skyports and $x_r$ indicate a served route. Open two ports and impose $y_1+y_3\leq1$. For each route with endpoints $i,j$, require $x_r\le y_i$ and $x_r\le y_j$. All market contributions are positive, so a feasible route between open ports is served at the optimum.

For each mode, compute $V=-0.5T-0.1C+A$ using time $T$, monetary cost $C$ and the stated acceptance factor $A$. The UAM share for route $r$ is $p_r=1/(1+\exp(V^G_r-V^A_r))$. Use full precision in the objective $\max\sum_r D_r p_r x_r$, with demands 2,000, 1,500 and 500. The corridor exclusion rules out the first market. The best remaining pair opens S2 and S3. The expected ridership is 1499.9996600731 passengers, reported as 1500 in the paper.
