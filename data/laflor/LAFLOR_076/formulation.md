# Station activation with disjoint paths

Construct the directed graph specified by the 30 km range. Define binary variables $y_i$ for opening station $i$ and $z_i$ for using it as a terminal, with $z_i\leq y_i$. For each delivery zone $d$, impose $\sum_{i:\,\operatorname{dist}(i,d)\leq15}z_i\geq1$.

For each candidate terminal $t$, use a binary flow $f_{ijt}$ on each admissible arc $(i,j)$ with $i\ne t$. The hub sends $2z_t$ units, and $t$ receives $2z_t$ units. At every other station $v$, incoming flow equals outgoing flow. Bound that incoming flow by both $y_v$ and $z_t$. The unit bound on intermediate stations prevents the two routes to $t$ from sharing a relay. Binary arc use prevents counting a direct arc twice. A separate flow for each terminal allows reuse across terminals.

Write $d_{ij}$ for the Euclidean arc length and $D=\sum_{(i,j)}d_{ij}$. Minimize

$$
\frac{0.5}{2D}\sum_{(i,j),t}d_{ij}f_{ijt}+\frac{0.5}{6}\sum_i y_i.
$$

All arc lengths are positive, so an optimal flow has no cycles and decomposes into two simple paths per terminal. An independent enumeration checks every station set, each subset of terminals that covers all zones, and all pairs of internally disjoint simple paths within the open set.

An optimal design opens S1 through S5 and uses S1, S3 and S5 as terminals.

| Terminal | First route | Second route |
| --- | --- | --- |
| S1 | H → S1 | H → S2 → S1 |
| S3 | H → S1 → S3 | H → S2 → S3 |
| S5 | H → S1 → S3 → S5 | H → S2 → S4 → S5 |

These routes give $L=250.1870309063$ km. With $D=503.2340546871$ km and five open stations, the minimum score is $0.5409562634$.
