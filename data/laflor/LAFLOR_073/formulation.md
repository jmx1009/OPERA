# Capacitated corridor design

For each directed corridor $e$, let $d_e$ and $u_e$ be its length and capacity. Open it with binary variable $y_e$. Let $f_{ek}\geq 0$ be the flow on that corridor ultimately bound for $k\in\{\mathrm{D1},\mathrm{D2}\}$. The objective is

$$
\min \sum_e d_e y_e + 0.01\sum_{e,k} d_e f_{ek}.
$$

For each destination $k$, outgoing flow minus incoming flow equals 10 at S, $-10$ at $k$ and zero at the other node. Every corridor obeys $\sum_k f_{ek}\leq u_e y_e$.

Let $t$ be the amount for D2 routed through D1. The loads on S → D1, S → D2 and D1 → D2 are $10+t$, $10-t$ and $t$. The three capacities give $4\leq t\leq6$, so all three corridors must open. A unit traveling via D1 flies 20 km and a unit delivered directly to D2 flies 14.14 km. The least-cost split is therefore $t=4$. The total freight-distance is $14(10)+6(14.14)+4(10)=264.84$, giving an objective of $34.14+0.01(264.84)=36.7884$.
