# Formulation

Index the bays by $b$ and the seven one-hour blocks by $t$. Let $N_b$ be the pack stock, $d_{bt}$ the swaps, $h_t$ the other load and $p_t$ the charging energy price. The integer $x_{bt}\ge0$ counts cycles started after that hour's swaps. Set $x_{b0}=x_{b7}=0$. Each cycle occupies its socket in its starting hour and the following hour, so $a_{bt}=x_{b,t-1}+x_{bt}\le2$ cycles run in bay $b$ during hour $t$.

Let $F_{bt}$ count full packs before the swaps at the start of hour $t$, with initial values 3 and 2. The following conditions cover pack availability, the depleted packs on the sockets, and completions before the next swap wave.

$$
d_{bt}\le F_{bt}\le N_b,\qquad
F_{bt}-d_{bt}+a_{bt}\le N_b,\qquad
F_{b,t+1}=F_{bt}-d_{bt}+x_{b,t-1}.
$$

At the close of hour 7 require $2\le F_{b8}\le N_b$ in each bay. The hourly meter reads $m_t=h_t+2\sum_b a_{bt}$ in kW. Impose $m_t\le12$ and $P\ge m_t$ for all hours. With one-hour blocks the objective in cents is

$$
\min\ \sum_{t=1}^{7} 2p_t\sum_b a_{bt}+70P.
$$

## Optimal charging plan

| Hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| North starts | 2 | 0 | 0 | 1 | 0 | 2 | 0 |
| South starts | 1 | 0 | 0 | 2 | 0 | 2 | 0 |
| Total meter (kW) | 8 | 11 | 1 | 9 | 10 | 9 | 11 |

The ten cycles use 40 kWh and leave two full packs in each bay. Charging costs 700 cents and the 11 kW peak costs 770 cents. The minimum bill is **1,470 cents**.

An independent check enumerates all hourly starts for each bay and simulates pack exchanges and completions. There are 38 feasible North schedules and 23 feasible South schedules before applying the shared connection limit. Of the 874 pairs, 409 respect the 12 kW limit. The least expensive energy-only schedule costs 680 cents for energy but reaches 12 kW, for a bill of 1,520 cents under the stated tariff.
