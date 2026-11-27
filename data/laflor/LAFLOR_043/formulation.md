# Formulation

Let $I$ be the five clinics and $F$ the listed bookings. Booking $f$ serves clinic $i(f)$ between visits $a_f$ and $b_f$, uses $e_f$ Wh and costs $c_f$ dollars. An outside pickup for clinic $i$ costs $p_i$. Let $x_f$ and $y_i$ be binary decisions to fly booking $f$ and use the outside courier for $i$, respectively. Minimize

$$
\sum_{f\in F}c_f x_f+\sum_{i\in I}p_i y_i,\qquad
y_i+\sum_{f:i(f)=i}x_f=1\quad(i\in I).
$$

For road leg $t=0,\ldots,6$, let $u_t=\sum_{f:a_f\le t<b_f}x_f$ be its aircraft occupancy and impose $u_t\le1$. The half-open intervals allow recovery and launch at the same visit. Write $B_t$ for charge on arrival at visit $t$ and $g_t$ for the amount charged during road leg $t$. Set $B_0=400$ and $0\le B_t\le400$ through visit 7. For each road leg impose

$$
\sum_{f:a_f=t}e_fx_f\le B_t,\qquad
0\le g_t\le150(1-u_t),\qquad
B_{t+1}=B_t-\sum_{f:a_f=t}e_fx_f+g_t.
$$

## Optimal collection plan

Fly P1 for Pine, E2 for Elm, O1 for Oak and V2 for Vale. Their launch and recovery intervals are $[0,1)$, $[3,4)$, $[4,5)$ and $[6,7)$. With maximum available charging, arrival charge at visits 0 through 7 is 400, 200, 350, 400, 250, 50, 200 and 0 Wh. The outside courier collects Reed. The four bookings cost $9+8+8+7=32$ dollars and the courier adds 29, giving a minimum of **61 USD**.

An independent check tries the two bookings and the courier for each clinic, for $3^5=243$ combinations. For each combination it sorts the bookings by launch, checks for overlap and carries the battery forward across the intervening van legs. Taking all available charge during idle legs maximizes charge at every subsequent launch, so this test decides feasibility for the chosen bookings. The least cost over feasible combinations is 61 USD. If battery feasibility is omitted, P1, R1, E2, O1 and V2 cost only 38 USD. At visit 1, R1 requires 250 Wh but P1 leaves just 200 Wh.
