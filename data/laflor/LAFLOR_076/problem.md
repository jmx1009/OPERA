# Charging stations with two independent supply routes

A parcel network has a hub H at (0, 0). Open a set of charging stations and designate some of them as terminals. Each delivery zone must be within 15 km of at least one terminal. The coordinates below are in kilometers. Use unrounded Euclidean distances throughout.

| Station | Coordinates | Delivery zone | Coordinates |
| --- | --- | --- | --- |
| S1 | (15, 10) | D1 | (18, 8) |
| S2 | (25, 5) | D2 | (30, 10) |
| S3 | (20, 25) | D3 | (22, 28) |
| S4 | (35, 20) | D4 | (38, 25) |
| S5 | (40, 35) | D5 | (45, 38) |
| S6 | (10, 30) | D6 | (12, 35) |

A directed flight leg may run from H to a station, or between two different stations, whenever the distance is at most 30 km. Both directions of a qualifying interstation leg are available. Every station on a selected route must open.

For every selected terminal, supply two distinct simple routes from H to that terminal. Its two routes may share the hub and destination only. They must use different directed legs, so a direct hub-to-terminal flight counts as one route. A station or leg may serve routes to several different terminals, and a terminal may relay flights to another terminal.

Let $D$ be the sum of the lengths of all admissible directed legs before any stations are selected. Let $L$ sum the lengths of both routes to each selected terminal. Count a leg each time a route uses it. Coverage from terminals to delivery zones contributes no length to $L$. If $n$ stations open, minimize

$$
0.5\,\frac{L}{2D}+0.5\,\frac{n}{6}.
$$

Report the minimum score, the open stations and the two supply routes for each terminal.
