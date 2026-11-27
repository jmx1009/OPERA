# Vertiport siting with daily ground-handling limits

A warehouse at (5, 5) dispatches packages to three vertiports each day. A ground crew at each open site delivers to nearby customer zones. Select exactly three of the five candidate sites. All five have sufficient ground space and unrestricted airspace. Coordinates are in kilometers, and all distance limits are inclusive Euclidean distances.

| Site | Coordinates | Daily capacity (packages) | Safety score | Noise score |
| --- | --- | ---: | ---: | ---: |
| S1 | (15, 15) | 1152 | 0.3 | 0.2 |
| S2 | (25, 10) | 4608 | 0.5 | 0.1 |
| S3 | (20, 25) | 288 | 0.2 | 0.4 |
| S4 | (30, 20) | 120 | 0.6 | 0.3 |
| S5 | (10, 25) | 1152 | 0.4 | 0.5 |

The daily capacity includes both pad and ground-handling limits. At S4 the ground crew can dispatch at most 120 packages per day. Each open site must be within 30 km of the warehouse. Every pair of open sites must be at least 2 km and at most 16 km apart.

| Customer zone | Coordinates | Daily packages |
| --- | --- | ---: |
| C1 | (18, 18) | 50 |
| C2 | (28, 12) | 80 |
| C3 | (22, 28) | 60 |
| C4 | (32, 22) | 70 |
| C5 | (12, 28) | 40 |
| C6 | (20, 15) | 90 |

Serve a zone's entire daily demand from one open site within 10 km, or leave the zone unserved. A site's total assigned packages must respect its daily capacity. Each unserved package incurs a penalty of 1,000. Opening a site incurs a penalty equal to 10 times the sum of its safety and noise scores.

Minimize the sum of these penalties. Identify the three sites, give a feasible zone assignment, and report the minimum penalty.
