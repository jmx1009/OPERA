# Vertiport siting with split deliveries and service commitments

A parcel operator receives six daily orders at a warehouse at (5, 5). It must open exactly three of the five candidate vertiports listed below. Drones carry packages from the warehouse to an open site, where ground couriers take them to the customer zone. One point of the operating penalty is charged for each unserved package.

Coordinates are in kilometers. Use Euclidean distances and include both endpoints of each range. Every open site must be within 30 km of the warehouse. Every pair of open sites must be at least 2 km and at most 10 km apart. A ground courier may serve a customer zone within 10 km of its site. All five sites have unrestricted airspace and sufficient space for the listed facilities.

Daily handling capacity includes the available pads and ground crew. Opening a site incurs 80 penalty points per unit of its safety score and 80 per unit of its noise score. Both charges apply once per open site, including a site that handles no packages.

| Site | Coordinates | Capacity (packages/day) | Safety score | Noise score |
| --- | --- | ---: | ---: | ---: |
| S1 | (15, 17) | 70 | 0.3 | 0.2 |
| S2 | (25, 17) | 160 | 0.5 | 0.1 |
| S3 | (20, 25) | 30 | 0.2 | 0.4 |
| S4 | (27, 20) | 70 | 0.6 | 0.3 |
| S5 | (10, 25) | 70 | 0.4 | 0.5 |

| Customer zone | Coordinates | Daily order (packages) |
| --- | --- | ---: |
| C1 | (18, 18) | 50 |
| C2 | (28, 12) | 80 |
| C3 | (22, 28) | 60 |
| C4 | (32, 22) | 70 |
| C5 | (12, 28) | 40 |
| C6 | (20, 15) | 90 |

The operator accepts or declines each zone's entire daily order. An accepted order may be divided among eligible open sites in continuous shares. A declined order receives no deliveries, and its full quantity incurs the unserved-package penalty. Each site's total dispatch must fit its own daily capacity.

Minimize the combined unserved-package, safety and noise penalties. Report the three sites, the accepted orders, a feasible dispatch plan, and the minimum penalty.
