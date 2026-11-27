# Multi-Visit Drone Routing Problem for Pickup and Delivery (MDRP-PD) - Simplified Scenario

## Background
A logistics company operates a combined truck-drone delivery system for simultaneous pickup and delivery services. Each truck is equipped with one drone. The drone can serve multiple customers per flight (multi-visit), carrying both delivery and pickup parcels. The objective is to minimize total cost: fixed deployment cost + truck transportation cost + drone energy cost.

## Problem Description
The company must determine:
1. Which truck-drone pairs to deploy
2. The truck route for each deployed pair
3. The drone flights (launch node, sequence of customers, retrieval node) for each pair

**Additional Constraint (Drone Service Mandate):** Due to the remote rooftop access at Customer 3's location, Customer 3 (pickup only, 1.2 kg) must be served by the drone rather than the truck. The building lacks ground-level loading facilities, making truck delivery infeasible for this customer.

## Simplified Scenario

We have **1 truck-drone pair** (K=1), **1 depot** (node 0 = start, node 4 = end depot), and **3 customers** (nodes 1, 2, 3).

- **Customer 1**: delivery demand d₁ = 1.0 kg, pickup demand p₁ = 0.5 kg
- **Customer 2**: delivery demand d₂ = 0.8 kg, pickup demand p₂ = 0.0 kg (delivery only)
- **Customer 3**: delivery demand d₃ = 0.0 kg, pickup demand p₃ = 1.2 kg (pickup only) — **must be served by drone**

All customers are drone-eligible (N_d = {1, 2, 3}).

## Parameters

### Truck Parameters
- Maximum load capacity: Q = 10 kg
- Travel speed: v_T = 30 km/h
- Unit travel cost: c_T = $0.78/km
- Service time per customer (truck): u₁ = 1/12 h, u₂ = 1/30 h, u₃ = 1/20 h
- Fixed deployment cost: c₀ = $22

### Drone Parameters
- Maximum load capacity: W = 3 kg
- Curb weight: w₀ = 6 kg
- Flight-energy coefficient: α = 3.5 Wh/(kg·km)
- Maximum battery capacity: E = 504 Wh
- Output power: P = 1008 W
- Unit energy cost: c_F = $0.00248/Wh
- Battery replacement time: BR = 1/60 h
- Service time per customer (drone): u'₁ = 1/12 h, u'₂ = 1/60 h, u'₃ = 1/30 h

### Node Coordinates (km)
| Node | X    | Y    | Description     |
|------|------|------|-----------------|
| 0    | 0.0  | 0.0  | Starting depot  |
| 1    | 2.0  | 3.0  | Customer 1      |
| 2    | 5.0  | 1.0  | Customer 2      |
| 3    | 4.0  | 5.0  | Customer 3      |
| 4    | 0.0  | 0.0  | Ending depot    |

### Distance Matrix (Euclidean, km; displayed values rounded)
|       | 0    | 1    | 2    | 3    | 4    |
|-------|------|------|------|------|------|
| 0     | 0    | 3.61 | 5.10 | 6.40 | 0    |
| 1     | 3.61 | 0    | 3.61 | 2.83 | 3.61 |
| 2     | 5.10 | 3.61 | 0    | 4.12 | 5.10 |
| 3     | 6.40 | 2.83 | 4.12 | 0    | 6.40 |
| 4     | 0    | 3.61 | 5.10 | 6.40 | 0    |

Both vehicles use Euclidean distances calculated directly from the coordinates at full precision; the rounded table is for display only.

### Payload, Energy, and Synchronization
On each drone sortie, delivery parcels are loaded before departure. At a served customer, the drone unloads that customer's delivery and then collects the pickup. Let L_i be the payload after service at node i, or the departure payload at a launch node. On a selected flight leg (i,j), the energy is exactly e_ij = α(w₀ + L_i)r_ij, where r_ij is the Euclidean distance. An unselected flight leg consumes zero energy. Payload must remain between 0 and W, and the sum of flight-leg energies in each sortie must not exceed E. A drone carried by the truck does not consume flight energy.

Only flight energy is charged and counted against the battery limit. Customer service, waiting, and battery replacement take time but consume no battery energy in this simplified instance. Flight time is e_ij/P hours. Launch and recovery must be synchronized with the truck, which may remain at the depot and wait for the drone. There is no delivery deadline or route-duration limit. Nodes 0 and 4 represent the same physical depot.

## Decision
Determine the truck route and drone flight assignments to minimize:
- Fixed cost of deploying the truck-drone pair
- Truck transportation cost (proportional to distance)
- Drone flight-energy cost (proportional to distance × the sum of curb weight and actual payload)

Subject to:
- Each customer served exactly once (by truck or drone)
- Truck load capacity ≤ Q at all times
- Drone load capacity ≤ W at all times
- Drone battery energy ≤ E per flight
- Time synchronization between truck and drone
- **Customer 3 must be served by drone** (drone service mandate)
