# Freight routing with shared corridor capacities

A depot S supplies 10 units of freight to D1 and 10 to D2 during one operating period. A planner decides which one-way air corridors to open. The operator may split a destination's freight across several routes and may forward freight through D1. Every unit must reach its destination during the period.

| Corridor | Length (km) | Capacity (freight units per period) |
| --- | ---: | ---: |
| S → D1 | 10.00 | 16 |
| S → D2 | 14.14 | 6 |
| D1 → D2 | 10.00 | 8 |

Use the lengths in the table as exact inputs. Opening a corridor incurs a fixed cost equal to its length. Its capacity is shared by all freight using it, including freight in transit. A closed corridor carries no freight.

Minimize the sum of the opening costs plus 0.01 times the total freight-distance. One unit traveling one kilometer contributes one unit of freight-distance on every leg it uses. Determine which corridors to open, how much freight each carries and the minimum cost.
