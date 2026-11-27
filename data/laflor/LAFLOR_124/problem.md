# Charging cargo-aircraft packs under a shared demand tariff

A cargo terminal runs a North bay and a South bay behind one electricity meter. Each bay handles a different pack type and has two charging sockets. At the opening of each hour, crews exchange one full pack for one depleted pack for every scheduled swap in their bay. All swaps are compulsory and occur before new charging cycles start. Packs stay in their own bay.

| Bay | Packs on hand | Initially full | Minimum full after hour 7 |
| --- | ---: | ---: | ---: |
| North | 4 | 3 | 2 |
| South | 3 | 2 | 2 |

| Hour | North swaps | South swaps | Other metered load (kW) | Energy price (cents/kWh) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 2 | 16 |
| 2 | 1 | 0 | 5 | 8 |
| 3 | 1 | 1 | 1 | 48 |
| 4 | 0 | 1 | 3 | 23 |
| 5 | 2 | 0 | 4 | 11 |
| 6 | 0 | 1 | 1 | 29 |
| 7 | 1 | 1 | 3 | 15 |

Initially all sockets are idle and the other packs are depleted. A newly returned pack can start charging in the hour of its swap. A cycle occupies one socket and draws exactly 2 kW for two consecutive one-hour blocks. A pack started in hour 1, for example, is ready before the swaps at the opening of hour 3. Every cycle must finish by the end of hour 7. The grid connection permits at most 12 kW of total load in any hour.

The terminal pays the hourly energy price on **charging consumption** and a single demand fee of **70 cents per kW** on the highest **total metered load** across all seven hours. The other metered load contributes to the peak. Its energy cost is settled separately.

Determine when each bay should start each charging cycle to meet all swaps and final reserves. Report each bay's hourly starts, the metered peak and the minimum bill in **cents**.
