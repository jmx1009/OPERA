"""Exhaustive one-sortie search with truck/drone synchronization at rendezvous."""
from functools import lru_cache
from itertools import combinations, permutations
from math import dist, inf
import json

sites = [(0, 0), (4, 10), (8, 2), (12, 0), (10, 9), (16, 3), (20, 1), (6, -2), (14, 7)]
weight = [0, 0.8, 1.0, 0.5, 0.7, 1.2, 1.5, 0.6, 0.9]
distance = [[dist(a, b) for b in sites] for a in sites]


@lru_cache(maxsize=None)
def best_flight(launch, recovery, customers):
    best = (inf, (), 0.0)
    for order in permutations(customers):
        load = sum(weight[i] for i in order)
        energy = time = 0.0
        tail = launch
        for head in (*order, recovery):
            hours = distance[tail][head] / 75
            energy += (80 + 30 * load) * hours
            time += hours
            load -= weight[head] if head in customers else 0
            tail = head
        if energy <= 500 and time < best[0]:
            best = (time, order, energy)
    return best


def solve():
    customers = set(range(1, 9))
    best = {"objective": inf}
    for count in range(4):
        for drone in combinations(sorted(customers - {6}), count):
            if sum(weight[i] for i in drone) > 3 + 1e-10:
                continue
            for tour in permutations(sorted(customers - set(drone))):
                road = (0, *tour, 0)
                arrival = [0.0]
                for a, b in zip(road, road[1:]):
                    arrival.append(arrival[-1] + distance[a][b] / 50)
                if 60 * arrival[-1] >= best["objective"]:
                    continue
                if not drone:
                    best = {"objective": 60*arrival[-1], "truck": road, "drone": (), "flight_wh": 0}
                    continue
                for end in range(1, len(road)):
                    for start in range(end):
                        flight, order, energy = best_flight(road[start], road[end], drone)
                        idle = max(0, flight - (arrival[end]-arrival[start]))
                        minutes = 60*(arrival[-1]+idle)
                        if minutes < best["objective"]:
                            best = {"objective": minutes, "truck": road, "drone": (road[start], *order, road[end]),
                                    "launch_stop": start, "recovery_stop": end, "flight_wh": energy}
    return {"status": "optimal", **best}


if __name__ == "__main__":
    print(json.dumps(solve()))
