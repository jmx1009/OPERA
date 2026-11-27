"""Enumerate the open ports and ground/air choices for the three markets."""
from itertools import combinations, product
from math import dist, inf
import json

points = [(0, 0), (10, 10), (0, 10), (10, 0)]
markets = [(0, 1, 100), (2, 3, 100), (0, 2, 50)]


def solve():
    best = {"objective": inf}
    for n in range(5):
        for ports in combinations(range(4), n):
            for modes in product((0, 1), repeat=3):
                cost = 500*n
                flows = {}
                for (a, b, volume), air in zip(markets, modes):
                    if not air:
                        cost += volume * dist(points[a], points[b]) / 0.5
                        continue
                    if not ports:
                        break
                    i, j = [min(ports, key=lambda k: (dist(points[v], points[k]), k)) for v in (a,b)]
                    if i == j:
                        break
                    access = dist(points[a], points[i]) + dist(points[j], points[b])
                    cost += volume * (access/0.5 + dist(points[i], points[j])/2)
                    arc = tuple(sorted((i, j)))
                    flows[arc] = flows.get(arc, 0) + volume
                else:
                    cost += 0.5 * flows.get((0, 1), 0) * flows.get((2, 3), 0)
                    if cost < best["objective"]:
                        best = {"objective": cost, "ports": ports, "air_by_market": modes}
    return {"status": "optimal", **best}


if __name__ == "__main__":
    print(json.dumps(solve()))
