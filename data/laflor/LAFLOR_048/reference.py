"""Choose three compatible vertiports and allocate each covered zone's demand."""

import json
from itertools import combinations
from math import dist

import gurobipy as gp
from gurobipy import GRB


def solve():
    warehouse = (5, 5)
    sites = {1: (15, 17), 2: (25, 17), 3: (20, 25), 4: (27, 20), 5: (10, 25)}
    capacity = {1: 70, 2: 160, 3: 30, 4: 70, 5: 70}
    scores = {1: (0.3, 0.2), 2: (0.5, 0.1), 3: (0.2, 0.4), 4: (0.6, 0.3), 5: (0.4, 0.5)}
    zones = {1: (18, 18), 2: (28, 12), 3: (22, 28), 4: (32, 22), 5: (12, 28), 6: (20, 15)}
    demand = {1: 50, 2: 80, 3: 60, 4: 70, 5: 40, 6: 90}
    routes = [(c, i) for c in zones for i in sites if dist(zones[c], sites[i]) <= 10]

    with gp.Model("vertiport_siting") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        opened = model.addVars(sites, vtype=GRB.BINARY, name="open")
        covered = model.addVars(zones, vtype=GRB.BINARY, name="covered")
        packages = model.addVars(routes, lb=0, vtype=GRB.CONTINUOUS, name="packages")

        model.addConstr(opened.sum() == 3)
        for i, j in combinations(sites, 2):
            if not 2 <= dist(sites[i], sites[j]) <= 10:
                model.addConstr(opened[i] + opened[j] <= 1)
        for i, point in sites.items():
            if dist(warehouse, point) > 30:
                opened[i].UB = 0
            model.addConstr(packages.sum("*", i) <= capacity[i] * opened[i])
        for c in zones:
            model.addConstr(packages.sum(c, "*") == demand[c] * covered[c])

        unserved = gp.quicksum(q * (1 - covered[c]) for c, q in demand.items())
        site_penalty = gp.quicksum(80 * sum(scores[i]) * opened[i] for i in sites)
        model.setObjective(unserved + site_penalty, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Vertiport siting ended with solver status {model.Status}")

        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "open_sites": [i for i in sites if opened[i].X > 0.5],
            "accepted_zones": [c for c in zones if covered[c].X > 0.5],
            "unserved_packages": unserved.getValue(),
            "site_penalty": site_penalty.getValue(),
            "deliveries": [
                {"zone": c, "site": i, "packages": amount.X}
                for (c, i), amount in packages.items() if amount.X > 1e-8
            ],
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
