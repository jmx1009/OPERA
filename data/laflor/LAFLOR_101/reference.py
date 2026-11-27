"""Open three vertiports and assign whole customer zones within daily site capacity."""

import json
from itertools import combinations
from math import dist

import gurobipy as gp
from gurobipy import GRB


def solve():
    warehouse = (5, 5)
    sites = {1: (15, 15), 2: (25, 10), 3: (20, 25), 4: (30, 20), 5: (10, 25)}
    capacity = {1: 1152, 2: 4608, 3: 288, 4: 120, 5: 1152}
    safety = {1: 0.3, 2: 0.5, 3: 0.2, 4: 0.6, 5: 0.4}
    noise = {1: 0.2, 2: 0.1, 3: 0.4, 4: 0.3, 5: 0.5}
    zones = [(18, 18), (28, 12), (22, 28), (32, 22), (12, 28), (20, 15)]
    demand = [50, 80, 60, 70, 40, 90]

    with gp.Model("vertiport_handling") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        opened = model.addVars(sites, vtype=GRB.BINARY, name="open")
        assignment = model.addVars(
            [
                (c, i)
                for c, point in enumerate(zones)
                for i in sites
                if dist(point, sites[i]) <= 10
            ],
            vtype=GRB.BINARY,
            name="assign",
        )
        model.addConstr(opened.sum() == 3)
        for i, point in sites.items():
            if dist(warehouse, point) > 30:
                opened[i].UB = 0
            model.addConstr(
                gp.quicksum(demand[c] * a for (c, k), a in assignment.items() if k == i)
                <= capacity[i] * opened[i]
            )
        for i, j in combinations(sites, 2):
            if not 2 <= dist(sites[i], sites[j]) <= 16:
                model.addConstr(opened[i] + opened[j] <= 1)
        for (_, i), a in assignment.items():
            model.addConstr(a <= opened[i])
        for c in range(len(zones)):
            model.addConstr(assignment.sum(c, "*") <= 1)

        missed = gp.quicksum(
            q * (1 - assignment.sum(c, "*")) for c, q in enumerate(demand)
        )
        site_penalty = gp.quicksum(
            10 * (safety[i] + noise[i]) * opened[i] for i in sites
        )
        model.setObjective(1000 * missed + site_penalty, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Vertiport model ended with solver status {model.Status}"
            )

        assigned = [None] * len(zones)
        for (c, i), decision in assignment.items():
            if decision.X > 0.5:
                assigned[c] = i
        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "open_sites": [i for i in sites if opened[i].X > 0.5],
            "assigned_site_by_zone": assigned,
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
