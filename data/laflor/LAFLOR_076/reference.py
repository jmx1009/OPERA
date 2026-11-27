"""Choose charging stations and two disjoint paths from the hub to every terminal."""

import json
from math import dist

import gurobipy as gp
from gurobipy import GRB


def solve():
    hub = 0
    location = [(0, 0), (15, 10), (25, 5), (20, 25), (35, 20), (40, 35), (10, 30)]
    zones = [(18, 8), (30, 10), (22, 28), (38, 25), (45, 38), (12, 35)]
    stations = range(1, len(location))
    length = {
        (i, j): dist(location[i], location[j])
        for i in range(len(location))
        for j in stations
        if i != j and dist(location[i], location[j]) <= 30
    }

    with gp.Model("disjoint_charging_routes") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        opened = model.addVars(stations, vtype=GRB.BINARY, name="open")
        terminal = model.addVars(stations, vtype=GRB.BINARY, name="terminal")
        flow = model.addVars(
            [(i, j, t) for i, j in length for t in stations if i != t],
            vtype=GRB.BINARY,
            name="route",
        )
        for point in zones:
            covered_by = [t for t in stations if dist(location[t], point) <= 15]
            model.addConstr(gp.quicksum(terminal[t] for t in covered_by) >= 1)
        for t in stations:
            model.addConstr(terminal[t] <= opened[t])
            model.addConstr(flow.sum(hub, "*", t) == 2 * terminal[t])
            model.addConstr(flow.sum("*", t, t) == 2 * terminal[t])
            for v in stations:
                if v == t:
                    continue
                incoming = flow.sum("*", v, t)
                model.addConstr(incoming == flow.sum(v, "*", t))
                model.addConstr(incoming <= opened[v])
                model.addConstr(incoming <= terminal[t])

        total_length = gp.quicksum(length[i, j] * f for (i, j, _), f in flow.items())
        model.setObjective(
            0.5 * total_length / (2 * sum(length.values()))
            + 0.5 * opened.sum() / len(stations),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Charging model ended with solver status {model.Status}"
            )

        paths = {}
        for t in stations:
            if terminal[t].X < 0.5:
                continue
            arcs = [(i, j) for i, j in length if i != t and flow[i, j, t].X > 0.5]
            successor = {i: j for i, j in arcs if i != hub}
            paths[t] = []
            for start in sorted(j for i, j in arcs if i == hub):
                route = [hub, start]
                while route[-1] != t:
                    route.append(successor[route[-1]])
                paths[t].append(route)
        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "open_sites": [s for s in stations if opened[s].X > 0.5],
            "routes": paths,
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
