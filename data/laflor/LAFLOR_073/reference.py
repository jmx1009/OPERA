"""Solve the freight corridor design and report aggregate loads on the open links."""

import json

import gurobipy as gp
from gurobipy import GRB


def solve():
    nodes = ("S", "D1", "D2")
    demands = {"D1": 10, "D2": 10}
    # Each entry gives the exact length and the shared capacity.
    corridors = {("S", "D1"): (10, 16), ("S", "D2"): (14.14, 6), ("D1", "D2"): (10, 8)}

    with gp.Model("capacitated_freight") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        opened = model.addVars(corridors, vtype=GRB.BINARY, name="open")
        flow = model.addVars(corridors, demands, name="flow")

        for (i, j), (_, capacity) in corridors.items():
            model.addConstr(flow.sum(i, j, "*") <= capacity * opened[i, j])
        for dest, demand in demands.items():
            supply = {"S": demand, dest: -demand}
            for node in nodes:
                model.addConstr(
                    flow.sum(node, "*", dest) - flow.sum("*", node, dest)
                    == supply.get(node, 0)
                )

        model.setObjective(
            gp.quicksum(
                length * (opened[i, j] + 0.01 * flow.sum(i, j, "*"))
                for (i, j), (length, _) in corridors.items()
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Freight model ended with solver status {model.Status}")
        return {
            "status": "optimal",
            "objective": model.ObjVal,
            "arc_loads": {
                f"{i}-{j}": flow.sum(i, j, "*").getValue()
                for i, j in corridors
                if opened[i, j].X > 0.5
            },
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
