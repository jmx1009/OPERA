"""Schedule two-hour pack charging cycles and price the shared metered peak."""

import json

import gurobipy as gp
from gurobipy import GRB


SWAPS = {"North": (1, 1, 1, 0, 2, 0, 1), "South": (1, 0, 1, 1, 0, 1, 1)}
PACKS = {"North": 4, "South": 3}
INITIALLY_FULL = {"North": 3, "South": 2}
BASE_KW = (2, 5, 1, 3, 4, 1, 3)
PRICE = (16, 8, 48, 23, 11, 29, 15)


def solve():
    hours = range(len(PRICE))
    with gp.Model("cargo_pack_charging") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        starts = model.addVars(SWAPS.keys(), hours, vtype=GRB.INTEGER, ub=2, name="starts")
        full = model.addVars(SWAPS.keys(), range(len(PRICE) + 1), vtype=GRB.INTEGER, name="full")
        peak = model.addVar(ub=12, name="peak_kw")
        charging_kw = {}
        for bay, demand in SWAPS.items():
            model.addConstr(full[bay, 0] == INITIALLY_FULL[bay])
            model.addConstr(starts[bay, hours[-1]] == 0)
            for t in hours:
                finishing = starts[bay, t - 1] if t else 0
                active = finishing + starts[bay, t]
                model.addConstr(active <= 2)
                after_swaps = full[bay, t] - demand[t]
                model.addConstr(after_swaps >= 0)
                model.addConstr(after_swaps + active <= PACKS[bay])
                model.addConstr(full[bay, t + 1] == after_swaps + finishing)
                charging_kw[bay, t] = 2 * active
            model.addConstr(full[bay, len(PRICE)] >= 2)

        loads = [gp.quicksum(charging_kw[bay, t] for bay in SWAPS) for t in hours]
        for t in hours:
            model.addConstr(peak >= BASE_KW[t] + loads[t])
        energy_cost = gp.quicksum(PRICE[t] * loads[t] for t in hours)
        model.setObjective(energy_cost + 70 * peak, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("The pack schedule did not solve to optimality")
        return {
            "status": "optimal",
            "objective": round(model.ObjVal),
            "starts": {bay: [round(starts[bay, t].X) for t in hours] for bay in SWAPS},
            "peak_kw": round(peak.X),
            "energy_cents": round(energy_cost.getValue()),
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
