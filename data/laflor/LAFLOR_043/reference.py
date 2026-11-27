"""Assign clinic pickups to timetable bookings or an outside courier."""

import json
from typing import NamedTuple

import gurobipy as gp
from gurobipy import GRB


class Booking(NamedTuple):
    clinic: str
    launch: int
    recovery: int
    energy: int
    fee: int


COURIER = {"Elm": 32, "Pine": 36, "Reed": 29, "Oak": 38, "Vale": 37}
BOOKINGS = {
    "E1": Booking("Elm", 0, 2, 200, 7),
    "E2": Booking("Elm", 3, 4, 150, 8),
    "P1": Booking("Pine", 0, 1, 200, 9),
    "P2": Booking("Pine", 2, 4, 200, 6),
    "R1": Booking("Reed", 1, 3, 250, 6),
    "R2": Booking("Reed", 3, 5, 150, 9),
    "O1": Booking("Oak", 4, 5, 200, 8),
    "O2": Booking("Oak", 5, 7, 250, 7),
    "V1": Booking("Vale", 4, 6, 200, 6),
    "V2": Booking("Vale", 6, 7, 200, 7),
}


def solve():
    with gp.Model("specimen_pickups") as model:
        model.Params.OutputFlag = 0
        model.Params.MIPGap = 0
        fly = model.addVars(BOOKINGS.keys(), vtype=GRB.BINARY, name="fly")
        courier = model.addVars(COURIER.keys(), vtype=GRB.BINARY, name="courier")
        battery = model.addVars(8, ub=400, name="battery")
        recharge = model.addVars(7, ub=150, name="recharge")
        model.addConstr(battery[0] == 400)
        for clinic in COURIER:
            assigned = gp.quicksum(fly[f] for f, b in BOOKINGS.items() if b.clinic == clinic)
            model.addConstr(assigned + courier[clinic] == 1)
        for leg in range(7):
            busy = gp.quicksum(fly[f] for f, b in BOOKINGS.items() if b.launch <= leg < b.recovery)
            draw = gp.quicksum(b.energy * fly[f] for f, b in BOOKINGS.items() if b.launch == leg)
            model.addConstr(busy <= 1)
            model.addConstr(draw <= battery[leg])
            model.addConstr(recharge[leg] <= 150 * (1 - busy))
            model.addConstr(battery[leg + 1] == battery[leg] - draw + recharge[leg])

        flight_cost = gp.quicksum(b.fee * fly[f] for f, b in BOOKINGS.items())
        courier_cost = gp.quicksum(fee * courier[i] for i, fee in COURIER.items())
        model.setObjective(flight_cost + courier_cost, GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("The pickup plan did not solve to optimality")
        flown = sorted((f for f in BOOKINGS if fly[f].X > 0.5), key=lambda f: BOOKINGS[f].launch)
        return {
            "status": "optimal",
            "objective": round(model.ObjVal),
            "bookings": flown,
            "courier_clinics": [i for i in COURIER if courier[i].X > 0.5],
        }


if __name__ == "__main__":
    result = solve()
    print(f"Optimal value: {result['objective']}")
    print(json.dumps(result))
