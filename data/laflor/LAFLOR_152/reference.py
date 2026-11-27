import itertools
import math

import gurobipy as gp
from gurobipy import GRB


STATIONS = ("A", "H", "B")
STATION_INDEX = {station: index for index, station in enumerate(STATIONS)}
MARKETS = (
    (("A", "H", "H", "B"), 7.0, 42.0, 16.0, 30.0),
    (("B", "H", "H", "A"), 5.0, 38.0, 15.0, 30.0),
    (("A", "H"), 4.0, 24.0, 8.0, 18.0),
    (("H", "B"), 6.0, 25.0, 9.0, 18.0),
    (("B", "H"), 3.0, 21.0, 8.0, 18.0),
    (("H", "A"), 2.5, 22.0, 7.0, 18.0),
)
FARES = (55.0, 70.0, 85.0)
ALPHA = 0.035
BETA = 0.18
MU = 12.0
PAD_COST = {
    "A": {1: 60.0, 2: 180.0, 3: 420.0},
    "H": {1: 90.0, 2: 190.0, 3: 360.0},
    "B": {1: 70.0, 2: 200.0, 3: 440.0},
}


def erlang_c(pads, movement_rate):
    if movement_rate == 0.0:
        return 0.0
    offered_load = movement_rate / MU
    utilization = movement_rate / (pads * MU)
    tail = offered_load**pads / (
        math.factorial(pads) * (1.0 - utilization)
    )
    finite_sum = sum(
        offered_load**k / math.factorial(k) for k in range(pads)
    )
    return tail / (finite_sum + tail)


def waiting_minutes(pads, movement_rate):
    return 30.0 * erlang_c(pads, movement_rate) / (
        pads * MU - movement_rate
    )


def captures_and_loads(pads, fare, station_loads):
    waits = [
        waiting_minutes(pads[j], station_loads[j])
        for j in range(len(STATIONS))
    ]
    captures = []
    implied_loads = [0.0 for _ in STATIONS]
    for path, demand, ground, pre_wait_uam, _ in MARKETS:
        total_wait = sum(
            waits[STATION_INDEX[station]] for station in path
        )
        captured = demand / (
            1.0
            + math.exp(
                ALPHA * fare
                + BETA * (pre_wait_uam + total_wait - ground)
            )
        )
        captures.append(captured)
        for station in path:
            implied_loads[STATION_INDEX[station]] += captured
    return captures, implied_loads, waits


def solve_fixed_point(pads, fare):
    capacities = [MU * pads[j] for j in range(len(STATIONS))]
    station_loads = [0.5 * capacity for capacity in capacities]
    damping = 0.02
    for _ in range(200000):
        _, implied_loads, _ = captures_and_loads(
            pads, fare, station_loads
        )
        residual = max(
            abs(station_loads[j] - implied_loads[j])
            for j in range(len(STATIONS))
        )
        if residual <= 1e-12:
            break
        station_loads = [
            (1.0 - damping) * station_loads[j]
            + damping
            * min(implied_loads[j], math.nextafter(capacities[j], 0.0))
            for j in range(len(STATIONS))
        ]
    else:
        raise RuntimeError("Demand-queue fixed point did not converge")

    captures, implied_loads, waits = captures_and_loads(
        pads, fare, station_loads
    )
    residual = max(
        abs(station_loads[j] - implied_loads[j])
        for j in range(len(STATIONS))
    )
    if residual > 1e-9:
        raise RuntimeError("Demand-queue fixed-point residual is too large")
    if any(station_loads[j] >= capacities[j] for j in range(len(STATIONS))):
        raise RuntimeError("Unstable station queue")
    return captures, station_loads, waits


# Each discrete design has a unique demand-queue fixed point. Compute those
# coefficients from the governing equations, then solve the exact 81-option
# multiple-choice master problem with Gurobi.
options = []
for pads in itertools.product((1, 2, 3), repeat=len(STATIONS)):
    for fare in FARES:
        captures, station_loads, waits = solve_fixed_point(pads, fare)
        passenger_contribution = sum(
            (fare - market[4]) * captured
            for market, captured in zip(MARKETS, captures)
        )
        objective = passenger_contribution - sum(
            PAD_COST[station][pads[STATION_INDEX[station]]]
            for station in STATIONS
        )
        options.append(
            {
                "pads": pads,
                "fare": fare,
                "captures": captures,
                "loads": station_loads,
                "waits": waits,
                "objective": objective,
            }
        )

model = gp.Model("SkyBridge_Queue_Logit_Design")
model.Params.OutputFlag = 0
select = model.addVars(len(options), vtype=GRB.BINARY, name="select")
model.addConstr(select.sum() == 1, name="choose_one_design")
model.setObjective(
    gp.quicksum(options[k]["objective"] * select[k] for k in range(len(options))),
    GRB.MAXIMIZE,
)
model.optimize()

if model.Status != GRB.OPTIMAL:
    raise RuntimeError(f"Gurobi status {model.Status}")

chosen_index = next(k for k in range(len(options)) if select[k].X > 0.5)
best = options[chosen_index]
print("Best decision (n_A,n_H,n_B,p):", best["pads"], best["fare"])
print("Market captures:", best["captures"])
print("Station loads:", best["loads"])
print("Station waits:", best["waits"])
print(f"Optimal value: {model.ObjVal:.15f}")

import json
print(json.dumps({"status":"optimal","objective":model.ObjVal,"pads":best["pads"],"fare":best["fare"],"rates":best["loads"],"waits":best["waits"],"designs":len(options)}))
