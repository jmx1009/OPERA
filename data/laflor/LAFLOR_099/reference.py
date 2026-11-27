"""Enumerate nominal response plans, certifying endurance under budgeted travel delays."""
from itertools import permutations
from math import dist
import json

xy = {0: (0, 0), 1: (3, 0), 2: (6, 0), 3: (4, 3), 4: (9, 0)}
priority = {1: 2, 2: 1, 3: 3}
service = {1: 0, 2: 0, 3: 5/60}


def solve():
    # A relaxation with independent vehicles and unlimited endurance is a lower bound.
    # Both vehicles start at the depot. For a fixed assignment, no detour or
    # rendezvous can advance the service of any customer in a fixed visit order.
    candidates = []
    for truck_mask in range(4):
        truck_sites = [i for i in (1, 2) if truck_mask & (1 << (i-1))]
        drone_sites = set(priority).difference(truck_sites)
        for truck_order in permutations(truck_sites):
            for drone_order in permutations(sorted(drone_sites)):
                starts = {}
                for order, speed in ((truck_order, 30), (drone_order, 60)):
                    t = 0
                    prev = 0
                    for i in order:
                        t += dist(xy[prev], xy[i]) / speed
                        starts[i] = t
                        t += service[i]
                        prev = i
                obj = sum(priority[i] * starts[i] for i in starts)
                candidates.append((obj, truck_order, drone_order))
    for obj, truck, drone in sorted(candidates):
        if obj > min(x[0] for x in candidates) + 1e-10:
            raise RuntimeError("The relaxed optimum was not certified feasible.")
        legs = [dist(xy[i], xy[j])/60 for i, j in zip((0, *drone), (*drone, 4))]
        # A budget of one permits one full flight-leg deviation on the route.
        worst_flight = sum(legs) + 0.3 * max(legs) + sum(service[i] for i in drone)
        if worst_flight <= 0.5 and sum({1: 5, 2: 3, 3: 0}[i] for i in drone) <= 10:
            # Joint launch at 0, independent service and joint recovery at 4.
            # The truck can wait at 4 under any truck travel-time realization;
            # the drone can land there on arrival when the truck is earlier.
            road = [dist(xy[i],xy[j])/30 for i,j in zip((0,*truck),(*truck,4))]
            # Early drone arrival may wait for the truck. The last meeting occurs
            # by the later worst-case arrival, also within the flight-duration limit.
            duration = max(worst_flight, sum(road) + 0.3*max(road))
            if duration <= 0.5:
                return {"status": "optimal", "objective": obj, "truck": truck, "drone": drone,
                        "worst_recovery_hours": duration}
    raise RuntimeError("No feasible relaxed optimum")


if __name__ == "__main__":
    print(json.dumps(solve()))
