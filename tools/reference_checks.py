"""Independent numerical checks of the released LAFLOR instances."""

from itertools import combinations, permutations, product
from math import dist, exp, factorial, inf, isclose, sqrt
from functools import lru_cache
from pathlib import Path

if __package__:
    from .laflor_048 import verify_solution as fractional_siting
else:
    from laflor_048 import verify_solution as fractional_siting


def subsets(items):
    items = tuple(items)
    for n in range(len(items) + 1):
        yield from combinations(items, n)


def patrol():
    sites = [(0, 0), (20, 0), (10, 10)]
    targets = [(5, 0), (10, 0), (15, 0)]
    best = inf
    for opened in subsets(range(3)):
        if not opened or (0 in opened and 1 in opened and 2 not in opened):
            continue
        shortest = {}
        for served in subsets(range(3)):
            if not served:
                continue
            length = min(
                sum(dist(a, b) for a, b in zip(path, path[1:]))
                for perm in permutations(served)
                for i, j in product(opened, repeat=2)
                for path in [[sites[i], *(targets[k] for k in perm), sites[j]]]
            )
            if length <= 22:
                shortest[frozenset(served)] = length

        @lru_cache(maxsize=None)
        def cover(remaining):
            if not remaining:
                return 0
            return min(
                (
                    v + cover(remaining - s)
                    for s, v in shortest.items()
                    if s <= remaining
                ),
                default=inf,
            )

        best = min(best, 100 * len(opened) + cover(frozenset(range(3))))
    return best


def charging_network():
    places = [(0, 0), (15, 10), (25, 5), (20, 25), (35, 20), (40, 35), (10, 30)]
    demand = [(18, 8), (30, 10), (22, 28), (38, 25), (45, 38), (12, 35)]
    sites = range(1, 7)
    lengths = {
        (a, b): dist(places[a], places[b])
        for a in range(7)
        for b in sites
        if a != b and dist(places[a], places[b]) <= 30
    }
    coverage = [{t for t in sites if dist(places[t], point) <= 15} for point in demand]
    return sites, lengths, coverage


def charging(theta):
    sites, lengths, coverage = charging_network()
    feasible_terminals = {
        group
        for group in subsets(sites)
        if all(not stations.isdisjoint(group) for stations in coverage)
    }
    scale = sum(lengths.values())
    best = inf
    for active in subsets(sites):
        nodes = (0, *active)
        paths = {
            (a, b): (0 if a == b else lengths.get((a, b), inf))
            for a in nodes
            for b in nodes
        }
        for c, a, b in product(nodes, repeat=3):
            paths[a, b] = min(paths[a, b], paths[a, c] + paths[c, b])
        for terminals in subsets(active):
            if terminals in feasible_terminals:
                value = (
                    theta * sum(paths[0, t] for t in terminals) / scale
                    + (1 - theta) * len(active) / 6
                )
                best = min(best, value)
    return best


def case_table_rows(task):
    path = Path(__file__).resolve().parents[1] / "data" / "laflor" / task / "problem.md"
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]


def specimen_pickups(result):
    rows = case_table_rows("LAFLOR_043")
    courier = {r[0]: int(r[1]) for r in rows if len(r) == 2 and r[1].isdigit()}
    bookings = {
        r[0]: (r[1], *map(int, r[2:]))
        for r in rows if len(r) == 6 and r[2].isdigit()
    }
    choices = [[None, *(f for f, b in bookings.items() if b[0] == clinic)] for clinic in courier]
    reported_couriers = sorted(result["courier_clinics"])
    best, witness_cost = inf, None
    for selected in product(*choices):
        flights = sorted((f for f in selected if f is not None), key=lambda f: bookings[f][1])
        outsourced = [clinic for clinic, f in zip(courier, selected) if f is None]
        cost = sum(courier[clinic] for clinic in outsourced)
        charge, last_recovery = 400, 0
        for f in flights:
            _, launch, recovery, energy, fee = bookings[f]
            if launch < last_recovery:
                break
            charge = min(400, charge + 150 * (launch - last_recovery))
            if charge < energy:
                break
            charge -= energy
            last_recovery = recovery
            cost += fee
        else:
            best = min(best, cost)
            if flights == result["bookings"] and sorted(outsourced) == reported_couriers:
                witness_cost = cost
    assert witness_cost is not None and witness_cost == result["objective"]
    return best


def bay_schedules(swaps, packs, initially_full, reserve):
    """Map each physically feasible start schedule to its hourly charging load in kW."""
    schedules = {}
    for prefix in product(range(3), repeat=len(swaps) - 1):
        starts = (*prefix, 0)
        full, finishing = initially_full, 0
        draw = []
        for demand, new in zip(swaps, starts):
            if full < demand:
                break
            full -= demand
            available = packs - full - finishing
            if new > available or new + finishing > 2:
                break
            draw.append(2 * (new + finishing))
            full += finishing
            finishing = new
        else:
            if full >= reserve:
                schedules[starts] = draw
    return schedules


def cargo_charging(result):
    rows = case_table_rows("LAFLOR_124")
    bay_data = {r[0]: tuple(map(int, r[1:])) for r in rows if r[0] in ("North", "South")}
    hours = [tuple(map(int, r)) for r in rows if r[0].isdigit()]
    _, north_swaps, south_swaps, base, prices = zip(*hours)
    north = bay_schedules(north_swaps, *bay_data["North"])
    south = bay_schedules(south_swaps, *bay_data["South"])
    reported = result["starts"]
    assert set(reported) == set(bay_data)
    reported_pair = (tuple(reported["North"]), tuple(reported["South"]))
    best, witness_cost = inf, None
    for (n, north_kw), (s, south_kw) in product(north.items(), south.items()):
        charging_kw = [a + b for a, b in zip(north_kw, south_kw)]
        peak = max(h + kw for h, kw in zip(base, charging_kw))
        if peak > 12:
            continue
        energy_cost = sum(price * kw for price, kw in zip(prices, charging_kw))
        cost = energy_cost + 70 * peak
        best = min(best, cost)
        if (n, s) == reported_pair:
            assert result["peak_kw"] == peak and result["energy_cents"] == energy_cost
            witness_cost = cost
    assert witness_cost is not None and witness_cost == result["objective"]
    return best


def disjoint_route_length(active, terminal, lengths):
    """Enumerate simple hub routes to one terminal using only active stations."""
    routes = []
    for relays in subsets(s for s in active if s != terminal):
        for order in permutations(relays):
            path = (0, *order, terminal)
            arcs = list(zip(path, path[1:]))
            if all(arc in lengths for arc in arcs):
                routes.append((set(relays), sum(lengths[arc] for arc in arcs)))
    # Each route occurs once, including the direct arc when present. Distinct
    # simple routes with disjoint relays cannot share an arc.
    return min(
        (
            a + b
            for (left, a), (right, b) in combinations(routes, 2)
            if left.isdisjoint(right)
        ),
        default=inf,
    )


def disjoint_charging(result):
    sites, lengths, coverage = charging_network()
    scale = sum(lengths.values())
    opened = set(result["open_sites"])
    reported = {int(t): pair for t, pair in result["routes"].items()}
    assert len(opened) == len(result["open_sites"]) and opened <= set(sites)
    assert reported.keys() <= opened
    assert all(not stations.isdisjoint(reported) for stations in coverage)
    flown = 0
    for t, pair in reported.items():
        assert len(pair) == 2 and pair[0] != pair[1]
        assert set(pair[0][1:-1]).isdisjoint(pair[1][1:-1])
        for path in pair:
            assert path[0] == 0 and path[-1] == t
            assert len(path) == len(set(path)) and set(path[1:]) <= opened
            flown += sum(lengths[arc] for arc in zip(path, path[1:]))
    assert isclose(result["objective"], 0.5 * (flown / (2 * scale) + len(opened) / 6))

    best = inf
    for active in subsets(sites):
        distance = {t: disjoint_route_length(active, t, lengths) for t in active}
        for terminals in subsets(active):
            if all(not stations.isdisjoint(terminals) for stations in coverage):
                total = sum(distance[t] for t in terminals)
                best = min(best, 0.5 * (total / (2 * scale) + len(active) / 6))
    return best


def shared_freight(result):
    # If d units go directly to D2, the three loads are 20-d, d and 10-d.
    low, high = max(0, 20 - 16, 10 - 8), min(10, 6)
    assert 0 < low <= high < 10  # Every feasible routing opens all three links.
    loads = result["arc_loads"]
    assert set(loads) == {"S-D1", "S-D2", "D1-D2"}
    direct = loads["S-D2"]
    assert low <= direct <= high
    assert isclose(loads["S-D1"], 20 - direct) and isclose(loads["D1-D2"], 10 - direct)

    def cost(d):
        return 34.14 + 0.01 * (10 * (20 - d) + 14.14 * d + 10 * (10 - d))

    assert isclose(result["objective"], cost(direct))
    return min(cost(low), cost(high))  # The routing cost is linear on this interval.


def capacitated_siting(result):
    sites = {1: (15, 15), 2: (25, 10), 3: (20, 25), 4: (30, 20), 5: (10, 25)}
    capacities = {1: 1152, 2: 4608, 3: 288, 4: 120, 5: 1152}
    site_penalty = {1: 5, 2: 6, 3: 6, 4: 9, 5: 9}
    zones = [(18, 18), (28, 12), (22, 28), (32, 22), (12, 28), (20, 15)]
    demand = [50, 80, 60, 70, 40, 90]
    reported_sites = tuple(sorted(result["open_sites"]))
    reported_assignment = tuple(result["assigned_site_by_zone"])
    best, witness_cost = inf, None
    for opened in combinations(sites, 3):
        if any(dist((5, 5), sites[i]) > 30 for i in opened):
            continue
        if any(
            not 2 <= dist(sites[i], sites[j]) <= 16 for i, j in combinations(opened, 2)
        ):
            continue
        choices = [
            [None, *(i for i in opened if dist(sites[i], point) <= 10)]
            for point in zones
        ]
        for assigned in product(*choices):
            if any(
                sum(q for q, k in zip(demand, assigned) if k == i) > capacities[i]
                for i in opened
            ):
                continue
            unserved = sum(q for q, k in zip(demand, assigned) if k is None)
            score = 1000 * unserved + sum(site_penalty[i] for i in opened)
            best = min(best, score)
            if (opened, assigned) == (reported_sites, reported_assignment):
                witness_cost = score
    assert witness_cost is not None and isclose(result["objective"], witness_cost)
    return best


def truck_time_bound():
    sites = [
        (0, 0),
        (4, 10),
        (8, 2),
        (12, 0),
        (10, 9),
        (16, 3),
        (20, 1),
        (6, -2),
        (14, 7),
    ]
    weights = [0, 0.8, 1, 0.5, 0.7, 1.2, 1.5, 0.6, 0.9]
    best = inf
    for drone in subsets(set(range(1, 9)) - {6}):
        if len(drone) > 3 or sum(weights[i] for i in drone) > 3 + 1e-10:
            continue
        for tour in permutations(sorted(set(range(1, 9)) - set(drone))):
            path = (0, *tour, 0)
            best = min(
                best,
                60 / 50 * sum(dist(sites[a], sites[b]) for a, b in zip(path, path[1:])),
            )
    return best


def inventory():
    # Relax both drone range and station assignment; at least one site must open.
    scenarios = [(10, 10, 10), (20, 25, 20)]
    best = inf
    for links in product((1, 2), repeat=3):
        capacities = [
            sum(min(15 * n, d) for n, d in zip(links, row)) for row in scenarios
        ]
        for stock in (0, *capacities):
            cost = 500 + 10 * sum(links) + 2 * stock
            for demand, capacity in zip(scenarios, capacities):
                served = min(stock, capacity)
                cost += 0.5 * (
                    served + 100 * (sum(demand) - served) + 5 * (stock - served)
                )
            best = min(best, cost)
    # Both drones can make all three A-to-zone round trips within their range.
    assert 2 * (5 + sqrt(50) + sqrt(125)) <= 50
    return best


def payload_case():
    xy = [(0, 0), (2, 3), (5, 1), (4, 5)]
    delivery = [0, 1, 0.8, 0]
    pickup = [0, 0.5, 0, 1.2]
    best = inf
    for order in permutations((1, 2, 3)):
        for stops in subsets((1, 2)):
            cut = (0, *stops, 3)
            total = 0
            for a, b in zip(cut, cut[1:]):
                group = order[a:b]
                load = sum(delivery[i] for i in group)
                energy = 0
                for tail, head in zip((0, *group), (*group, 0)):
                    if not 0 <= load <= 3 + 1e-10:
                        energy = inf
                        break
                    energy += 3.5 * (6 + load) * dist(xy[tail], xy[head])
                    load += pickup[head] - delivery[head]
                if energy > 504:
                    total = inf
                    break
                total += energy
            best = min(best, 22 + 0.00248 * total)
    assert best < 22 + 2 * 0.78 * min(dist(xy[0], point) for point in xy[1:])
    return best


def pad_case():
    # Solve in waiting times independently of the reference's arrival-rate iteration.
    from scipy.optimize import least_squares
    import numpy as np

    counts = np.array(
        [[1, 2, 1], [1, 2, 1], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 0]]
    )
    demand = np.array([7, 5, 4, 6, 3, 2.5])
    ground = np.array([42, 38, 24, 25, 21, 22])
    base = np.array([16, 15, 8, 9, 8, 7])
    cost = np.array([30, 30, 18, 18, 18, 18])
    pad_cost = [[60, 180, 420], [90, 190, 360], [70, 200, 440]]
    best = -inf
    screened = []
    for pads in product((1, 2, 3), repeat=3):
        for fare in (55, 70, 85):

            def quantities(wait):
                return demand / (
                    1 + np.exp(0.035 * fare + 0.18 * (base + counts @ wait - ground))
                )

            def residual(wait):
                rates = counts.T @ quantities(wait)
                # Multiply through by positive denominators. At a root this excludes
                # overload because the right side is positive and wait is nonnegative.
                errors = []
                for n, lam, w in zip(pads, rates, wait):
                    u = lam / 12
                    poly = sum(u**k / factorial(k) for k in range(n))
                    term = u**n / factorial(n)
                    errors.append(
                        w * (12 * n - lam) * (poly * (1 - u / n) + term) - 30 * term
                    )
                return errors

            result = least_squares(
                residual, [5, 5, 5], bounds=(0, 100), gtol=1e-13, ftol=1e-13, xtol=1e-13
            )
            assert max(abs(v) for v in residual(result.x)) < 1e-8
            capture = quantities(result.x)
            assert np.all(counts.T @ capture < 12 * np.array(pads))
            profit = float(
                (fare - cost) @ capture
                - sum(row[n - 1] for row, n in zip(pad_cost, pads))
            )
            best = max(best, profit)
            if np.all(counts.T @ quantities(np.zeros(3)) < 12 * np.array(pads)):
                screened.append(profit)
    assert len(screened) == 12 and abs(max(screened) - 114.592318) < 1e-6
    return best


def fleet():
    # Enumerate integer designs. A forward fill tests feasibility; with constant
    # recharge wear the minimum daytime purchase is max(0, use - initial charge).
    schedules = [
        [
            (18, 1, 0),
            (25, 1, 0.5),
            (12, 0, 0.8 - 0.05 - 2 / 60 - 12 / 150),
            (15, 1, 0.05),
        ],
        [
            (22, 1, 0),
            (8, 0, 0.6 - 0.05 - 2 / 60 - 8 / 150),
            (30, 1, 0.05),
            (20, 1, 0.4),
        ],
    ]
    best = inf
    for empty, mtow, hover, cruise in [
        (450, 900, 0.055, 0.00028),
        (600, 1200, 0.042, 0.00035),
    ]:
        for mass in range(50, 301):
            if empty + mass + 90 > mtow or 3 * mass < hover * mtow:
                continue
            cap = 0.32 * mass
            for power, purchase in [
                (50, 28401 + 45506),
                (150, 75000 + 47781),
                (350, 140000 + 65984),
            ]:
                draw = charge = 0
                feasible = True
                for schedule in schedules:
                    total = 0
                    soc = cap
                    for km, pax, window in schedule:
                        energy = (empty + mass + 90 * pax) * (
                            hover * 2 / 60 + cruise * km
                        )
                        soc = min(cap, soc + power * window) - energy
                        if soc < -1e-9:
                            feasible = False
                        total += energy
                    draw += total
                    charge += max(0, total - cap)
                if feasible:
                    cost = (
                        2 * empty * 770 * 0.7 + 8 * purchase + 2 * 0.4 * mass * 285
                    ) / (13 * 330)
                    cost += 2 * 51300 / 330 + 0.25 * draw + 285 / 2000 * charge
                    best = min(best, cost)
    return best


def passenger_cost():
    # For each open set, minimize over the air/ground assignment with a separate MILP.
    import gurobipy as gp
    from gurobipy import GRB

    xy = [(0, 0), (10, 10), (0, 10), (10, 0)]
    best = inf
    for opened in subsets(range(4)):
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        air = m.addVars(3, vtype=GRB.BINARY)
        flow = {(0, 1): gp.LinExpr(), (2, 3): gp.LinExpr()}
        cost = gp.LinExpr(500 * len(opened))
        for k, (a, b, n) in enumerate(((0, 1, 100), (2, 3, 100), (0, 2, 50))):
            road = 2 * dist(xy[a], xy[b])
            cost += n * road
            if not opened:
                air[k].UB = 0
                continue
            i, j = [min(opened, key=lambda p: (dist(xy[x], xy[p]), p)) for x in (a, b)]
            if i == j:
                air[k].UB = 0
                continue
            time = 2 * (dist(xy[a], xy[i]) + dist(xy[b], xy[j])) + 0.5 * dist(
                xy[i], xy[j]
            )
            cost += n * (time - road) * air[k]
            diagonal = tuple(sorted((i, j)))
            if diagonal in flow:
                flow[diagonal] += n * air[k]
        m.setObjective(cost + 0.5 * flow[0, 1] * flow[2, 3])
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError("Skyport assignment failed")
        best = min(best, m.ObjVal)
        m.dispose()
    return best


def verify(tid, result):
    if tid == "LAFLOR_101":
        return capacitated_siting(result)
    if tid == "LAFLOR_048":
        return fractional_siting(result)
    if tid == "LAFLOR_152":
        assert (
            result["pads"] == [1, 2, 1]
            and result["fare"] == 85
            and result["designs"] == 81
        )
        for actual, expected in [
            (result["rates"], [6.885112, 14.139337, 7.254224]),
            (result["waits"], [3.365231, 1.328981, 3.821411]),
        ]:
            assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-6
        return pad_case()
    if tid == "LAFLOR_106":
        return patrol()
    if tid == "LAFLOR_116":
        return passenger_cost()
    if tid == "LAFLOR_118":
        # The S1/S3 conflict removes route 1, and two ports can open at most one other market.
        return max(
            1500 / (1 + exp((-25 - 0.8) - (-6 - 4.5))),
            500 / (1 + exp((-20 - 1.2) - (-10 - 4 + 2))),
        )
    if tid == "LAFLOR_076":
        return disjoint_charging(result)
    if tid == "LAFLOR_124":
        return cargo_charging(result)
    if tid == "LAFLOR_022":
        return charging(0.5)
    if tid == "LAFLOR_136":
        return truck_time_bound()
    if tid == "LAFLOR_144":
        # Forced route costs $0.50. On-time departures and path 0 on flights 2–4
        # trigger none of the four conflict pairs, so the lower bound is attained.
        return 0.5
    if tid == "LAFLOR_145":
        flights = {tuple(leg["arc"]): leg for leg in result["flights"]}
        expected = {(0, 2): 1.8, (2, 3): 1.0, (3, 1): 2.2, (1, 4): 1.7}
        assert not result["truck_travel_arcs"] and set(flights) == set(expected)
        assert all(
            abs(flights[arc]["payload"] - weight) < 1e-6
            for arc, weight in expected.items()
        )
        assert abs(sum(leg["energy"] for leg in flights.values()) - 418.5647859) < 1e-5
        return payload_case()
    if tid == "LAFLOR_043":
        return specimen_pickups(result)
    if tid == "LAFLOR_062":
        return max(
            100 * (1 in o and 2 in o)
            + 110 * (1 in o and 3 in o)
            + sum({1: 20, 2: 30, 3: 30}[i] for i in o)
            for o in subsets((1, 2, 3))
            if sum({1: 10, 2: 15, 3: 12}[i] for i in o) <= 25
        )
    if tid == "LAFLOR_073":
        return shared_freight(result)
    if tid == "LAFLOR_019":
        # Connecting both demand points requires S->D1. The third link can improve
        # operating cost only when the direct diagonal is present.
        return min(20 + 0.01 * 300, 24.14 + 0.01 * 241.4, 34.14 + 0.01 * 241.4)
    if tid == "LAFLOR_077":
        # Triangle lower bounds on arrival times are jointly attained in monotone order.
        return min(
            20 * len(o) + sum(min(abs(i - j) for j in o) for i in (1, 2, 8, 9))
            for o in ((0,), (10,), (0, 10))
        )
    if tid == "LAFLOR_087":
        return fleet()
    if tid == "LAFLOR_099":
        # Different possible partitions: enumerate direct tours again as an assignment bound.
        xy = [(0, 0), (3, 0), (6, 0), (4, 3)]
        values = []
        for t in subsets((1, 2)):
            d = set((1, 2, 3)) - set(t)
            for road, flight in product(permutations(t), permutations(d)):
                total = 0.0
                for speed, visits in ((30, road), (60, flight)):
                    time = 0.0
                    prev = 0
                    for i in visits:
                        time += dist(xy[prev], xy[i]) / speed
                        total += time * {1: 2, 2: 1, 3: 3}[i]
                        time += 5 / 60 if i == 3 else 0
                        prev = i
                values.append(total)
        assert result["worst_recovery_hours"] <= 0.5
        return min(values)
    if tid == "LAFLOR_005":
        return inventory()
    raise ValueError(f"Missing independent check for {tid}")
