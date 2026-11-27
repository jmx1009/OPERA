"""Enumerate LAFLOR_048 and measure the effect of changing one modeling rule."""

import sys
from itertools import combinations, product
from math import dist, isclose, isfinite
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.objective_check import DEFAULT_ABSOLUTE_TOLERANCE, objective_matches

PROBLEM = ROOT / "data/laflor/LAFLOR_048/problem.md"
COMPARISONS = {
    "reference": "All stated rules",
    "single_source": "Assign each accepted order to one site",
    "partial": "Give credit for partial orders",
    "unlimited": "Remove handling capacities at open sites",
    "ungated": "Let closed sites dispatch at their stated capacities",
    "neighbor": "Require just one neighbor within 10 km per open site",
    "at_most": "Open at most three sites",
    "no_site_cost": "Omit safety and noise costs",
    "strict": "Require intersite distances strictly below 10 km",
    "no_upper": "Omit the intersite distance upper bound",
}


class Site(NamedTuple):
    point: tuple[int, int]
    capacity: int
    safety: float
    noise: float


class Zone(NamedTuple):
    point: tuple[int, int]
    demand: int


class Design(NamedTuple):
    sites: tuple[int, ...]
    served: int
    objective: float


def read_instance():
    sites, zones = {}, {}
    for line in PROBLEM.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("| S", "| C")):
            continue
        label, coordinates, *values = [
            cell.strip() for cell in line.strip("|").split("|")
        ]
        if not label[1:].isdigit():
            continue
        x, y = map(int, coordinates.strip("()").split(","))
        index = int(label[1:])
        if label.startswith("S"):
            capacity, safety, noise = values
            sites[index] = Site((x, y), int(capacity), float(safety), float(noise))
        else:
            zones[index] = Zone((x, y), int(values[0]))
    if len(sites) != 5 or len(zones) != 6:
        raise ValueError(
            "Expected the five sites and six customer zones in the problem"
        )
    return sites, zones


def subsets(items):
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def legal_sites(sites, rule):
    candidates = tuple(i for i, site in sites.items() if dist((5, 5), site.point) <= 30)
    sizes = range(4) if rule == "at_most" else (3,)
    for size in sizes:
        for group in combinations(candidates, size):
            pairs = {
                (i, j): dist(sites[i].point, sites[j].point)
                for i, j in combinations(group, 2)
            }
            if any(d < 2 for d in pairs.values()):
                continue
            if rule == "neighbor":
                if all(
                    any(i in pair and d <= 10 for pair, d in pairs.items())
                    for i in group
                ):
                    yield group
            elif rule == "no_upper" or all(
                (d < 10 if rule == "strict" else d <= 10) for d in pairs.values()
            ):
                yield group


def indivisible_volume(sites, zones, routes):
    best = 0
    customers = tuple(zones)
    for assignment in product(*[(None, *routes[c]) for c in customers]):
        loads = dict.fromkeys(sites, 0)
        for c, site in zip(customers, assignment):
            if site is not None:
                loads[site] += zones[c].demand
        if all(load <= sites[i].capacity for i, load in loads.items()):
            best = max(best, sum(loads.values()))
    return best


def service_volume(sites, zones, opened, rule):
    active = opened
    if rule == "ungated":
        active = tuple(i for i, site in sites.items() if dist((5, 5), site.point) <= 30)
    routes = {
        c: {i for i in active if dist(zone.point, sites[i].point) <= 10}
        for c, zone in zones.items()
    }
    if rule == "single_source":
        return indivisible_volume(sites, zones, routes)
    if rule == "unlimited":
        return sum(zones[c].demand for c, options in routes.items() if options)

    # For any subset of sites T, all orders with no route outside T must fit
    # inside its capacity. The largest shortfall over T is the undeliverable
    # volume by max-flow/min-cut, including orders with no eligible site.
    cuts = [
        (
            sum(sites[i].capacity for i in group),
            {c for c, options in routes.items() if options.issubset(group)},
        )
        for group in subsets(active)
    ]

    def shortfall(accepted):
        return max(
            sum(zones[c].demand for c in accepted if c in confined) - capacity
            for capacity, confined in cuts
        )

    if rule == "partial":
        return sum(zone.demand for zone in zones.values()) - shortfall(zones)
    best = 0
    for accepted in subsets(tuple(zones)):
        if shortfall(accepted) <= 0:
            best = max(best, sum(zones[c].demand for c in accepted))
    return best


def enumerate_designs(sites, zones, rule="reference"):
    if rule not in COMPARISONS:
        raise ValueError(f"Unknown comparison: {rule}")
    total = sum(zone.demand for zone in zones.values())
    designs = []
    for opened in legal_sites(sites, rule):
        served = service_volume(sites, zones, opened, rule)
        charge = sum(80 * sites[i].safety + 80 * sites[i].noise for i in opened)
        if rule == "no_site_cost":
            charge = 0
        designs.append(Design(opened, served, total - served + charge))
    return designs


def verify_solution(result):
    sites, zones = read_instance()
    designs = enumerate_designs(sites, zones)
    opened = tuple(sorted(result["open_sites"]))
    assert opened in {d.sites for d in designs}
    accepted = result["accepted_zones"]
    assert len(set(accepted)) == len(accepted) and set(accepted) <= zones.keys()
    received = dict.fromkeys(zones, 0.0)
    dispatched = dict.fromkeys(opened, 0.0)
    for delivery in result["deliveries"]:
        c, i, q = delivery["zone"], delivery["site"], delivery["packages"]
        assert c in zones and i in opened and isfinite(q) and q > 0
        assert dist(zones[c].point, sites[i].point) <= 10
        received[c] += q
        dispatched[i] += q
    assert all(q <= sites[i].capacity + 1e-7 for i, q in dispatched.items())
    for c, q in received.items():
        assert isclose(q, zones[c].demand if c in accepted else 0, abs_tol=1e-7)
    missed = sum(zone.demand for c, zone in zones.items() if c not in accepted)
    charge = sum(80 * sites[i].safety + 80 * sites[i].noise for i in opened)
    assert isclose(result["unserved_packages"], missed, abs_tol=1e-7)
    assert isclose(result["site_penalty"], charge, abs_tol=1e-7)
    assert isclose(result["objective"], missed + charge, abs_tol=1e-7)
    best = min(d.objective for d in designs)
    assert isclose(result["objective"], best, abs_tol=1e-7)
    return best


def main():
    sites, zones = read_instance()
    designs = enumerate_designs(sites, zones)
    reference = min(d.objective for d in designs)
    print("Open sites | Served packages | Penalty")
    for design in designs:
        print(f"{design.sites} | {design.served} | {design.objective:g}")
    print(f"\nOne-rule comparison | Optimal penalty | Absolute error | Within {DEFAULT_ABSOLUTE_TOLERANCE:g}")
    for rule, label in COMPARISONS.items():
        value = min(d.objective for d in enumerate_designs(sites, zones, rule))
        error = abs(value - reference)
        verdict = "yes" if objective_matches(value, reference) else "no"
        print(f"{label} | {value:g} | {error:g} | {verdict}")


if __name__ == "__main__":
    main()
