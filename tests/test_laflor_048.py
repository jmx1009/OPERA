"""Cross-check every siting comparison with a separate SciPy/HiGHS formulation."""

from itertools import combinations
from math import dist, inf
import unittest

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from tools.laflor_048 import COMPARISONS, enumerate_designs, read_instance


def highs_penalty(sites, zones, rule, fixed_sites=None):
    # Variables are site openings, order acceptances and fractions of each order
    # carried on each eligible link. Binary link fractions impose single sourcing.
    opened = {i: k for k, i in enumerate(sites)}
    accepted = {c: k for k, c in enumerate(zones, start=len(sites))}
    routes = [
        (c, i)
        for c, zone in zones.items()
        for i, site in sites.items()
        if dist(zone.point, site.point) <= 10 and dist((5, 5), site.point) <= 30
    ]
    share = {arc: k for k, arc in enumerate(routes, start=len(sites) + len(zones))}
    size = len(sites) + len(zones) + len(share)
    total = sum(zone.demand for zone in zones.values())
    objective = np.zeros(size)
    integer = np.zeros(size, dtype=int)
    lower, upper = np.zeros(size), np.ones(size)
    for i, k in opened.items():
        integer[k] = 1
        if rule != "no_site_cost":
            objective[k] = 80 * sites[i].safety + 80 * sites[i].noise
        if fixed_sites is not None:
            lower[k] = upper[k] = int(i in fixed_sites)
        if dist((5, 5), sites[i].point) > 30:
            upper[k] = 0
    for c, k in accepted.items():
        objective[k] = -zones[c].demand
        integer[k] = int(rule != "partial")
    if rule == "single_source":
        integer[list(share.values())] = 1

    rows, lows, highs = [], [], []

    def add(terms, low, high):
        row = np.zeros(size)
        for k, coefficient in terms:
            row[k] += coefficient
        rows.append(row)
        lows.append(low)
        highs.append(high)

    add([(k, 1) for k in opened.values()], 0 if rule == "at_most" else 3, 3)
    for i, j in combinations(sites, 2):
        separation = dist(sites[i].point, sites[j].point)
        upper_violation = separation >= 10 if rule == "strict" else separation > 10
        if separation < 2 or (upper_violation and rule not in {"no_upper", "neighbor"}):
            add([(opened[i], 1), (opened[j], 1)], -inf, 1)
    for i, site in sites.items():
        if rule == "neighbor":
            neighbors = [
                j
                for j in sites
                if j != i and 2 <= dist(site.point, sites[j].point) <= 10
            ]
            add([(opened[i], 1), *((opened[j], -1) for j in neighbors)], -inf, 0)
        load = [(k, zones[c].demand) for (c, j), k in share.items() if j == i]
        if rule == "ungated":
            add(load, -inf, site.capacity)
        else:
            capacity = total if rule == "unlimited" else site.capacity
            add([*load, (opened[i], -capacity)], -inf, 0)
    for c in zones:
        allocation = [(k, 1) for (customer, _), k in share.items() if customer == c]
        add([*allocation, (accepted[c], -1)], 0, 0)

    result = milp(
        objective,
        integrality=integer,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(rows, lows, highs),
        options={"mip_rel_gap": 0},
    )
    if result.status == 2:
        return None
    if not result.success:
        raise RuntimeError(result.message)
    return total + result.fun


class FormulationComparisonTests(unittest.TestCase):
    def test_flow_cuts_agree_with_highs_for_every_design(self):
        sites, zones = read_instance()
        for rule in COMPARISONS:
            expected = {
                d.sites: d.objective for d in enumerate_designs(sites, zones, rule)
            }
            with self.subTest(rule=rule):
                self.assertAlmostEqual(
                    highs_penalty(sites, zones, rule), min(expected.values())
                )
            sizes = range(4) if rule == "at_most" else (3,)
            for k in sizes:
                for group in combinations(sites, k):
                    with self.subTest(rule=rule, sites=group):
                        value = highs_penalty(sites, zones, rule, group)
                        if group in expected:
                            self.assertAlmostEqual(value, expected[group])
                        else:
                            self.assertIsNone(value)
