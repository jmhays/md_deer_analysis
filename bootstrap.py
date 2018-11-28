#!/usr/bin/env python
"""
Bootstraps over simulation ensemble members to calculate Jensen-Shannon
divergence and errors.
"""

import argparse
import json
from md_deer_analysis.experimental import MultiPair
from md_deer_analysis.simulation_ensemble import SimulationEnsemble
from md_deer_analysis.utils import jensen_shannon

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', help='path to simulation data')
    parser.add_argument('-f2', help='path to experimental data')
    parser.add_argument('-s', help='sigma for smoothing', type=float)
    parser.add_argument('-o', help='path to output (json)')
    parser.add_argument(
        '-n', help='number of bootstrap resampling iterations', type=int)
    args = parser.parse_args()

    sim = SimulationEnsemble(args.f1)
    exp = MultiPair()
    exp.read_from_json(args.f2)
    out = args.o

    pairs = sim.pairs
    bins = None

    js_div = {'average': {}, 'bootstrap': {}}
    exp_dict = {}

    for pair in pairs:
        js_div['bootstrap'][pair] = []
        exp_dict[pair] = exp.get_as_single_dataset()[pair]['distribution']
        if not bins:
            bins = exp.get_as_single_dataset()[pair]['bins']
        js_div['average'][pair] = jensen_shannon(
            sim.get_distributions(bins, pair=pair), exp_dict[pair])

    for i in range(args.n):
        sim_dict = sim.re_sample(bins=bins, sigma=args.s)
        for pair in pairs:
            js_div['bootstrap'][pair].append(
                jensen_shannon(sim_dict[pair], exp_dict[pair]))

    for pair in pairs:
        js_div['bootstrap'][pair].sort()

    json.dump(js_div, open(args.o, "w"))
