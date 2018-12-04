#!/usr/bin/env python
"""
Bootstraps over simulation ensemble members to calculate Jensen-Shannon
divergence and errors.
"""

import json
import numpy as np
from md_deer_analysis.experimental import MultiPair
from md_deer_analysis.simulation_ensemble import SimulationEnsemble
from md_deer_analysis.utils import jensen_shannon


class Bootstrap:
    def __init__(self, sim_file, exp_file, sigma=0.25):
        exp = MultiPair()
        exp.read_from_json(exp_file)
        exp_dict = exp.get_as_single_dataset()

        self.pairs = exp.get_names()
        self.bins = exp_dict[self.pairs[0]]['bins']  # Use the first pair's bins as the default

        self.exp_distributions = {}
        for pair in self.pairs:
            self.exp_distributions[pair] = exp_dict[pair]['distribution']

        self.sigma = sigma
        self.sim = SimulationEnsemble(sim_file)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_bins(self, bins):
        self.bins = bins

    def calculate_average_js(self):
        avg_js = {}
        for pair in self.pairs:
            avg_js[pair] = jensen_shannon(
                self.sim.get_distributions(self.bins, pair=pair),
                self.exp_distributions[pair])
        return avg_js

    def bootstrap_histograms(self, n=1000, json_filename=None):
        histograms = {}
        js_div = {}

        for i in range(n):
            sim_dict = self.sim.re_sample(bins=self.bins, sigma=self.sigma)
            for pair in self.pairs:
                if pair not in histograms:
                    histograms[pair] = sim_dict[pair]
                    js_div[pair] = [jensen_shannon(histograms[pair],
                                                   self.exp_distributions[pair])]
                else:
                    histograms[pair] += sim_dict[pair]
                    js_div[pair].append(jensen_shannon(sim_dict[pair],
                                                       self.exp_distributions[pair]))

        for pair in self.pairs:
            histograms[pair] /= np.sum(histograms[pair])
            histograms[pair] = histograms[pair].tolist()
            js_div[pair] = np.sort(js_div[pair]).tolist()

        if json_filename:
            json.dump(histograms, open(json_filename, "w"))
            json.dump(js_div, open('{}_js.json'.format(json_filename[:-5]), "w"))

        return histograms, js_div


def get_quartiles(js_div):
    quartiles = []
    for pair in js_div.keys():
        percentiles = [np.percentile(js_div[pair], q=25),
                       np.percentile(js_div[pair], q=75)]
        quartiles.append(percentiles)

    return quartiles
