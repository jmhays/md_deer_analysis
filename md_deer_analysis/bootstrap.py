#!/usr/bin/env python
"""
Bootstraps over simulation ensemble members to calculate Jensen-Shannon
divergence and errors.
"""

import numpy as np
from md_deer_analysis.experimental import MultiPair
from md_deer_analysis.simulation_ensemble import SimulationEnsemble
from md_deer_analysis.utils import jensen_shannon


class JS:
    def __init__(self, sim: SimulationEnsemble, exp: MultiPair):
        self.sim = sim
        self.exp = exp
        self.pairs = self.exp.get_names()

        sigma = None
        bins = []
        self.deer_distributions = {}

        exp_as_dict = self.exp.get_as_single_dataset()
        for pair in self.pairs:
            pair_metadata = exp_as_dict[pair]
            if not sigma:
                sigma = pair_metadata['sigma']
            if not bins:
                bins = pair_metadata['bins']

            self.deer_distributions[pair] = pair_metadata['distribution']

        # Initialize the simulation histograms
        self.sim.calculate_distributions(bins=bins, sigma=sigma)

    def get_average_js(self):
        js = {}
        for pair in self.pairs:
            js[pair] = jensen_shannon(self.sim.ensemble_average[pair],
                                      self.deer_distributions[pair])
        return js

    def bootstrap(self, n=1000):
        js = {}
        for pair in self.pairs:
            js[pair] = np.zeros(shape=n)

        for i in range(n):
            resampled = self.sim.re_sample()
            for pair in self.pairs:
                js[pair][i] = jensen_shannon(resampled[pair],
                                             self.deer_distributions[pair])

        for pair in self.pairs:
            js[pair].sort()
        return js


class PlotBootstrap:
    def __init__(self, js_avg: dict, js_bootstrap: dict):
        self.js_avg = js_avg
        self.first_quartiles = {}
        self.fourth_quartiles = {}

        for pair in self.js_avg.keys():
            self.first_quartiles[pair] = np.quantile(js_bootstrap[pair], q=0.25)
            self.fourth_quartiles[pair] = np.quantile(js_bootstrap[pair], q=0.75)

    def write_to_table(self, fnm):
        with open(fnm, "w") as myfile:
            myfile.write("# pair\tjs\tfirst_quartile\tfourth_quartile\n")
            for pair in self.js_avg.keys():
                myfile.write("pair_{}\t{}\t{}\t{}\n".format(
                    pair, self.js_avg[pair], self.first_quartiles[pair],
                    self.fourth_quartiles[pair]))
