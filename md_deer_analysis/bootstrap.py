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

        self.js_bootstrap = {}
        self.js_avg = {}

        for pair in self.pairs:
            self.js_avg[pair] = jensen_shannon(self.sim.ensemble_average[pair],
                                               self.deer_distributions[pair])

    def js_single_member(self, pair, member="mem_0"):
        return jensen_shannon(self.deer_distributions[pair],
                              self.sim.distributions[pair][member])

    def get_name(self):
        return self.sim.get_name()

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

        self.js_bootstrap = js

    def quantiles(self, lower=0.25, upper=0.75):
        first_quantile = {}
        last_quantile = {}
        for pair in self.pairs:
            first_quantile[pair] = np.quantile(
                self.js_bootstrap[pair], q=lower)
            last_quantile[pair] = np.quantile(self.js_bootstrap[pair], q=upper)
        return first_quantile, last_quantile


def write_to_table(fnm, js_data: list):
    with open(fnm, "w") as my_file:
        my_file.write(
            "Ensemble,Pair,Divergence,first_quartile,fourth_quartile\n")

        for js in js_data:
            ensemble_name = js.get_name()
            first_quantile, last_quantile = js.quantiles()
            for pair in js.pairs:
                pair_name = pair.replace("_", "/")
                my_file.write("{},{},{},{},{}\n".format(
                    ensemble_name, pair_name, js.js_avg[pair],
                    first_quantile[pair], last_quantile[pair]))
