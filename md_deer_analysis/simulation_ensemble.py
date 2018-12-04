import json
import numpy as np
from md_deer_analysis.utils import gaussian_smoothing


class SimulationEnsemble:
    def __init__(self, json_filename):
        """
        Loads and manipulates a simulation ensemble.

        Parameters
        ----------
        json_filename : (str)
            This file should be structured as {"pair_name": {"member_name": ...}}
        """
        self.metadata = json.load(open(json_filename))
        self.pairs = list(self.metadata.keys())
        self.members = list(self.metadata[self.pairs[0]].keys())

        self.num_pairs = len(self.pairs)
        self.num_members = len(self.members)

    def get_samples(self, pair=None, member=None):
        if pair and member:
            data = self.metadata[pair][member]
        elif pair:
            data = np.concatenate(list(self.metadata[pair].values()))
        elif member:
            data = {}
            for pair in self.pairs:
                data[pair] = self.metadata[pair][member]
        else:
            data = {}
            for pair in self.pairs:
                data[pair] = np.concatenate(
                    (list(self.metadata[pair].values())))
        return data

    def get_distributions(self, bins, sigma=0.25, pair=None, member=None):
        num_bins = len(bins)
        bin_width = bins[1] - bins[0]

        samples = self.get_samples(pair, member)

        if pair:
            dist = gaussian_smoothing(data=samples,
                                      sigma=sigma,
                                      num_bins=num_bins,
                                      bin_width=bin_width)
            dist /= np.sum(dist)
        else:
            dist = {}
            for pair in self.pairs:
                dist[pair] = gaussian_smoothing(data=samples[pair],
                                                sigma=sigma,
                                                num_bins=num_bins,
                                                bin_width=bin_width)
                dist[pair] /= np.sum(dist[pair])
        return dist

    def re_sample(self, bins, sigma=0.25):
        re_sampled_mems = np.random.choice(self.members,
                                           self.num_members,
                                           replace=True)

        re_sampled = {}
        for pair in self.pairs:
            re_sampled[pair] = np.zeros(shape=(len(bins)))
            for mem in re_sampled_mems:
                re_sampled[pair] += self.get_distributions(bins,
                                                           sigma,
                                                           pair=pair,
                                                           member=mem)
            re_sampled[pair] /= np.sum(re_sampled[pair])
            # re_sampled[pair] = re_sampled[pair].tolist()

        return re_sampled
