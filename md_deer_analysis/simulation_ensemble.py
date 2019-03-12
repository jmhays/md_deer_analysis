import json
import numpy as np
from md_deer_analysis.utils import gaussian_smoothing


class SimulationEnsemble:
    def __init__(self, json_filename, name="myensemble"):
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
        self.num_bins = 0
        self.distributions = {}
        self.ensemble_average = {}

        self.__name = name

    def get_name(self):
        return self.__name

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

    def calculate_distributions(self, bins, sigma=0.25):

        num_bins = len(bins)
        self.num_bins = num_bins
        bin_width = bins[1] - bins[0]
        dist = {}

        for pair in self.pairs:
            dist[pair] = {}
            for member in self.members:
                dist[pair][member] = gaussian_smoothing(
                    self.get_samples(pair, member),
                    sigma=sigma,
                    num_bins=num_bins,
                    bin_width=bin_width
                )

            self.ensemble_average[pair] = gaussian_smoothing(
                self.get_samples(pair),
                sigma=sigma,
                num_bins=num_bins,
                bin_width=bin_width
            )

        self.distributions = dist

    def re_sample(self, exclude=[]):
        if not list(self.distributions.keys()):
            raise IndexError("Distributions have not yet been calculated. "
                             "Please calculate distributions for the ensemble before "
                             "running resampling")

        members = []
        for member in self.members:
            if member not in exclude:
                members.append(member)

        re_sampled_mems = np.random.choice(members,
                                           self.num_members,
                                           replace=True)

        re_sampled = {}
        for pair in self.pairs:
            re_sampled[pair] = np.zeros(shape=self.num_bins)
            for mem in re_sampled_mems:
                re_sampled[pair] += self.distributions[pair][mem]

            re_sampled[pair] /= np.sum(re_sampled[pair])
            # re_sampled[pair] = re_sampled[pair].tolist()

        return re_sampled
