import numpy as np
from scipy.stats import entropy


def gaussian_smoothing(data, sigma, num_bins, bin_width):
    """
    Smooth discrete histogram according to method in DOI: 10.1021/jp3110369.
    """
    n_samples = len(data)

    norm = 1.0 / n_samples * np.sqrt(1.0 / (2 * np.pi * sigma**2))
    sim_hist = np.zeros(shape=num_bins)

    for n in range(num_bins):
        arg_exp = np.divide(
            np.square(np.subtract(n * bin_width, data)), 2 * sigma**2)
        sim_hist[n] = norm * np.sum(np.exp(-arg_exp), axis=0)
    return sim_hist.tolist()


def jensen_shannon(ps, qs):
    nbins = len(ps)
    if nbins != len(qs):
        raise IndexError(
            "The experimental num bins ({}) and simulation num bins ({}) are not equal".
            format(len(qs), nbins))

    # Normalization
    # ps = np.divide(ps, np.sum(ps))
    # qs = np.divide(qs, np.sum(qs))
    ms = np.multiply(np.add(ps, qs), 0.5)

    DPM = entropy(ps, ms)
    DQM = entropy(qs, ms)

    js = 0.5 * DPM + 0.5 * DQM
    return js
