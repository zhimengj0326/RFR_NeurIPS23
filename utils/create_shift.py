import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
# from KDEpy import NaiveKDE


def create_shift(
    data,
    alpha=0.0,
    beta=1.0,
):
    """
    Creates covariate shift sampling of data into disjoint source and target set.
    Let \mu and \sigma be the mean and the standard deviation of the first principal component retrieved by PCA on the whole data.
    The target is randomly sampled based on a Gaussian with mean = \mu and standard deviation = \sigma.
    The source is randomly sampled based on a Gaussian with mean = \mu + alpha and standard devaition = \sigma / beta
    data: [m, n]
    alpha, beta: the parameter that distorts the gaussian used in sampling
                   according to the first principle component
    output: source indices, target indices, ratios based on kernel density estimation with bandwidth = kdebw and smoothed by eps
    """
    source_size = np.shape(data)[0]

    pca = PCA(n_components=2)
    pc2 = pca.fit_transform(data)
    pc = pc2[:, 0]
    pc = pc.reshape(-1, 1)

    pc_mean = np.mean(pc)
    pc_std = np.std(pc)

    sample_mean = pc_mean + alpha
    sample_std = pc_std / beta

    # sample according to the probs
    prob_s = norm.pdf(pc, loc=sample_mean, scale=sample_std)
    sum_s = np.sum(prob_s)
    prob_s = prob_s / sum_s

    source_ind = np.random.choice(
        range(source_size), size=source_size, replace=False, p=np.reshape(prob_s, (source_size))
    )

    return source_ind


# class KDEAdapter:
#     def __init__(self, kde=NaiveKDE(kernel="gaussian", bw=0.3)):
#         self._kde = kde

#     def fit(self, sample):
#         self._kde.fit(sample)
#         return self

#     def pdf(self, sample):
#         density = self._kde.evaluate(sample)
#         return density

#     def p(self, sample, eps=0):
#         density = self._kde.evaluate(sample)
#         return (density + eps) / np.sum(density + eps)