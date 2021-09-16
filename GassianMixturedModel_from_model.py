import matplotlib.pyplot as plt
from numpy import random
from numpy.core.function_base import _needs_add_docstring
from scipy.sparse.coo import coo_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel

sns.set()

# Generate some data
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

'''
# plot the data with K Means Labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=2021)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis")
plt.show()
'''

from sklearn.cluster import KMeans
from scipy.spatial.distance import _nbool_correspond_all, cdist
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:,0], X[:,1], c=labels, s=40, cmap='viridis', zorder=2)
    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels==i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc="#CCCCCC", lw=3, alpha=0.5, zorder=1))

# kmeans = KMeans(n_clusters=4, random_state=2021)
# plot_kmeans(kmeans, X)
# plt.show()

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2,2))
# kmeans = KMeans(n_clusters=4, random_state=2021)
# plot_kmeans(kmeans, X_stretched)
# plt.show()

# Gassian Mixtured Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
# plt.show()

probs = gmm.predict_proba(X) # [n_samples, n_clusters]: 每个点属于给定类别的概率
# print(probs)

# size = 50 * probs.max(1) ** 2  # square emphasizes differences
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=size)
# plt.show()

# 基于高斯混合模型的分类划分标识
from matplotlib.patches import Ellipse

def draw_ellipse(possition, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(possition, nsig*width, nsig*height, angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    ax.axis('equal')
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)

    w_factor = 0.2 / gmm.weights_.max()

    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        print(covar)
        draw_ellipse(pos, covar, alpha=w * w_factor)

# gmm = GaussianMixture(n_components=4, random_state=2021)
# plot_gmm(gmm, X)
# plt.show()

# gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=2021)
# plot_gmm(gmm, X_stretched)
# plt.show()


# data preparation
# m1 = np.array([1, 3])
# m2 = np.array([3, 0])
# C1 = np.array([[3, 0], [0, 3]], np.float)
# C2 = np.array([[0.5, 0], [0, 0.5]], np.float)
# N = 200

# data1 = np.random.randn(N, 2)
# data2 = np.random.randn(N, 2)
# A1 = np.linalg.cholesky(C1)
# A2 = np.linalg.cholesky(C2)

# new_data1 = data1 @ A1.T + m1
# new_data2 = data2 @ A2.T + m2

# X = np.concatenate([new_data1, new_data2], axis=0)
# gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=2021)
# plot_gmm(gmm, X)
# plt.show()

# new data from biotechnology
import pandas as pd
dat = pd.read_table("CorData.txt", header=0, index_col=0, sep="\t")
cor = dat.iloc[:, [0, 2]].dropna(axis=0, how="any").values

# n_components = np.arange(1, 5)
# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(cor)
#           for n in n_components]

# plt.plot(n_components, [m.bic(cor) for m in models], label='BIC')
# plt.plot(n_components, [m.aic(cor) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')
# plt.show()

mn1 = np.mean(np.array([d[0] for d in cor]))
mn2 = np.mean(np.array([d[1] for d in cor]))

print(mn1)
print(mn2)

gmm = GaussianMixture(n_components=2, covariance_type="spherical", init_params="random", random_state=2021)
plot_gmm(gmm, cor)
plt.show()
# clf = gmm.fit(cor)
# print(clf.weights_)
# print(clf.means_)