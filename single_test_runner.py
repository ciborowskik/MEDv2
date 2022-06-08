import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.datasets import make_blobs
from sklearn.metrics import fowlkes_mallows_score

from my_clustering import my_python_clustering
from scipy_clustering import scipy_clustering
from test_batch_settings import CLUSTERS_COLUMN_NAME


def plot(data, clusters, title):
    df = DataFrame(data)
    df[CLUSTERS_COLUMN_NAME] = clusters
    groups = df.groupby(CLUSTERS_COLUMN_NAME)
    df.drop(CLUSTERS_COLUMN_NAME, inplace=True, axis=1)

    if len(df.columns) == 2:
        for name, group in groups:
            plt.plot(group[0], group[1], marker='o', linestyle='')

    if len(df.columns) == 3:
        fig3d = plt.axes(projection='3d')
        for name, group in groups:
            fig3d.scatter3D(group[0], group[1], group[2])

    plt.title(title)
    plt.show()


if __name__ == '__main__':
    features, original_cluster_ids = make_blobs(
        n_samples=200,
        n_features=3,
        centers=5,
        cluster_std=0.5,
        shuffle=True)

    #features = np.array([[0, 0], [0, 1], [1, 0],
    #              [0, 4], [0, 3], [1, 4],
    #              [4, 0], [3, 0], [4, 1],
    #              [4, 4], [3, 4], [4, 3]])

    my_cluster_ids = my_python_clustering(features, 5, 'single', 'euclidean')
    scipy_cluster_ids = scipy_clustering(features, 5, 'single', 'euclidean')
    # print(cluster_ids)

    plot(features, original_cluster_ids, 'Original clusters')
    plot(features, scipy_cluster_ids, 'Scipy clusters')
    plot(features, my_cluster_ids, 'My clusters')

    print(f'Scipy Fowlkes–Mallows index: {fowlkes_mallows_score(scipy_cluster_ids, my_cluster_ids)}')
    print(f'My Fowlkes–Mallows index: {fowlkes_mallows_score(scipy_cluster_ids, my_cluster_ids)}')
