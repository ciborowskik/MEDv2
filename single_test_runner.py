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


def run(data, clusters_count, method, metric, original_clusters):
    my_cluster_ids = my_python_clustering(data, clusters_count, method, metric)
    scipy_cluster_ids = scipy_clustering(data, clusters_count, method, metric)

    plot(data, original_clusters, 'Original clusters')
    plot(data, scipy_cluster_ids, 'Scipy clusters')
    plot(data, my_cluster_ids, 'My clusters')

    print(f'Scipy Fowlkes–Mallows index: {fowlkes_mallows_score(scipy_cluster_ids, my_cluster_ids)}')
    print(f'My Fowlkes–Mallows index: {fowlkes_mallows_score(scipy_cluster_ids, my_cluster_ids)}')


def generate_and_run(features_count, attributes_count, clusters_count, std, method, metric):
    features, original_cluster = make_blobs(
        n_samples=features_count,
        n_features=attributes_count,
        centers=clusters_count,
        cluster_std=std,
        shuffle=True)

    run(features, clusters_count, method, metric, original_cluster)


if __name__ == '__main__':
    generate_and_run(200, 3, 5, 0.5, 'single', 'euclidean')
