from scipy.cluster.hierarchy import linkage, fcluster


def scipy_clustering(data, clusters_count, method, metric):
    linkage_matrix = linkage(data, method, metric)
    clusters = fcluster(linkage_matrix, clusters_count, criterion='maxclust')

    return clusters