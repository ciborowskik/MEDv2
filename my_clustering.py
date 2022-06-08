import math

import numpy as np
from scipy.spatial.distance import pdist

from cython_dist_merge_functions import _single_dist_merge, _complete_dist_merge, _average_dist_merge, _ward_dist_merge


# calculate condensed index from matrix index
def condensed_index(n, i, j):
    if i < j:
        return int(n * i - (i * (i + 1) / 2) + (j - i - 1))
    elif i > j:
        return int(n * j - (j * (j + 1) / 2) + (i - j - 1))


# calculate matrix index from condensed index
def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2


def matrix_index(k, n):
    i = int(math.ceil((1/2.) * (- (-8*k + 4 * n**2 - 4*n - 7)**0.5 + 2*n - 1) - 1))
    j = int(n - elem_in_i_rows(i + 1, n) + k)

    return i, j


# dist merge functions
def single_dist_merge(dist_xz, dist_yz, dist_xy, size_z, size_x, size_y):
    return min(dist_xz, dist_yz)


def complete_dist_merge(dist_xz, dist_yz, dist_xy, size_z, size_x, size_y):
    return max(dist_xz, dist_yz)


def average_dist_merge(dist_xz, dist_yz, dist_xy, size_z, size_x, size_y):
    return (size_x * dist_xz + size_y * dist_yz) / (size_x + size_y)


def ward_dist_merge(dist_xz, dist_yz, dist_xy, size_z, size_x, size_y):
    t = 1.0 / (size_x + size_y + size_z)

    return math.sqrt((size_z + size_x) * t * dist_xz * dist_xz +
                     (size_z + size_y) * t * dist_yz * dist_yz -
                     size_z * t * dist_xy * dist_xy)


# normalize vectors of clusters ids
def normalize_cluster_ids(cluster_ids):
    unique = np.unique(cluster_ids)

    for i in range(len(unique)):
        cluster_ids[cluster_ids == unique[i]] = i


cython_dist_merge_functions = {
    'single': _single_dist_merge,
    'complete': _complete_dist_merge,
    'average': _average_dist_merge,
    'ward': _ward_dist_merge
}

python_dist_merge_functions = {
    'single': single_dist_merge,
    'complete': complete_dist_merge,
    'average': average_dist_merge,
    'ward': ward_dist_merge
}

max_double = np.finfo('d').max


# clustering implementation
def my_python_clustering(data, clusters_count, method, metric, use_cython=False):
    dist_merge_function = cython_dist_merge_functions[method] if use_cython else python_dist_merge_functions[method]

    n = data.shape[0]

    cluster_ids = np.arange(n, dtype=int)
    cluster_sizes = np.ones(n, dtype=int)

    distances = pdist(data, metric)

    for i in range(n - clusters_count):
        current_min_id = np.argmin(distances)
        current_min = distances[current_min_id]
        x, y = matrix_index(current_min_id, n)

        size_x = cluster_sizes[x]
        size_y = cluster_sizes[y]

        cluster_sizes[x] = 0
        cluster_sizes[y] = size_x + size_y

        cluster_ids[cluster_ids == x] = y

        distances[condensed_index(n, y, x)] = max_double

        for z in range(n):
            if z != y:
                size_z = cluster_sizes[z]

                if size_z != 0:
                    id_zy = condensed_index(n, z, y)
                    id_zx = condensed_index(n, z, x)

                    distances[id_zy] = dist_merge_function(distances[id_zx], distances[id_zy], current_min, size_z, size_x, size_y)

                    distances[id_zx] = max_double

    normalize_cluster_ids(cluster_ids)

    return cluster_ids


def my_cython_clustering(data, clusters_count, method, metric):
    return my_python_clustering(data, clusters_count, method, metric, True)


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0],
                  [0, 4], [0, 3], [1, 4],
                  [4, 0], [3, 0], [4, 1],
                  [4, 4], [3, 4], [4, 3]])

    ids = my_python_clustering(X, 4, 'single', 'euclidean')
    print(ids)
