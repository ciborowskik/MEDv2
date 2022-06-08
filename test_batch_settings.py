from my_clustering import my_python_clustering, my_cython_clustering
from scipy_clustering import scipy_clustering

FUNCTIONAL_TESTS_DIRECTORY = 'datasets/functional_tests'

PERFORMANCE_TESTS_DIRECTORY = 'datasets/performance_tests'
FEATURES_PERFORMANCE_TESTS_DIRECTORY = 'datasets/performance_tests/features'
OBSERVATIONS_PERFORMANCE_TESTS_DIRECTORY = 'datasets/performance_tests/observations'

CLUSTERS_COLUMN_NAME = 'Cluster'

FUNCTIONAL_TESTS_CLUSTER_STD = 0.5
FUNCTIONAL_TESTS_OBSERVATIONS_COUNT = 200
FUNCTIONAL_TESTS_FEATURES_COUNT_VARIANTS = [2, 3]
FUNCTIONAL_TESTS_CLUSTERS_COUNT_VARIANTS = [2, 3, 4, 5, 6]

PERFORMANCE_TESTS_CLUSTER_STD = 1
PERFORMANCE_TESTS_CLUSTER_COUNT = 5
PERFORMANCE_TESTS_OBSERVATIONS_CONST = 1000
PERFORMANCE_TESTS_FEATURES_VARIANTS = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PERFORMANCE_TESTS_FEATURES_CONST = 10
PERFORMANCE_TESTS_OBSERVATIONS_VARIANTS = [10, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

EXPERIMENT_VARIANTS = [
    {'method': 'single', 'metric': 'euclidean'},
    {'method': 'single', 'metric': 'seuclidean'},
    {'method': 'single', 'metric': 'chebyshev'},
    {'method': 'single', 'metric': 'cityblock'},
    {'method': 'complete', 'metric': 'euclidean'},
    {'method': 'complete', 'metric': 'seuclidean'},
    {'method': 'complete', 'metric': 'chebyshev'},
    {'method': 'complete', 'metric': 'cityblock'},
    {'method': 'average', 'metric': 'euclidean'},
    {'method': 'average', 'metric': 'seuclidean'},
    {'method': 'average', 'metric': 'chebyshev'},
    {'method': 'average', 'metric': 'cityblock'},
    {'method': 'ward', 'metric': 'euclidean'},
]

FEATURES_EXPERIMENTS_VARIANTS = [
    experiment for experiment in EXPERIMENT_VARIANTS if experiment['method'] == 'single'
]

OBSERVATIONS_EXPERIMENTS_VARIANTS = [
    experiment for experiment in EXPERIMENT_VARIANTS if experiment['metric'] == 'euclidean'
]

IMPLEMENTATIONS_FOR_FUNCTIONAL_TESTS = [
    {'name': 'SciPy clustering', 'callable': scipy_clustering},
    {'name': 'Own Python clustering', 'callable': my_python_clustering}
]

IMPLEMENTATIONS_FOR_FEATURES_PERFORMANCE_TESTS = [
    {'name': 'SciPy clustering', 'callable': scipy_clustering},
    {'name': 'Own Python clustering', 'callable': my_python_clustering}
]

IMPLEMENTATIONS_FOR_OBSERVATIONS_PERFORMANCE_TESTS = [
    {'name': 'SciPy clustering', 'callable': scipy_clustering},
    {'name': 'Own Python clustering', 'callable': my_python_clustering},
    {'name': 'Own Cython clustering', 'callable': my_cython_clustering}
]

PROP_DATASET_NAME = 'Dataset name'
PROP_OBSERVATIONS_COUNT = 'Observations count'
PROP_FEATURES_COUNT = 'Features count'
PROP_CLUSTERS_COUNT = 'Cluster count'
