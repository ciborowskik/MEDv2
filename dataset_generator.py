from os import path

from pandas import DataFrame
from sklearn.datasets import make_blobs

from test_batch_settings import *


def create_file_name(observations_count, features_count, clusters_count):
    return f'{observations_count}_{features_count}_{clusters_count}.csv'


def generate_functional_tests_datasets():
    for features_count in FUNCTIONAL_TESTS_FEATURES_COUNT_VARIANTS:
        for clusters_count in FUNCTIONAL_TESTS_CLUSTERS_COUNT_VARIANTS:
            features, clusters = make_blobs(
                n_samples=FUNCTIONAL_TESTS_OBSERVATIONS_COUNT,
                n_features=features_count,
                centers=clusters_count,
                cluster_std=FUNCTIONAL_TESTS_CLUSTER_STD,
                shuffle=True)

            df = DataFrame(features)
            df[CLUSTERS_COLUMN_NAME] = clusters

            file_name = create_file_name(FUNCTIONAL_TESTS_OBSERVATIONS_COUNT, features_count, clusters_count)
            file_path = path.join(FUNCTIONAL_TESTS_DIRECTORY, file_name)

            df.to_csv(file_path, index=False)


def generate_performance_tests_datasets():
    for observations_count in PERFORMANCE_TESTS_OBSERVATIONS_VARIANTS:
        features, clusters = make_blobs(
            n_samples=observations_count,
            n_features=PERFORMANCE_TESTS_FEATURES_CONST,
            centers=PERFORMANCE_TESTS_CLUSTER_COUNT,
            cluster_std=PERFORMANCE_TESTS_CLUSTER_STD,
            shuffle=True)

        df = DataFrame(features)
        df[CLUSTERS_COLUMN_NAME] = clusters

        file_name = create_file_name(observations_count, PERFORMANCE_TESTS_FEATURES_CONST, PERFORMANCE_TESTS_CLUSTER_COUNT)
        file_path = path.join(OBSERVATIONS_PERFORMANCE_TESTS_DIRECTORY, file_name)

        df.to_csv(file_path, index=False)

    for features_count in PERFORMANCE_TESTS_FEATURES_VARIANTS:
        features, clusters = make_blobs(
            n_samples=PERFORMANCE_TESTS_OBSERVATIONS_CONST,
            n_features=features_count,
            centers=PERFORMANCE_TESTS_CLUSTER_COUNT,
            cluster_std=PERFORMANCE_TESTS_CLUSTER_STD,
            shuffle=True)

        df = DataFrame(features)
        df[CLUSTERS_COLUMN_NAME] = clusters

        file_name = create_file_name(PERFORMANCE_TESTS_OBSERVATIONS_CONST, features_count, PERFORMANCE_TESTS_CLUSTER_COUNT)
        file_path = path.join(FEATURES_PERFORMANCE_TESTS_DIRECTORY, file_name)

        df.to_csv(file_path, index=False)


if __name__ == '__main__':
    generate_functional_tests_datasets()
    generate_performance_tests_datasets()
