import json
import os
from time import time

import pandas as pd
from matplotlib import pyplot as plt
from PyPDF2 import PdfMerger
from sklearn.metrics.cluster import fowlkes_mallows_score

from test_batch_settings import *


def experiment_label(experiment, implementation):
    return f"{implementation['name']}, {experiment['method']}, {experiment['metric']}"


def get_files(directory, extension):
    files_paths = []

    for path, _, files in os.walk(directory):
        files_paths.extend([os.path.join(path, file) for file in files if file.endswith(f'.{extension}')])

    return files_paths


def create_excel(df, stats_file_path):
    writer = pd.ExcelWriter(stats_file_path, engine='xlsxwriter')

    add_table_to_excel(writer, df)

    writer.save()


def add_table_to_excel(writer, df):
    df.to_excel(writer, startrow=1, header=False, index=True)

    column_settings = [{'header': 'Sequence'}]
    for header in df.columns:
        column_settings.append({'header': header})

    (max_row, max_col) = df.shape
    worksheet = writer.sheets['Sheet1']
    worksheet.add_table(0, 0, max_row, max_col, {'columns': column_settings})
    worksheet.set_column(0, max_col, 25)
    worksheet.freeze_panes(0, 1)


def plot_cluster_points(df, clusters, path, experiment_name, show, save):
    title = f'Dataset {os.path.basename(path)} - {experiment_name}'
    path = f'{path}_{experiment_name}.pdf'

    df[CLUSTERS_COLUMN_NAME] = clusters
    groups = df.groupby(CLUSTERS_COLUMN_NAME)
    df.drop(CLUSTERS_COLUMN_NAME, inplace=True, axis=1)

    if len(df.columns) == 2:
        for name, group in groups:
            plt.plot(group['0'], group['1'], marker='.', linestyle='')

    if len(df.columns) == 3:
        fig3d = plt.axes(projection='3d')
        for name, group in groups:
            fig3d.scatter3D(group['0'], group['1'], group['2'])

    plt.title(title)

    if save:
        plt.savefig(path)

    if show:
        plt.show()

    plt.clf()

    return path


def run_functional_tests(directory):
    dataset_paths = get_files(directory, 'csv')
    stats = []

    for dataset_path in dataset_paths:
        path = os.path.splitext(dataset_path)[0]
        dataset_name = os.path.basename(path)

        metadata = dataset_name.split('_')
        observations_count = int(metadata[0])
        features_count = int(metadata[1])
        clusters_count = int(metadata[2])

        df = pd.read_csv(dataset_path)
        original_clusters = df.pop(CLUSTERS_COLUMN_NAME)
        data = df.to_numpy()

        plot_path = plot_cluster_points(df, original_clusters, path, 'original', False, True)

        dataset_stats = {
            PROP_DATASET_NAME: dataset_name,
            PROP_OBSERVATIONS_COUNT: observations_count,
            PROP_FEATURES_COUNT: features_count,
            PROP_CLUSTERS_COUNT: clusters_count
        }

        plot_paths = [plot_path]
        for experiment in EXPERIMENT_VARIANTS:
            for implementation in IMPLEMENTATIONS_FOR_FUNCTIONAL_TESTS:
                clusters = implementation['callable'](data, clusters_count, experiment['method'], experiment['metric'])

                dataset_stats[experiment_label(experiment, implementation)] = fowlkes_mallows_score(original_clusters, clusters)
                plot_path = plot_cluster_points(df, clusters, path, experiment_label(experiment, implementation), False, True)
                plot_paths.append(plot_path)

        merger = PdfMerger()
        for pdf in plot_paths:
            merger.append(pdf)

        merger.write(f'{path}.pdf')
        merger.close()

        for pdf in plot_paths:
            os.remove(pdf)

        stats.append(dataset_stats)

    df = pd.DataFrame(stats).set_index([PROP_DATASET_NAME])
    create_excel(df, os.path.join(directory, 'stats.xlsx'))


def run_performance_tests(directory, experiments, implementations):
    datasets_paths = get_files(directory, 'csv')

    for dataset_path in datasets_paths:
        path_without_extension = os.path.splitext(dataset_path)[0]
        dataset_name = os.path.basename(path_without_extension)

        metadata = dataset_name.split('_')
        observations_count = int(metadata[0])
        features_count = int(metadata[1])
        clusters_count = int(metadata[2])

        stats = {
            PROP_DATASET_NAME: dataset_name,
            PROP_OBSERVATIONS_COUNT: observations_count,
            PROP_FEATURES_COUNT: features_count,
            PROP_CLUSTERS_COUNT: clusters_count
        }

        df = pd.read_csv(dataset_path)
        df.drop(CLUSTERS_COLUMN_NAME, axis=1, inplace=True)
        data = df.to_numpy()

        for experiment in experiments:
            for implementation in implementations:
                start = time()
                implementation['callable'](data, clusters_count, experiment['method'], experiment['metric'])
                end = time()

                stats[experiment_label(experiment, implementation)] = round(end - start, 2)

        dict_path = f'{path_without_extension}.txt'
        json.dump(stats, open(dict_path, 'w'), indent=4)


def create_stats(directory, experiments, main_property, implementations):
    stats_paths = get_files(directory, 'txt')

    stats = [json.load(open(file_path)) for file_path in stats_paths]

    df = pd.DataFrame(stats).set_index([PROP_DATASET_NAME])
    df = df.sort_values(by=[main_property])

    create_excel(df, os.path.join(directory, 'stats.xlsx'))

    values = df[main_property]

    for experiment in experiments:
        for implementation in implementations:
            times = df[experiment_label(experiment, implementation)]

            plt.plot(values, times.tolist(), label=implementation['name'])

        plt.legend()
        plt.xlabel(main_property)
        plt.ylabel('Time [s]')

        file_name = f"{experiment['method']}-{experiment['metric']}.pdf"

        plt.savefig(f'{os.path.join(directory, file_name)}')
        plt.clf()


if __name__ == '__main__':
    run_functional_tests(FUNCTIONAL_TESTS_DIRECTORY)

    run_performance_tests(
        FEATURES_PERFORMANCE_TESTS_DIRECTORY,
        FEATURES_EXPERIMENTS_VARIANTS,
        IMPLEMENTATIONS_FOR_FEATURES_PERFORMANCE_TESTS)

    run_performance_tests(
        OBSERVATIONS_PERFORMANCE_TESTS_DIRECTORY,
        OBSERVATIONS_EXPERIMENTS_VARIANTS,
        IMPLEMENTATIONS_FOR_OBSERVATIONS_PERFORMANCE_TESTS)

    create_stats(
        FEATURES_PERFORMANCE_TESTS_DIRECTORY,
        FEATURES_EXPERIMENTS_VARIANTS,
        PROP_FEATURES_COUNT,
        IMPLEMENTATIONS_FOR_FEATURES_PERFORMANCE_TESTS)

    create_stats(
        OBSERVATIONS_PERFORMANCE_TESTS_DIRECTORY,
        OBSERVATIONS_EXPERIMENTS_VARIANTS,
        PROP_OBSERVATIONS_COUNT,
        IMPLEMENTATIONS_FOR_OBSERVATIONS_PERFORMANCE_TESTS)
