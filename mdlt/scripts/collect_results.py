import argparse
import json
import os
import random
import sys
import numpy as np

from mdlt.dataset import datasets
from mdlt.learning import algorithms, model_selection
from mdlt.utils import misc, reporting
from mdlt.utils.query import Q


def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, f"{mean:.1f} $\\pm$ {err:.1f}"
    else:
        return mean, err, f"{mean:.1f} +/- {err:.1f}"


def print_table(table, header_text, row_labels, col_labels, colwidth=10, latex=True):
    """Pretty-print a 2D array of dataset, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}" for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")


def print_results_tables(records, selection_method, latex):
    grouped_records = reporting.get_grouped_records(records).map(
        lambda group: {**group,
                       "sweep_acc": selection_method.sweep_acc(group["records"]),
                       "sweep_acc_worst": selection_method.sweep_acc_worst(group["records"]),
                       "sweep_acc_env": selection_method.sweep_acc_env(group["records"]),
                       "sweep_acc_shot": selection_method.sweep_acc_shot(group["records"])}
    ).filter(lambda g: g["sweep_acc"] is not None)

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
                 [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    # print a summary table for each dataset
    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*envs, "Avg", "Worst", "Many", "Medium", "Few", "Zero"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            # env_0, env_1, ..., env_n
            trial_accs = (grouped_records.filter_equals(
                "dataset, algorithm", (dataset, algorithm)
            ).select("sweep_acc_env"))
            for env in envs:
                mean, err, table[i][env] = format_mean([x[env] for x in trial_accs], latex)
            # avg
            trial_accs = (grouped_records.filter_equals(
                "dataset, algorithm", (dataset, algorithm)
            ).select("sweep_acc"))
            _, _, table[i][env + 1] = format_mean(trial_accs, latex)
            # worst
            trial_accs = (grouped_records.filter_equals(
                "dataset, algorithm", (dataset, algorithm)
            ).select("sweep_acc_worst"))
            _, _, table[i][env + 2] = format_mean(trial_accs, latex)
            # shot-wise
            trial_accs = (grouped_records.filter_equals(
                "dataset, algorithm", (dataset, algorithm)
            ).select("sweep_acc_shot"))
            for s in range(4):
                _, _, table[i][env + 3 + s] = format_mean([x[s] for x in trial_accs], latex)

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg", "Worst", "Many", "Medium", "Few", "Zero"
        ]
        header_text = (f"Dataset: {dataset}, "
                       f"model selection method: {selection_method.name}")
        print_table(table, header_text, alg_names, list(col_labels), colwidth=20 if latex else 15, latex=latex)

    # print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (
                grouped_records.filter_equals(
                    "algorithm, dataset", (algorithm, dataset)
                ).group("seed").map(
                    lambda trial_seed, group: group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = f"{sum(means) / len(means):.1f}"

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25 if latex else 20, latex=latex)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full MDLT results}")
        print("% Total records:", len(records))
    else:
        print(f"Total records: [{len(records)}]")

    SELECTION_METHODS = [
        model_selection.ValMeanAccSelectionMethod
    ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex)

    if args.latex:
        print("\\end{document}")
