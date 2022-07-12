import argparse
import numpy as np
from mdlt.learning import model_selection
from mdlt.utils import reporting


if __name__ == "__main__":
    """Example usage:
    python -u -m mdlt.scripts.list_top_hparams --algorithm ERM --dataset VLCS --input_dir ...
    """

    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--algorithm', required=True)
    args = parser.parse_args()

    records = reporting.load_records(args.input_dir)
    print("Total records:", len(records))

    records = reporting.get_grouped_records(records)
    records = records.filter(
        lambda r: r['dataset'] == args.dataset and
                  r['algorithm'] == args.algorithm
    )

    SELECTION_METHODS = [
        model_selection.ValMeanAccSelectionMethod
    ]

    for selection_method in SELECTION_METHODS:
        print(f'Model selection: {selection_method.name}')

        for group in records:
            print(f"(trial) seed: {group['seed']}")
            best_hparams = selection_method.hparams_accs(group['records'])
            # 'best_hparams' sorted by 'val_acc'
            run_acc, hparam_records = best_hparams[0]
            print(f"\t{run_acc}")
            for r in hparam_records:
                assert(r['hparams'] == hparam_records[0]['hparams'])
            print("\t\thparams:")
            for k, v in sorted(hparam_records[0]['hparams'].items()):
                print('\t\t\t{}: {}'.format(k, v))
            print("\t\toutput_dirs:")
            output_dirs = hparam_records.select('args.output_dir').unique()
            for output_dir in output_dirs:
                print(f"\t\t\t{output_dir}")
