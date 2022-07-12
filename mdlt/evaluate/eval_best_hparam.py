import argparse
import collections
import os
from os.path import join
import numpy as np
import torch.utils.data

from mdlt.dataset import datasets
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.dataset.fast_dataloader import FastDataLoader
from mdlt.learning import model_selection
from mdlt.utils import reporting


def load_records():
    records = reporting.load_records(join(args.output_dir, args.folder_name))
    if 'Imbalance' in args.dataset:
        records = reporting.get_imbalanced_grouped_records(records)
        records = records.filter(
            lambda r: r['dataset'] == args.dataset and
                      r['algorithm'] == args.algorithm and
                      r['imb_type'] == args.imb_type and
                      r['imb_factor'] == args.imb_factor
        )
    else:
        records = reporting.get_grouped_records(records)
        records = records.filter(
            lambda r: r['dataset'] == args.dataset and
                      r['algorithm'] == args.algorithm
        )
    selection_method = model_selection.ValMeanAccSelectionMethod

    assert len(records) == 1
    group = records[0]
    print(f"(trial) seed: {group['seed']}")
    sorted_hparams = selection_method.hparams_accs(group['records'])
    # 'sorted_hparams' sorted by 'val_acc'
    run_acc, best_hparam_records = sorted_hparams[0]
    print(f"\t{run_acc}")
    for r in best_hparam_records:
        assert(r['hparams'] == best_hparam_records[0]['hparams'])
    print("\t\thparams:")
    for k, v in sorted(best_hparam_records[0]['hparams'].items()):
        print('\t\t\t{}: {}'.format(k, v))
    print("\t\toutput_dirs:")
    output_dir = best_hparam_records.select('args.output_dir').unique()
    assert len(output_dir) == 1
    print(f"\t\t\t{output_dir[0]}")
    return run_acc, best_hparam_records[0]['hparams'], output_dir[0]


def validate(algorithm, dataset):
    algorithm.eval()
    test_splits = []
    for env in dataset:
        test_splits.append((env, None))

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in test_splits
    ]
    eval_weights = [None for _, weights in test_splits]
    eval_loader_names = [f'env{i}_test' for i in range(len(test_splits))]

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    class_acc_output = collections.defaultdict(list)
    env_acc_output = {}
    for name, loader, weights in sorted(evals, key=lambda x: x[0]):
        acc, shot_acc, class_acc = misc.accuracy(
            algorithm, loader, weights, [], many_shot_thr=100, few_shot_thr=20, device=device, class_shot_acc=True)
        if 'test' in name:
            class_acc_output[name.split('_')[0]] = list(class_acc)
            env_acc_output[name.split('_')[0]] = acc

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{np.mean(list(env_acc_output.values())):.3f}]\n\tworst:\t[{min(env_acc_output.values()):.3f}]")
    print("Class-wise accuracy:")
    for env in sorted(class_acc_output):
        print('\t[{}] overall {:.3f}, class-wise {}'.format(
            env, env_acc_output[env], (np.array2string(
                np.array(class_acc_output[env]), separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation using best hparams given a (algo, dataset, seed) pair')
    # related args
    parser.add_argument('--dataset', type=str, default="PACS", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--folder_name', type=str)
    # imbalance related
    parser.add_argument('--imb_type', type=str, default="eee")
    parser.add_argument('--imb_factor', type=float, default=0.1)
    # others
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Example usage:
    python -u -m mdlt.evaluate.eval_best_hparam --algorithm ERM --dataset VLCS \
    --data_dir ... --output_dir ... --folder_name ...
    """
    args = parse_args()
    run_acc, hparams, input_dir = load_records()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)
    else:
        raise NotImplementedError

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset), hparams)

    checkpoint_path = join(input_dir, 'model.best.pkl')
    assert os.path.isfile(checkpoint_path), f"No checkpoint found at '{checkpoint_path}'!"
    checkpoint = torch.load(checkpoint_path)
    algorithm.load_state_dict(checkpoint['model_dict'], strict=False)
    algorithm.to(device)
    print(f"===> Loaded checkpoint '{checkpoint_path}'")

    validate(algorithm, dataset)
