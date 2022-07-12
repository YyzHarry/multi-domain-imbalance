import argparse
import collections
import os
import numpy as np
import torch.utils.data

from mdlt.dataset import datasets
from mdlt.hparams_registry import default_hparams
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.dataset.fast_dataloader import FastDataLoader


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
        class_acc_output[name.split('_')[0]] = list(class_acc)
        env_acc_output[name.split('_')[0]] = acc

    print("\nTest accuracy:")
    print(f"\tmean:\t[{np.mean(list(env_acc_output.values())):.3f}]\n\tworst:\t[{min(env_acc_output.values()):.3f}]")
    print("Per-Env accuracy:")
    for env in sorted(class_acc_output):
        print(f'\t{env}:\t[{env_acc_output[env]:.3f}]')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation using a given checkpoint')
    parser.add_argument('--dataset', type=str, default="PACS", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="BoDA", choices=algorithms.ALGORITHMS)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data_dir', type=str, default="./data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Example usage:
    python -u -m mdlt.evaluate.eval_checkpoint --algorithm ERM --dataset VLCS --data_dir ... --checkpoint ...
    """
    args = parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams = default_hparams(args.algorithm, args.dataset)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)
    else:
        raise NotImplementedError

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes, len(dataset), hparams)

    assert os.path.isfile(args.checkpoint), f"No checkpoint found at '{args.checkpoint}'!"
    checkpoint = torch.load(args.checkpoint)
    algorithm.load_state_dict(checkpoint['model_dict'], strict=False)
    algorithm.to(device)
    print(f"===> Loaded checkpoint '{args.checkpoint}'")

    validate(algorithm, dataset)
