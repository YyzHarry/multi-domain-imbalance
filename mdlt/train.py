import argparse
import collections
import json
import os
import random
import sys
import time
import shutil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from tensorboard_logger import Logger

from mdlt import hparams_registry
from mdlt.dataset import datasets
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Domain LT')
    # training
    parser.add_argument('--dataset', type=str, default="PACS", choices=datasets.DATASETS)
    parser.add_argument('--algorithm', type=str, default="ERM", choices=algorithms.ALGORITHMS)
    parser.add_argument('--output_folder_name', type=str, default='debug')
    # imbalance related
    parser.add_argument('--imb_type', type=str, default="eeee",
                        help='Length should be equal to # of envs, each refers to imb_type within that env')
    parser.add_argument('--imb_factor', type=float, default=0.1)
    # others
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for random hparams (0 for "default hparams")')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--selected_envs', type=int, nargs='+', default=None, help='Train only on selected envs')
    # two-stage related
    parser.add_argument('--stage1_folder', type=str, default='vanilla')
    parser.add_argument('--stage1_algo', type=str, default='ERM')
    # checkpoints
    parser.add_argument('--resume', '-r', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    start_step = 0
    args.best_val_acc = 0
    best_env_acc = {}
    best_shot_acc = {}
    best_class_acc = collections.defaultdict(list)
    store_prefix = f"{args.dataset}_{args.imb_type}_{args.imb_factor}" if 'Imbalance' in args.dataset else args.dataset
    args.store_name = f"{store_prefix}_{args.algorithm}_hparams{args.hparams_seed}_seed{args.seed}"
    if args.selected_envs is not None:
        args.store_name = f"{args.store_name}_env{str(args.selected_envs).replace(' ', '')[1:-1]}"

    misc.prepare_folders(args)
    args.output_dir = os.path.join(args.output_dir, args.output_folder_name, args.store_name)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    tb_logger = Logger(logdir=args.output_dir, flush_secs=2)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    if 'Imbalance' in args.dataset:
        hparams.update({'imb_type_per_env': [misc.IMBALANCE_TYPE[x] for x in args.imb_type],
                        'imb_factor': args.imb_factor})

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset in vars(datasets):
        train_dataset = vars(datasets)[args.dataset](args.data_dir, 'train', hparams)
        val_dataset = vars(datasets)[args.dataset](args.data_dir, 'val', hparams)
        test_dataset = vars(datasets)[args.dataset](args.data_dir, 'test', hparams)
    else:
        raise NotImplementedError

    num_workers = train_dataset.N_WORKERS
    input_shape = train_dataset.input_shape
    num_classes = train_dataset.num_classes
    n_steps = args.steps or train_dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or train_dataset.CHECKPOINT_FREQ
    many_shot_thr = train_dataset.MANY_SHOT_THRES
    few_shot_thr = train_dataset.FEW_SHOT_THRES

    if args.selected_envs is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, args.selected_envs)
        val_dataset = torch.utils.data.Subset(val_dataset, args.selected_envs)
        test_dataset = torch.utils.data.Subset(test_dataset, args.selected_envs)
    env_ids = args.selected_envs if args.selected_envs is not None else np.arange(len(train_dataset))

    print("Dataset:")
    for i, (tr, va, te) in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        print(f"\tenv{env_ids[i]}:\t{len(tr)}\t|\t{len(va)}\t|\t{len(te)}")

    # Split each env into train, val, test
    train_splits, val_splits, test_splits = [], [], []
    train_labels = dict()
    for i, env in enumerate(zip(train_dataset, val_dataset, test_dataset)):
        env_train, env_val, env_test = env
        if hparams['class_balanced']:
            train_weights = misc.make_balanced_weights_per_sample(
                env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy())
            val_weights = misc.make_balanced_weights_per_sample(
                env_val.targets if 'Imbalance' not in args.dataset else env_val.tensors[1].numpy())
            test_weights = misc.make_balanced_weights_per_sample(
                env_test.targets if 'Imbalance' not in args.dataset else env_test.tensors[1].numpy())
        else:
            train_weights, val_weights, test_weights = None, None, None
        train_splits.append((env_train, train_weights))
        val_splits.append((env_val, val_weights))
        test_splits.append((env_test, test_weights))
        train_labels[f"env{env_ids[i]}"] = env_train.targets if 'Imbalance' not in args.dataset else env_train.tensors[1].numpy()

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=num_workers)
        for env, env_weights in train_splits
    ]
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in (val_splits + test_splits)
    ]
    # loader for online training feature updates
    train_feat_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=num_workers)
        for env, _ in train_splits
    ] if 'BoDA' in args.algorithm else None
    eval_weights = [None for _, weights in (val_splits + test_splits)]
    eval_loader_names = [f'env{env_ids[i]}_val' for i in range(len(val_splits))]
    eval_loader_names += [f'env{env_ids[i]}_test' for i in range(len(test_splits))]
    feat_loader_names = [f'env{env_ids[i]}' for i in range(len(train_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(input_shape, num_classes, len(train_dataset), hparams, env_labels=train_labels)

    # load stage1 model if using 2-stage algorithm
    if 'CRT' in args.algorithm:
        args.pretrained = os.path.join(
            args.output_dir.replace(args.output_folder_name, args.stage1_folder), hparams['stage1_model']
        ).replace(args.algorithm, args.stage1_algo)
        args.pretrained = args.pretrained.replace(
            f"seed{args.pretrained[args.pretrained.find('seed') + len('seed')]}", 'seed0')
        assert os.path.isfile(args.pretrained)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_dict'].items():
            if 'classifier' not in k and 'network.1.' not in k:
                new_state_dict[k] = v
        algorithm.load_state_dict(new_state_dict, strict=False)
        print(f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]")
        print(f"===> Pre-trained model loaded: '{args.pretrained}'")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_step = checkpoint['start_step']
            args.best_val_acc = checkpoint['best_val_acc']
            algorithm.load_state_dict(checkpoint['model_dict'])
            print(f"===> Loaded checkpoint '{args.resume}' (step [{start_step}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in train_splits])

    def save_checkpoint(best=False, filename='model.pkl', curr_step=0):
        if args.skip_model_save:
            return
        filename = os.path.join(args.output_dir, filename)
        save_dict = {
            "args": vars(args),
            "best_val_acc": args.best_val_acc,
            "start_step": curr_step + 1,
            "num_classes": num_classes,
            "num_domains": len(train_dataset),
            "model_input_shape": input_shape,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, filename)
        if best:
            shutil.copyfile(filename, filename.replace('pkl', 'best.pkl'))

    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]

        # update features before training step
        train_features = {}
        if 'BoDA' in args.algorithm and (step > 0 and step % hparams["feat_update_freq"] == 0):
            curr_tr_feats, curr_tr_labels = collections.defaultdict(list), collections.defaultdict(list)
            for name, loader in sorted(zip(feat_loader_names, train_feat_loaders), key=lambda x: x[0]):
                algorithm.eval()
                with torch.no_grad():
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        feats = algorithm.return_feats(x)
                        curr_tr_feats[name].extend(feats.data)
                        curr_tr_labels[name].extend(y.data)
            train_features = {'feats': curr_tr_feats, 'labels': curr_tr_labels}

        algorithm.train()
        step_vals = algorithm.update(minibatches_device, train_features)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            class_acc_output = collections.defaultdict(list)
            shot_acc_output = collections.defaultdict(list)
            env_acc_output = {}
            for name, loader, weights in sorted(evals, key=lambda x: x[0]):
                if 'test' in name:
                    acc, shot_acc, class_acc = misc.accuracy(
                        algorithm, loader, weights, train_labels[name.split('_')[0]],
                        many_shot_thr, few_shot_thr, device, class_shot_acc=True)
                    class_acc_output[name.split('_')[0]] = list(class_acc)
                    env_acc_output[name.split('_')[0]] = acc
                    shot_acc_output['many'].extend(shot_acc[0])
                    shot_acc_output['median'].extend(shot_acc[1])
                    shot_acc_output['few'].extend(shot_acc[2])
                    shot_acc_output['zero'].extend(shot_acc[3])
                else:
                    acc = misc.accuracy(algorithm, loader, weights, train_labels[name.split('_')[0]],
                                        many_shot_thr, few_shot_thr, device, class_shot_acc=False)
                results[name] = acc

            # shot-wise results
            for shot in ['many', 'median', 'few', 'zero']:
                if len(shot_acc_output[shot]) == 0:
                    shot_acc_output[shot].append(-1)
                results[f"sht_{shot}"] = np.mean(shot_acc_output[shot])

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = list(results.keys())
            if results_keys != last_results_keys:
                print("\n")
                misc.print_row([key for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys if key not in {'mem_gb', 'step_time'}], colwidth=8)

            results.update({
                'hparams': hparams,
                'args': vars(args),
                'class_acc': class_acc_output
            })

            epochs_path = os.path.join(args.output_dir, 'results.json')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            # record best validation accuracy (mean over all envs)
            val_env_keys = [f'env{i}_val' for i in env_ids if f'env{i}_val' in results.keys()]
            val_acc_mean = np.mean([results[key] for key in val_env_keys])
            is_best = val_acc_mean > args.best_val_acc
            args.best_val_acc = max(val_acc_mean, args.best_val_acc)
            if is_best:
                best_class_acc = class_acc_output
                best_env_acc = env_acc_output
                best_shot_acc = {s: np.mean(shot_acc_output[s]) for s in ['many', 'median', 'few', 'zero']}

            save_checkpoint(best=is_best, curr_step=step)

            # tensorboard logger
            for key in checkpoint_vals.keys() - {'step_time'}:
                tb_logger.log_value(key, results[key], step)
            tb_logger.log_value('val_acc', val_acc_mean, step)
            tb_logger.log_value('test_acc_mean', np.mean(list(env_acc_output.values())), step)
            tb_logger.log_value('test_acc_worst', min(env_acc_output.values()), step)
            for i in env_ids:
                tb_logger.log_value(f'test_env{i}_acc', results[f"env{i}_test"], step)
            for s in ['many', 'median', 'few', 'zero']:
                tb_logger.log_value(f'shot_{s}', results[f"sht_{s}"], step)
            if hasattr(algorithm, 'optimizer'):
                tb_logger.log_value('learning_rate', algorithm.optimizer.param_groups[0]['lr'], step)

            checkpoint_vals = collections.defaultdict(lambda: [])

    print("\nTest accuracy (best validation checkpoint):")
    print(f"\tmean:\t[{np.mean(list(best_env_acc.values())):.3f}]\n\tworst:\t[{min(best_env_acc.values()):.3f}]")
    print("Shot-wise accuracy:")
    for s in ['many', 'median', 'few', 'zero']:
        print(f"\t[{s[:4]}]:\t[{best_shot_acc[s]:.3f}]")
    print("Class-wise accuracy:")
    for env in sorted(best_class_acc):
        print('\t[{}] overall {:.3f}, class-wise {}'.format(
            env, best_env_acc[env], (np.array2string(
                np.array(best_class_acc[env]), separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))))

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
