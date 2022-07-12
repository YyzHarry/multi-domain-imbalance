import argparse
import copy
import os
import shutil
import numpy as np
import shlex
import tqdm

from mdlt import command_launchers
from mdlt.dataset import datasets
from mdlt.learning import algorithms
from mdlt.utils import misc
from mdlt.sweep import ask_for_confirmation


# For each sweep on imbalanced datasets, double check (modify) this function for customized need
def all_imb_combinations(n, candidates=('b', 'i', 'e')):
    """Generate all possible imb_type given imbalanced config"""
    if n == 1:
        return [i for i in candidates]
    elif n == 2:
        return [f"{i}{j}" for i in candidates for j in candidates]
    elif n == 3:
        return [f"{i}{j}{k}" for i in candidates for j in candidates for k in candidates]
    else:
        raise NotImplementedError(f"[{n}] envs not supported now.")


def make_imb_type_for_selected_envs(n_envs, select_envs):
    init_imb_types = all_imb_combinations(len(select_envs))
    final_imb_types = []
    for imb_type in init_imb_types:
        curr_type = "+" * n_envs
        for i, loc in enumerate(select_envs):
            curr_type = curr_type[:loc] + imb_type[i] + curr_type[loc + 1:]
        final_imb_types.append(curr_type)
    return final_imb_types


class ImbalanceJob:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args):
        self.output_dir = os.path.join(
            args.output_dir,
            args.output_folder_name,
            f"{train_args['dataset']}_{train_args['imb_type']}_{train_args['imb_factor']}_{train_args['algorithm']}"
            f"_hparams{train_args['hparams_seed']}_seed{train_args['seed']}"
        )
        if 'selected_envs' in train_args:
            self.output_dir += f"_env{str(train_args['selected_envs']).replace(' ', '')[1:-1]}"

        self.train_args = copy.deepcopy(train_args)
        command = ['python', '-m', 'mdlt.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = ImbalanceJob.DONE
        elif os.path.exists(self.output_dir):
            self.state = ImbalanceJob.INCOMPLETE
        else:
            self.state = ImbalanceJob.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
                    self.train_args['algorithm'],
                    self.train_args['hparams_seed'],
                    self.train_args['seed'])
        return f'{self.state}: {self.output_dir} {job_info}'

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def make_args_list(n_trials, dataset_names, imb_factor, algorithms, n_hparams_from, n_hparams, steps,
                   stage1_folder, stage1_algo, selected_envs, output_folder_name, hparams):
    args_list = []
    for trial_seed in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                imb_types = make_imb_type_for_selected_envs(datasets.num_environments(dataset), selected_envs) \
                    if selected_envs is not None else all_imb_combinations(datasets.num_environments(dataset))
                for imb_type in imb_types:
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['imb_type'] = imb_type
                        train_args['imb_factor'] = imb_factor
                        train_args['algorithm'] = algorithm
                        train_args['output_folder_name'] = output_folder_name
                        train_args['hparams_seed'] = hparams_seed
                        train_args['seed'] = trial_seed
                        if stage1_folder is not None:
                            train_args['stage1_folder'] = stage1_folder
                        if stage1_algo is not None:
                            train_args['stage1_algo'] = stage1_algo
                        if selected_envs is not None:
                            train_args['selected_envs'] = selected_envs
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep over synthetic MDLT datasets')
    # pass through commands / change here each run
    parser.add_argument('command', choices=['launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--command_launcher', type=str, default='multi_gpu')
    parser.add_argument('--output_folder_name', type=str, required=True)
    parser.add_argument('--dataset', nargs='+', type=str, default=["ImbalancedDigits"])
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--n_trials', type=int, default=1)
    # optional usage
    parser.add_argument('--selected_envs', type=int, nargs='+', default=None, help='Train only on selected envs')
    parser.add_argument('--stage1_folder', type=str, default=None)
    parser.add_argument('--stage1_algo', type=str, default=None)
    # default fixed
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.dataset,
        imb_factor=args.imb_factor,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        stage1_folder=args.stage1_folder,
        stage1_algo=args.stage1_algo,
        selected_envs=args.selected_envs,
        output_folder_name=args.output_folder_name,
        hparams=args.hparams
    )

    jobs = [ImbalanceJob(train_args) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == ImbalanceJob.DONE]),
        len([j for j in jobs if j.state == ImbalanceJob.INCOMPLETE]),
        len([j for j in jobs if j.state == ImbalanceJob.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == ImbalanceJob.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        ImbalanceJob.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == ImbalanceJob.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        ImbalanceJob.delete(to_delete)

    elif args.command == 'delete_all':
        to_delete = [j for j in jobs if j.state == ImbalanceJob.INCOMPLETE or j.state == ImbalanceJob.DONE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        ImbalanceJob.delete(to_delete)
