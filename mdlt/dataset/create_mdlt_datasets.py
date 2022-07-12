import numpy as np
import pandas as pd
import argparse
import os
from os.path import join
import matplotlib.pyplot as plt


DATASETS = {
    'VLCS': 'VLCS',
    'PACS': 'PACS',
    'OfficeHome': 'office_home',
    'TerraIncognita': 'terra_incognita',
    'DomainNet': 'domain_net',
    'SVIRO': 'sviro',
}
NUM_SAMPLES_PER_CLASS = {
    # (num_test, num_val) tuple, by default num_val is set to half of num_test
    'VLCS': (30, 15),
    'PACS': (50, 25),
    'OfficeHome': (10, 5),
    'TerraIncognita': (20, 10),
    'DomainNet': (40, 20),
    'SVIRO': (100, 50),
}
ALPHA = 0.8


def visualize_dataset(dataset, verbose=False):
    file_path = join(args.output_dir, "temp", f"{dataset}.csv")
    df = pd.read_csv(file_path)
    domain_names = pd.unique(df['env'])
    num_domain = len(domain_names)
    num_class = len(pd.unique(df['label']))

    _, ax = plt.subplots(num_domain, 1, figsize=(10, 24), sharex='all', sharey='all',
                         gridspec_kw={'left': 0.08, 'right': 0.96, 'bottom': 0.04, 'top': 0.92, 'wspace': 0.1})
    plt.suptitle(f"Dataset: {dataset.upper()}", fontsize=24)
    for i, env_name in enumerate(domain_names):
        hist_targets = df[df['env'] == env_name].label
        ax[i].hist(hist_targets, range(num_class + 1), width=0.9, alpha=ALPHA, align='left')
        min_num_per_cls = 1e10
        for rect in ax[i].patches:
            height = rect.get_height()
            ax[i].annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 1), textcoords='offset points', ha='center', va='bottom', fontsize=15)
            min_num_per_cls = min(min_num_per_cls, height)
        if verbose:
            if i == 0:
                print(f"\n{dataset}:")
            print(f"\t[{env_name}] minimum #sample:\t{int(min_num_per_cls)}")
    for i in range(num_domain):
        ax[i].set_title(f"{domain_names[i]}", fontsize=18)
        ax[i].set_xticks(range(num_class))
        ax[i].tick_params(axis="x", labelsize=18)
        ax[i].tick_params(axis="y", labelsize=18)
    plt.show()


def make_balanced_testset(dataset, seed=666, verbose=True, vis=True, save=False):
    import random
    random.seed(seed)
    file_path = join(args.output_dir, "temp", f"{dataset}.csv")
    df = pd.read_csv(file_path)
    df['label'] = df.label.astype(int)
    domain_names = pd.unique(df['env'])
    val_set, test_set = [], []
    for i, env_name in enumerate(domain_names):
        df_env = df[df['env'] == env_name]
        for target in pd.unique(df['label']):
            curr_df = df_env[df_env['label'] == target]
            curr_data = curr_df['path'].values
            random.shuffle(curr_data)
            split_size = min(len(curr_data), int(np.sum(NUM_SAMPLES_PER_CLASS[dataset])))
            val_set += list(curr_data[:split_size // 3])
            test_set += list(curr_data[split_size // 3:split_size])
    if verbose:
        print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
    assert len(set(val_set).intersection(set(test_set))) == 0
    combined_set = dict(zip(val_set, ['val' for _ in range(len(val_set))]))
    combined_set.update(dict(zip(test_set, ['test' for _ in range(len(test_set))])))
    df['split'] = df['path'].map(combined_set)
    df['split'].fillna('train', inplace=True)
    if verbose:
        print(df)
    if save:
        df.to_csv(str(join(args.output_dir, f"{dataset}.csv")), index=False)
    if vis:
        num_domain = len(domain_names)
        num_class = len(pd.unique(df['label']))
        _, ax = plt.subplots(num_domain, 3, figsize=(20, 24), sharex='all', sharey='all',
                             gridspec_kw={'left': 0.04, 'right': 0.98, 'bottom': 0.03, 'top': 0.95, 'wspace': 0.1})
        plt.suptitle(f"Dataset: {dataset.upper()}", fontsize=24)
        for i, env_name in enumerate(domain_names):
            df_env = df[df['env'] == env_name]
            hist_train = df_env[df_env['split'] == 'train'].label
            hist_val = df_env[df_env['split'] == 'val'].label
            hist_test = df_env[df_env['split'] == 'test'].label
            ax[i, 0].hist(hist_train, range(num_class + 1), width=0.9, alpha=ALPHA, align='left')
            ax[i, 1].hist(hist_val, range(num_class + 1), width=0.9, alpha=ALPHA, align='left')
            ax[i, 2].hist(hist_test, range(num_class + 1), width=0.9, alpha=ALPHA, align='left')
            for j in range(3):
                for rect in ax[i, j].patches:
                    height = rect.get_height()
                    ax[i, j].annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, height),
                                      xytext=(0, 1), textcoords='offset points', ha='center', va='bottom', fontsize=15)
        mapping = {0: 'train', 1: 'val', 2: 'test'}
        for i in range(num_domain):
            for j in range(3):
                ax[i, j].set_title(f"{domain_names[i]} ({mapping[j]})", fontsize=18)
                ax[i, j].set_xticks(range(num_class))
                ax[i, j].tick_params(axis="x", labelsize=18)
                ax[i, j].tick_params(axis="y", labelsize=18)
        plt.show()


def create(dataset, save_temp=False, vis=False, make_balanced=False, save_final=False):
    dataset_path = join(args.data_dir, DATASETS[dataset])
    env_ids, labels, img_paths = [], [], []

    for env in sorted(entry.name for entry in os.scandir(dataset_path) if entry.is_dir()):
        for label, class_name in enumerate(sorted(os.listdir(join(dataset_path, env)))):
            for img_name in os.listdir(join(dataset_path, env, class_name)):
                env_ids.append(env)
                labels.append(label)
                img_paths.append(join(env, class_name, img_name))

    outputs = dict(env=env_ids, label=labels, path=img_paths)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = join(args.output_dir, "temp", f"{dataset}.csv")
    df = pd.DataFrame(data=outputs)
    if save_temp:
        df.to_csv(str(output_path), index=False)

    if vis:
        visualize_dataset(dataset)
    if make_balanced:
        make_balanced_testset(dataset, save=save_final, vis=True)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', nargs='+', type=str, default=DATASETS)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    for dset in args.dataset:
        # visualize_dataset(dset)
        create(dset, save_temp=False, vis=False, make_balanced=True, save_final=False)
