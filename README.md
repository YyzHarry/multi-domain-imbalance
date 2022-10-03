# Multi-Domain Long-Tailed Recognition (MDLT)

This repository contains the implementation for paper: [On Multi-Domain Long-Tailed Recognition, Imbalanced Domain Generalization and Beyond](https://arxiv.org/abs/2203.09513) (ECCV 2022).

It is also a (living) PyTorch suite containing benchmark datasets and algorithms for Multi-Domain Long-Tailed Recognition (MDLT). Currently we support [8 MDLT datasets](./mdlt/dataset/datasets.py) (3 synthetic + 5 real), as well as [~20 algorithms](./mdlt/learning/algorithms.py) that span different learning strategies. Feel free to send us a PR to add your algorithm / dataset for MDLT!
___

<div align="center">
    <img src="mdlt/assets/teaser.gif" width="750"><br>
<b>Multi-Domain Long-Tailed Recognition (MDLT)</b> aims to learn from multi-domain imbalanced data, address label imbalance, domain shift, and divergent label distributions across domains, and generalize to all domain-class pairs.
</div>

## MDLT: From Single- to Multi-Domain Imbalanced Learning
Existing studies on data imbalance focus on __single-domain__ settings, i.e., samples are from the same data distribution. However, natural data can originate from __distinct domains__, where a minority class in one domain could have abundant instances from other domains. We systematically investigate __Multi-Domain Long-Tailed Recognition (MDLT)__, which learns from multi-domain imbalanced data, addresses _label imbalance_, _domain shift_, and _divergent label distributions across domains_, and generalizes to all domain-class pairs.

We develop the _domain-class transferability graph_, and show that such transferability governs the success of learning in MDLT. We then propose _BoDA_, a theoretically grounded learning strategy that tracks the upper bound of transferability statistics, and ensures balanced alignment and calibration across imbalanced domain-class distributions.
We curate MDLT benchmark datasets based on widely-used multi-domain datasets, and benchmark ~20 algorithms that span different learning strategies for MDLT.


## Beyond MDLT: Domain Generalization under Data Imbalance
Further, as a byproduct, we demonstrate that _BoDA_ strengthens Domain Generalization (DG) algorithms, and consistently improves the results on DG benchmarks.
Note that all current standard DG benchmarks naturally exhibit heavy class imbalance within domains and label distributions shift across domains, confirming that data imbalance is an intrinsic problem in DG, but has yet been overlooked by past works.

The results shed light on how label imbalance can affect out-of-distribution generalization, and highlight the importance of integrating label imbalance into practical DG algorithm design.


## Getting Started

### Installation

#### Prerequisites
1. Download the original datasets, and place them in your `data_path`
```bash
python -m mdlt.scripts.download --data_dir <data_path>
```
2. Place the `.csv` files of train/val/test splits for each MDLT dataset (provided in `mdlt/dataset/split/`) in the corresponding dataset folder under your `data_path`

#### Dependencies
1. PyTorch (>=1.4, tested on 1.4 / 1.9)
2. pandas
3. TensorboardX


### Code Overview

#### Main Files
- [`train.py`](./mdlt/train.py): main training script
- [`sweep.py`](./mdlt/sweep.py): launch a sweep with all selected algorithms (provided in `mdlt/learning/algorithms.py`) on all real MDLT datasets (VLCS-MLT, PACS-MLT, OfficeHome-MLT, TerraInc-MLT, DomainNet-MLT)
- [`sweep_synthetic.py`](./mdlt/sweep_synthetic.py): launch a sweep with all selected algorithms on the synthetic MDLT dataset (Digits-MLT)
- [`collect_results.py`](./mdlt/scripts/collect_results.py): collect sweep results to automatically generate result tables (as in the paper)
- [`eval_best_hparam.py`](./mdlt/evaluate/eval_best_hparam.py) & [`eval_checkpoint.py`](./mdlt/evaluate/eval_checkpoint.py): scripts for evaluating trained models


#### Main Arguments
- __train.py__:
    - `--dataset`: name of chosen MDLT dataset
    - `--algorithm`: choose algorithm used for running
    - `--data_dir`: data path
    - `--output_dir`: output path
    - `--output_folder_name`: output folder name (under `output_dir`) for the current run
    - `--hparams_seed`: seed for different hyper-parameters
    - `--seed`: seed for different runs
    - `--selected_envs`: train on selected envs (only used for Digits-MLT)
    - `--imb_type` & `--imb_factor`: arguments for customized Digits-MLT label distributions
    - `--stage1_folder` & `--stage1_algo`: arguments for two-stage algorithms
- __sweep.py__:
    - `--n_hparams`: how many hparams to run for each <dataset, algorithm> pair
    - `--best_hp` & `--n_trials`: after sweeping hparams, fix best hparam and run trials with different seeds


### Usage

#### Train a single model
```bash
python -m mdlt.train --algorithm <algo> --dataset <dset> --output_folder_name <output_folder_name> --data_dir <data_path> --output_dir <output_path>
```

#### Train a model using 2-stage (second stage classifier learning)
```bash
python -m mdlt.train --algorithm CRT --dataset <dset> --output_folder_name <output_folder_name> --data_dir <data_path> --output_dir <output_path> --stage1_folder <stage1_model_folder> --stage1_algo <stage1_algo>
```

Note that for $\text{BoDA}_{r,c}$ the command is the same as above, with changes only on `stage1_algo` & `stage1_folder`

#### Train a model on Digits-MLT, with imbalance type all `Forward-LT` and imbalance ratio `0.01`
```bash
python -m mdlt.train --algorithm <algo> --dataset ImbalancedDigits \
       --imb_type eee \
       --imb_factor 0.01 \
       --selected_envs 1 2
```

Note that for Digits-MLT, we additionally provide `MNIST` as another domain.
To maintain the same setting as in paper (2 domains), you only need to set `selected_envs` to be `1 2` as above

#### Launch a sweep with different hparams
```bash
python -m mdlt.sweep launch --algorithms <...> --dataset <...> --n_hparams <num_of_hparams> --n_trials 1
```

#### Launch a sweep after fixing hparam with different seeds
```bash
python -m mdlt.sweep launch --algorithms <...> --dataset <...> --best_hp --input_folder <...> --n_trials <num_of_trials>
```

#### Collect the results of your sweep
```bash
python -m mdlt.scripts.collect_results --input_dir <...>
```

#### Evaluate the best hparam model for a <dataset, algo> pair
```bash
python -u -m mdlt.evaluate.eval_best_hparam --algorithm <...> --dataset <...> --data_dir <...> --output_dir <...> --folder_name <...>
```

#### Evaluate a trained checkpoint
```bash
python -u -m mdlt.evaluate.eval_checkpoint --algorithm <...> --dataset <...> --data_dir <...> --checkpoint <...>
```

### Reproduced Benchmarks and Model Zoo

|   Model   | VLCS-MLT | PACS-MLT | OfficeHome-MLT | TerraInc-MLT | DomainNet-MLT |
| :-------: | :-----: | :-------: | :---------: | :------: | :------: |
|  BoDA (r) |  76.9 / [model](https://drive.google.com/file/d/1b8zYbR4j7zFAWJWWKLOdoldBofp-jvl7/view?usp=sharing) | 97.0 / [model](https://drive.google.com/file/d/1xayaXsssv1TtL6JhoEirJRtxNUuSE3GU/view?usp=sharing) | 81.5 / [model](https://drive.google.com/file/d/1br7DKxmp5Ohk77gIovdWJVgcgSNmfadP/view?usp=sharing) | 78.6 / [model](https://drive.google.com/file/d/1w0VDiFfHi0pWf6-pYAjxEGBItIzNI5MK/view?usp=sharing) | 60.1 / [model](https://drive.google.com/file/d/1LvQJyv33r-Lpk7gK3WHABIaoURhz-GdN/view?usp=sharing) |
| BoDA (r,c)|  77.3 / [model](https://drive.google.com/file/d/1YF44Db7F6zfZvS1cKA46V5CFA-KAfj1j/view?usp=sharing) | 97.2 / [model](https://drive.google.com/file/d/1WbTg7uhfgAyKP36Vxut8YhHjFf49fXF_/view?usp=sharing) | 82.3 / [model](https://drive.google.com/file/d/1P6jjl2gI5EGfauV5LiOpv4iMzK8kln4_/view?usp=sharing) | 82.3 / [model](https://drive.google.com/file/d/1Le1B9LIq9uxivEoVcmHXO07PVLIWSag8/view?usp=sharing) | 61.7 / [model](https://drive.google.com/file/d/1OVFRvuyhgoVeAOaptJAoEruKupCJNwq1/view?usp=sharing) |


## Updates
- __[10/2022]__ Check out the [Oral talk video](https://youtu.be/oUky8S8d1_Y) (10 mins) for our ECCV paper.
- __[07/2022]__ We create a [Blog post](https://towardsdatascience.com/how-to-learn-imbalanced-data-arising-from-multiple-domains-7d0c0d6e3c17) for this work (version in Chinese is also [available here](https://zhuanlan.zhihu.com/p/539749541)). Check it out for more details!
- __[07/2022]__ Paper accepted to ECCV 2022. We have released the code and models.
- __[03/2022]__ [arXiv version](https://arxiv.org/abs/2203.09513) posted. The code is currently under cleaning. Please stay tuned for updates.


## Acknowledgements
This code is partly based on the open-source implementations from [DomainBed](https://github.com/facebookresearch/DomainBed).


## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{yang2022multi,
  title={On Multi-Domain Long-Tailed Recognition, Imbalanced Domain Generalization and Beyond},
  author={Yang, Yuzhe and Wang, Hao and Katabi, Dina},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```


## Contact
If you have any questions, feel free to contact us through email (yuzhe@mit.edu) or Github issues. Enjoy!
