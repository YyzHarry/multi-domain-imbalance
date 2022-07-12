import numpy as np
from mdlt.utils import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'ImbalancedColoredMNIST', 'ImbalancedRotatedMNIST', 'ImbalancedDigits']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    # nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, False])))

    if algorithm in ['ReSamp', 'CRT']:
        _hparam('class_balanced', True, lambda r: True)
    else:
        _hparam('class_balanced', False, lambda r: False)

    # Algorithm-specific hparam definitions
    # Each block of code below corresponds to exactly one algorithm

    if 'BoDA' in algorithm:
        if dataset == 'DomainNet':
            _hparam('boda_start_step', 3000, lambda r: int(1000 * r.choice(range(2, 6))))
            _hparam('feat_update_freq', 2000, lambda r: int(10 * r.uniform(150, 300)))
        else:
            _hparam('boda_start_step', 500, lambda r: int(100 * r.choice(range(3, 7))))
            _hparam('feat_update_freq', 120, lambda r: int(r.uniform(100, 150)))
        _hparam('nu', 0.5, lambda r: 10**r.uniform(-0.5, 0))
        _hparam('boda_weight', .1, lambda r: 10**r.uniform(-2, -0.5))
        _hparam('temperature', 1., lambda r: 10**r.uniform(-1, 0.5))
        _hparam('macro_weight', 1., lambda r: 10**r.uniform(-1, 1))
        _hparam('momentum', .2, lambda r: r.uniform(0, 0.4))

    elif algorithm == 'CBLoss':
        _hparam('beta', 0.9999, lambda r: 1 - 10**r.uniform(-5, -2))

    elif algorithm == 'Focal':
        _hparam('gamma', 1, lambda r: 0.5 * 10**r.uniform(0, 1))

    elif algorithm == 'LDAM':
        _hparam('max_m', 0.5, lambda r: 10**r.uniform(-1, -0.1))
        _hparam('scale', 30., lambda r: r.choice([10., 30.]))

    elif 'DANN' in algorithm:
        _hparam('lambda', 1.0, lambda r: 10**r.uniform(-2, 2))
        _hparam('weight_decay_d', 0., lambda r: 10**r.uniform(-6, -2))
        _hparam('d_steps_per_g_step', 1, lambda r: int(2**r.uniform(0, 3)))
        _hparam('grad_penalty', 0., lambda r: 10**r.uniform(-2, 1))
        _hparam('beta1', 0.5, lambda r: r.choice([0., 0.5]))
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    elif algorithm == 'Fish':
        _hparam('meta_lr', 0.5, lambda r: r.choice([0.05, 0.1, 0.5]))

    elif algorithm == "SagNet":
        _hparam('sag_w_adv', 0.1, lambda r: 10**r.uniform(-2, 1))

    elif algorithm == "IRM":
        _hparam('irm_lambda', 1e2, lambda r: 10**r.uniform(-1, 5))
        _hparam('irm_penalty_anneal_iters', 500, lambda r: int(10**r.uniform(0, 4)))

    elif "Mixup" in algorithm:
        _hparam('mixup_alpha', 0.2, lambda r: 10**r.uniform(-1, -1))

    elif "GroupDRO" in algorithm:
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))

    elif algorithm in ["MMD", "CORAL"]:
        _hparam('mmd_gamma', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MLDG":
        _hparam('mldg_beta', 1., lambda r: 10**r.uniform(-1, 1))

    elif algorithm == "MTL":
        _hparam('mtl_ema', .99, lambda r: r.choice([0.5, 0.9, 0.99, 1.]))

    # Dataset-and-algorithm-specific hparam definitions
    # Each block of code below corresponds to exactly one hparam. Avoid nested conditionals

    if dataset in SMALL_IMAGES:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    else:
        _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if dataset in SMALL_IMAGES:
        _hparam('weight_decay', 0., lambda r: 0.)
    else:
        _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -3))

    if dataset in SMALL_IMAGES:
        _hparam('batch_size', 64, lambda r: int(2**r.uniform(3, 9)))
    elif dataset == 'DomainNet':
        _hparam('batch_size', 32, lambda r: 32)
    else:
        _hparam('batch_size', 24, lambda r: 24)

    if 'DANN' in algorithm and dataset in SMALL_IMAGES:
        _hparam('lr_g', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif 'DANN' in algorithm:
        _hparam('lr_g', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if 'DANN' in algorithm and dataset in SMALL_IMAGES:
        _hparam('lr_d', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    elif 'DANN' in algorithm:
        _hparam('lr_d', 5e-5, lambda r: 10**r.uniform(-5, -3.5))

    if 'DANN' in algorithm and dataset in SMALL_IMAGES:
        _hparam('weight_decay_g', 0., lambda r: 0.)
    elif 'DANN' in algorithm:
        _hparam('weight_decay_g', 0., lambda r: 10**r.uniform(-6, -2))

    if 'CRT' in algorithm:
        _hparam('stage1_model', 'model.best.pkl', lambda r: 'model.best.pkl')

    # Imbalanced dataset(-and-algorithm)-specific hparam definitions
    # Each block of code below corresponds to one specific imbalanced dataset

    if dataset == 'ImbalancedColoredMNIST':
        _hparam('imb_type_per_env',
                ['balance', 'balance', 'balance', 'balance'],
                lambda r: ['balance', 'balance', 'balance', 'balance'])
        _hparam('imb_factor', 0.1, lambda r: 0.1)
        _hparam('rand_seed', 0, lambda r: 0)

    if dataset == 'ImbalancedRotatedMNIST':
        _hparam('imb_type_per_env',
                ['balance', 'balance', 'balance', 'balance', 'balance', 'balance'],
                lambda r: ['balance', 'balance', 'balance', 'balance', 'balance', 'balance'])
        _hparam('imb_factor', 0.1, lambda r: 0.1)
        _hparam('rand_seed', 0, lambda r: 0)

    if dataset == 'ImbalancedDigits':
        _hparam('imb_type_per_env', ['balance', 'balance', 'balance'], lambda r: ['balance', 'balance', 'balance'])
        _hparam('imb_factor', 0.1, lambda r: 0.1)
        _hparam('rand_seed', 0, lambda r: 0)

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
