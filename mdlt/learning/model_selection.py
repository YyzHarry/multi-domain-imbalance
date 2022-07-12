import numpy as np


class SelectionMethod:
    """
    Abstract class whose subclasses implement strategies for model selection across hparams & steps
    """
    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(cls, run_records):
        """
        Given records from a run, return a {val_acc, test_acc, ...} dict representing
        the best val-acc, corresponding test-acc and other test metrics for that run.
        """
        raise NotImplementedError

    @classmethod
    def hparams_accs(cls, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed').map(
            lambda _, run_records: (
                cls.run_acc(run_records),
                run_records
            )
        ).filter(lambda x: x[0] is not None).sorted(key=lambda x: x[0]['val_acc'])[::-1])

    @classmethod
    def sweep_acc(cls, records):
        """
        Given all records from a single (dataset, algorithm) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

    @classmethod
    def sweep_acc_worst(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_worst']
        else:
            return None

    @classmethod
    def sweep_acc_env(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_env']
        else:
            return None

    @classmethod
    def sweep_acc_shot(cls, records):
        _hparams_accs = cls.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc_shot']
        else:
            return None


class OracleSelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_test_acc for env in train_envs))"""
    name = "test set mean accuracy (oracle)"
    max_num_envs = 20

    @classmethod
    def _step_acc(cls, record):
        """
        Given a single record, return a {val_acc, test_acc, ...} dict.
        Here val_acc == test_acc.
        """
        test_keys = []
        for i in range(cls.max_num_envs):
            if f'env{i}_val' not in record:
                continue
            test_keys.append(f'env{i}_test')
        return {
            'val_acc': np.mean([record[key] for key in test_keys]),
            'test_acc': np.mean([record[key] for key in test_keys]),
            'test_acc_worst': min([record[key] for key in test_keys]),
            'test_acc_env': [record[key] for key in test_keys],
            'test_acc_shot': [record.get("sht_many", -1), record.get("sht_median", -1),
                              record.get("sht_few", -1), record.get("sht_zero", -1)]
        }

    @classmethod
    def run_acc(cls, run_records):
        return run_records.map(cls._step_acc).argmax('test_acc')


class ValMeanAccSelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_val_acc for env in train_envs))"""
    name = "validation set mean accuracy"
    max_num_envs = 20

    @classmethod
    def _step_acc(cls, record):
        """Given a single record, return a {val_acc, test_acc, ...} dict."""
        val_keys, test_keys = [], []
        for i in range(cls.max_num_envs):
            if f'env{i}_val' not in record:
                continue
            val_keys.append(f'env{i}_val')
            test_keys.append(f'env{i}_test')
        return {
            'val_acc': np.mean([record[key] for key in val_keys]),
            'test_acc': np.mean([record[key] for key in test_keys]),
            'test_acc_worst': min([record[key] for key in test_keys]),
            'test_acc_env': [record[key] for key in test_keys],
            'test_acc_shot': [record.get("sht_many", -1), record.get("sht_median", -1),
                              record.get("sht_few", -1), record.get("sht_zero", -1)]
        }

    @classmethod
    def run_acc(cls, run_records):
        if not len(run_records):
            return None
        return run_records.map(cls._step_acc).argmax('val_acc')
