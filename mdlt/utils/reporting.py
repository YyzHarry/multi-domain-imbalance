import collections
import json
import os
import tqdm
from mdlt.utils.query import Q


def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))), ncols=80, leave=False):
        results_path = os.path.join(path, subdir, "results.json")
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
        except IOError:
            pass

    return Q(records)


def get_grouped_records(records):
    """Group records by (seed, dataset, algorithm)."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        group = (r["args"]["seed"],
                 r["args"]["dataset"],
                 r["args"]["algorithm"])
        result[group].append(r)
    return Q([{"seed": s, "dataset": d, "algorithm": a, "records": Q(r)}
              for (s, d, a), r in result.items()])


def get_imbalanced_grouped_records(records):
    """Group records by (seed, dataset, algorithm, imb_type, imb_factor)."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        group = (r["args"]["seed"],
                 r["args"]["dataset"],
                 r["args"]["algorithm"],
                 r["args"]["imb_type"],
                 r["args"]["imb_factor"])
        result[group].append(r)
    return Q([{"seed": s, "dataset": d, "algorithm": a, "imb_type": t, "imb_factor": f,
               "records": Q(r)} for (s, d, a, t, f), r in result.items()])
