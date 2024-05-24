import numpy as np
import wandb



class Average:
    def __init__(self, split, name):
        self._split = split
        self._name = name
        self.values = []

    def add(self, values):
        if isinstance(values, list):
            self.values.extend(values)
        elif isinstance(values, np.ndarray):
            self.values.extend(values.tolist())
        else:
            self.values.append(values)

    # Returns dict {split/name_mean: value_mean, split/name_count: value_count}
    def get_dict(self, histogram=False):
        result = {
            f'{self._split}/{self._name}_mean': np.mean(self.values),
            f'{self._split}/{self._name}_count': len(self.values)
        }
        if histogram:
            result[f'{self._split}/{self._name}_histogram'] = \
                wandb.Histogram(self.values)
        self.values = []
        return result