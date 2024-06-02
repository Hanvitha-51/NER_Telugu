import time
import numpy as np

class ProgressTracker:
    def __init__(self, target_steps, verbosity=1):
        self.target_steps = target_steps
        self.values_dict = {}
        self.unique_keys = []
        self.start_time = time.time()
        self.current_step = 0
        self.verbosity = verbosity

    def update(self, current_step, value_pairs=[], exact_values=[], strict_values=[]):
        self._update_values(current_step, value_pairs, exact_values, strict_values)
        self.current_step = current_step

    def increment(self, steps, value_pairs=[]):
        self.update(self.current_step + steps, value_pairs)

    def _update_values(self, current_step, value_pairs, exact_values, strict_values):
        for key, value in value_pairs:
            if key not in self.values_dict:
                self.values_dict[key] = [value * (current_step - self.current_step), current_step - self.current_step]
                self.unique_keys.append(key)
            else:
                self.values_dict[key][0] += value * (current_step - self.current_step)
                self.values_dict[key][1] += (current_step - self.current_step)
        
        for key, value in exact_values + strict_values:
            if key not in self.values_dict:
                self.unique_keys.append(key)
            self.values_dict[key] = [value, 1] if key in exact_values else value

    def print_final_info(self):
        current_time = time.time()
        info = f'Total Time: {int(current_time - self.start_time)}s'
        for key in self.unique_keys:
            if isinstance(self.values_dict[key], list):
                info += f' - {key}: {self.values_dict[key][0] / max(1, self.values_dict[key][1]):.4f}'
            else:
                info += f' - {key}: {self.values_dict[key]}'
        print(info)

# Example use:
tracker = ProgressTracker(target_steps=100, verbosity=1)

for step in range(100):
    time.sleep(0.1)
    loss = np.random.random()
    accuracy = np.random.random()
    tracker.update(step + 1, value_pairs=[('loss', loss), ('accuracy', accuracy)])

tracker.print_final_info()
