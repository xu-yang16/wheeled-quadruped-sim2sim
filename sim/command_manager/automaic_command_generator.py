import numpy as np


class AutomaticCommandGenerator:
    def __init__(self, loop=False):
        self.commands_lib = [
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            {"duration": 7, "vel": [1.4, 0, 0.0]},
            {"duration": 4, "vel": [0.0, 0.4, 0.0]},
            {"duration": 3, "vel": [0.0, 0.0, 0.5]},
            {"duration": 5, "vel": [0.0, 0.0, 0.0]},
            # {"duration": 5, "vel": [0.0, 0.0, 1.0]},
            # {"duration": 5, "vel": [0.0, 0.5, 0.0]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.2]},
            # {"duration": 5, "vel": [1.0, 0.0, 0.3]},
        ]
        self.loop = loop
        self._user_input_start_time = 0.0
        self._elapsed_time = 0.0
        self._current_commands = np.zeros(3)
        self._current_input_idx = 0

    def reset(self, sim_time):
        self._user_input_start_time = sim_time
        self._elapsed_time = 0.0
        self._current_commands = np.zeros(3)
        self._current_input_idx = 0

    def update(self, sim_time, transition_ratio=0.2):
        if self._current_input_idx > len(self.commands_lib) - 1:
            self._current_commands = np.zeros(3)
            return
        self._elapsed_time = sim_time - self._user_input_start_time
        duration = self.commands_lib[self._current_input_idx]["duration"]
        _current_commands = np.array(self.commands_lib[self._current_input_idx]["vel"])
        if self._elapsed_time >= duration:
            self._current_input_idx += 1
            if self._current_input_idx > len(self.commands_lib) - 1 and self.loop:
                self._current_input_idx = 0
            self._user_input_start_time = sim_time
            self._elapsed_time = 0.0

        if self._elapsed_time <= transition_ratio * duration:
            ratio = self._elapsed_time / (transition_ratio * duration)
            self._current_commands = self.interpolate(
                np.zeros(3), _current_commands, ratio
            )
            return
        elif self._elapsed_time <= (1 - transition_ratio) * duration:
            self._current_commands = _current_commands
            return
        elif self._elapsed_time <= duration:
            ratio = (self._elapsed_time - (1 - transition_ratio) * duration) / (
                transition_ratio * duration
            )
            self._current_commands = self.interpolate(
                _current_commands, np.zeros(3), ratio
            )
            return

    @property
    def current_commands(self) -> np.array:
        return self._current_commands

    def interpolate(self, start_cmd, end_cmd, ratio):
        """
        start_cmd: numpy array
        end_cmd: numpy array
        ratio: float, [0.0, 1.0]
        """
        assert 0.0 <= ratio <= 1.0
        interpolated_cmd = (1 - ratio) * start_cmd + ratio * end_cmd
        return interpolated_cmd
