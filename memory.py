from collections import namedtuple


Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state', 'next_action', 'is_terminal'))


class Memory:
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Experience(*args))

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = []