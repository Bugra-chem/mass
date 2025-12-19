from .random_mt19937 import MT19937

class RandomNumber:
    def __init__(self, seed):
        self.rng = MT19937(seed)

    def get_random_number(self):
        return self.rng.random_double()
