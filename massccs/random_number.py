import random

class RandomNumber:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def get_random_number(self):
        return self.rng.random()
