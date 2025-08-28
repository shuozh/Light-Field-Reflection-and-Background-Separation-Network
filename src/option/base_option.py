import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
    def initialize(self):
        self.parser.add_argument('--seed', type=int, default=2012)
        self.initialized = True


