# optimizers/base.py
class Optimizer:
    def update(self, param, grad, name):
        raise NotImplementedError
