class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        raise NotImplementedError

    def get_params(self):
        return {}

    def set_params(self, params):
        pass
