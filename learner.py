import numpy as np
import numpy.typing as npt


def sigmoid(x: float | npt.NDArray[float]) -> float | npt.NDArray[float]:
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, weights: list[npt.NDArray[float]], biases: list[npt.NDArray[float]],
                 activation: callable = sigmoid):
        self.weights = weights
        self.biases = biases
        self.activation = activation

        # Check the dimensions of the weights and biases
        assert weights and all(weight_mat.ndim == 2 for weight_mat in weights)
        assert biases and all(bias_mat.ndim == 1 for bias_mat in biases)

        # Check that the number of layers is consistent.
        layer_count = len(weights)
        assert layer_count >= 1
        assert len(biases) == layer_count
        self.layer_count = layer_count

        # Check that the dimensions are consistent between layers.
        for i in range(layer_count):
            assert self.layer_output_dim(i) == len(biases[i])
            if i + 1 < layer_count:
                assert self.layer_output_dim(i) == self.layer_input_dim(i + 1)

    def layer_input_dim(self, layer_idx: int) -> int:
        """Return the dimension of input of specified layer."""
        return len(self.weights[layer_idx][0])

    def layer_output_dim(self, layer_idx: int) -> int:
        """Return the dimension of output of specified layer."""
        return len(self.weights[layer_idx])

    def input_dim(self) -> int:
        """Return the input dimension of the network."""
        return self.layer_input_dim(0)

    def output_dim(self) -> int:
        """Return the output dimension of the network."""
        return self.layer_output_dim(-1)

    def evaluate(self, x: npt.NDArray[float]) -> npt.NDArray[float]:
        """Evaluate given x."""
        assert x.ndim == 1 or x.ndim == 2
        if x.ndim == 1:
            for layer_idx in range(self.layer_count):
                x = self.activation(self.weights[layer_idx] @ x + self.biases[layer_idx])
            if len(x) == 1:
                x = x[0]
            return x
        else:
            return np.array([self.evaluate(x_elem) for x_elem in x])

    def __call__(self, *args, **kwargs):
        """Evaluate one or many arguments."""
        assert len(args) > 0
        if len(args) == 1:
            return self.evaluate(args[0])
        else:
            return self.evaluate(np.asarray(args))

    @classmethod
    def xavier(cls, dimensions: list[int], *args, **kwargs):
        """Create a new Network with Xavier initialization."""
        weights = [(np.random.rand(dimensions[i + 1], dimensions[i]) - 0.5) * 2 / np.sqrt(dimensions[i])
                   for i in range(len(dimensions) - 1)]
        biases = [np.zeros(k) for k in dimensions[1:]]
        return cls(weights, biases, *args, **kwargs)

    def __repr__(self):
        return "< Network: " + \
               '-'.join(map(str, [self.input_dim()] + [self.layer_output_dim(i) for i in range(self.layer_count)])) + \
               ' >'


mynet = Network.xavier([96, 48, 1])
