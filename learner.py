import numpy as np
import numpy.typing as npt
import constants
import pickle
import time
import os
from multiprocessing import Pool
import game


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
        assert biases and all(bias_vec.ndim == 1 for bias_vec in biases)

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

    def __call__(self, *args, **kwargs) -> npt.NDArray[float]:
        """Evaluate one or many arguments."""
        assert len(args) > 0
        if len(args) == 1:
            return self.evaluate(args[0])
        else:
            return self.evaluate(np.asarray(args))

    def dump(self):
        """Dump the network's weights and biases into a 1D array."""
        return np.concatenate((*(weight_mat.flatten() for weight_mat in self.weights), *self.biases))

    def load(self, chromosome):
        """Load weights and biases from 1D array."""
        weight_count = sum(weight_mat.size for weight_mat in self.weights)
        bias_count = sum(bias_vec.size for bias_vec in self.biases)
        assert weight_count + bias_count == len(chromosome)
        head = 0
        for i, weight_mat in enumerate(self.weights):
            tail = head + weight_mat.size
            self.weights[i] = np.reshape(chromosome[head:tail], self.weights[i].shape)
            head = tail
        for i, bias_vec in enumerate(self.biases):
            tail = head + bias_vec.size
            self.biases[i] = chromosome[head:tail]
            head = tail
        assert head == len(chromosome)

    @classmethod
    def xavier(cls, dimensions: tuple[int, ...], *args, **kwargs):
        """Create a new Network with Xavier initialization."""
        weights = [(np.random.rand(dimensions[i + 1], dimensions[i]) - 0.5) * 2 / np.sqrt(dimensions[i])
                   for i in range(len(dimensions) - 1)]
        biases = [np.zeros(k) for k in dimensions[1:]]
        return cls(weights, biases, *args, **kwargs)

    def __repr__(self):
        return "< Network: " + '-'.join(map(str, self.shape())) + ' >'

    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the network."""
        return tuple([self.input_dim()] + [self.layer_output_dim(i) for i in range(self.layer_count)])


class Breaker:
    input_layer_size = constants.DIM_X * constants.DIM_Y * 2 - constants.DIM_X + 1  # == 103
    hidden_layer_size = 64

    def __init__(self, network=None):
        if network is None:
            network = Network.xavier((Breaker.input_layer_size, Breaker.hidden_layer_size, 1))
        self.network = network

    def evaluate(self, grid: list[list[int]], points: list[list[int]], shoot_pos_x: float) -> float:
        """Evaluate given game state and return angle at which to shoot balls."""
        grid_preprocessed = np.log2(np.asarray(grid[:-1]).flatten() + 1)  # DIM_X * (DIM_Y - 1) == 48
        points_preprocessed = np.asarray(points).flatten()  # DIM_X * DIM_Y == 54
        shoot_pos_preprocessed = shoot_pos_x / constants.RES_X

        # Create input vector
        input_vector = np.concatenate((grid_preprocessed, points_preprocessed, [shoot_pos_preprocessed]))
        # Receive output vector
        output_value = self.network(input_vector)

        # Clip and map range
        clip_size = 0.2
        zero_one_range_val = (np.clip(output_value, clip_size, 1 - clip_size) - clip_size) / (1 - 2 * clip_size)
        angle = (constants.ANGLE_MAX_RAD - constants.ANGLE_MIN_RAD) * zero_one_range_val + constants.ANGLE_MIN_RAD
        return angle

    def __call__(self, *args, **kwargs) -> float:
        return self.evaluate(*args, **kwargs)

    def __repr__(self):
        return f"[ Breaker with {self.network} ]"

    def dump(self):
        """Dump the network's weights and biases into a 1D array."""
        return self.network.dump()

    def load(self, chromosome):
        """Load weights and biases from 1D array."""
        return self.network.load(chromosome)

    def run(self, draw=False, fps_cap=constants.FPS, block=False):
        return game.main(breaker_override=self, draw=draw, fps_cap=fps_cap, block=block)


def mutate(chromosome: npt.NDArray[float], mutation_rate: float = 0.1):
    """Mutate given chromosome and return copy."""
    mutation_mask = np.random.rand(len(chromosome)) < mutation_rate
    deviation = np.random.normal(loc=0, scale=0.05, size=len(chromosome))
    return chromosome + deviation * mutation_mask


def crossover(chromosome1: npt.NDArray[float], chromosome2: npt.NDArray[float]) \
        -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Perform crossover of two chromosomes and return the two results."""
    k = np.random.randint(0, len(chromosome1) - 1)
    result_1 = np.concatenate((chromosome1[:k], chromosome2[k:]))
    result_2 = np.concatenate((chromosome2[:k], chromosome1[k:]))
    return result_1, result_2


def batch_simulate(population: list[Breaker], fitness: callable, process_count: None | int = None) \
        -> list[tuple[int, int]]:
    """Given population of Breakers and a fitness function, perform parallelized simulation.

    Returns a list of tuples in (chromosome index, fitness score) form.
    """
    if process_count is None:
        process_count = os.cpu_count()
    with Pool(process_count) as p:
        scores = list(enumerate(p.map(fitness, population)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def breaker_run_5(breaker, *args, **kwargs):
    scores = sorted([breaker.run(*args, **kwargs) for _ in range(5)])
    return sum(scores[1:-1]) / 3


class Generation:
    def __init__(self, generation: int, population: list[Breaker], scores: list[tuple[int, int]]):
        self.generation = generation
        self.population = population
        self.scores = scores

    def __repr__(self):
        return f"< Gen {self.generation}: {self.scores[0][1]}>"


POPULATION_SIZE = 1024
TEMPLATE_COUNT = 16


def main():
    population = [Breaker() for _ in range(POPULATION_SIZE)]
    for generation in range(288):
        print(f"Generation {generation} took... ", end='')
        start = time.time()
        scores = batch_simulate(population, breaker_run_5)
        end = time.time()
        print(f"{end - start:.1f} seconds to simulate.")

        # Save to file
        with open(f"./generations/{population[0].network.shape()}-gen-{generation}.pickle", 'wb') as f:
            genobj = Generation(generation, population, scores)
            pickle.dump(genobj, f)

        # Showcase the fittest Breaker
        print(f"Best of Generation {generation}: {scores[0][1]:.1f}")
        population[scores[0][0]].run(draw=True, fps_cap=288, block=False)

        selected_indexes = [pair[0] for pair in scores[:TEMPLATE_COUNT]]
        template_chromosomes = [population[idx].dump() for idx in selected_indexes]
        # 16 will be direct copies. 528 will be mutations of the templates.
        chromosome_pool = template_chromosomes[:]
        for idx in range(TEMPLATE_COUNT):
            for _ in range(33):
                chromosome_pool.append(mutate(template_chromosomes[idx]))
        # 240 will be crossovers between the templates.
        crossovers = []
        for idx1 in range(TEMPLATE_COUNT):
            for idx2 in range(idx1 + 1, TEMPLATE_COUNT):
                crossovers += list(crossover(template_chromosomes[idx1], template_chromosomes[idx2]))
        chromosome_pool += crossovers
        # 240 will be mutations of the crossed-over templates.
        for chromosome in crossovers:
            chromosome_pool.append(mutate(chromosome))
        # This will amount to a total of 1024 chromosomes, for the next generation!
        assert len(chromosome_pool) == POPULATION_SIZE
        for idx, breaker in enumerate(population):
            breaker.load(chromosome_pool[idx])


if __name__ == "__main__":
    main()