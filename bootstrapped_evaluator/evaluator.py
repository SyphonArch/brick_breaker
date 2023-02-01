import torch
import torch.nn as nn
import torch.nn.functional as torchf
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.optim as optim
import game
import numpy.typing as npt
import numpy as np
from bootstrapped_evaluator.paths import *
import matplotlib.pyplot as plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract(gamevar: game.Game) -> npt.NDArray[float]:
    """Given the Game after a simulation, return the vector to input to an evaluator network.

    Reads the grid_before_gen attribute as opposed to the grid attribute, as the random generation is not of interest.
    """
    return convert(gamevar.grid_before_gen, gamevar.ball_count)


def convert(grid: npt.NDArray[float], ball_count: int) -> npt.NDArray[float]:
    """Given the grid and the ball count, return the vector to input to an evaluator network."""
    grid_segment = grid[1:-1]
    together = np.append(grid_segment.flatten(), ball_count)
    return together.astype(np.float32)


class Evaluator(nn.Module):
    """Given the game state (pre-generation of new bricks), evaluate how close it is to game-over.

    Input: brick layout (excluding top and bottom rows) [7x6] + ball count [1] == [43]
    Output: log((steps to game-over) + 1)

    This means that the output is always non-negative.
    A larger output signifies a 'better' game state.
    """

    def __init__(self):
        super().__init__()
        dim_x = 6
        dim_y = 9
        h1 = 24
        h2 = 12
        self.lin1 = nn.Linear(dim_x * (dim_y - 2) + 1, h1)
        self.drop1 = nn.Dropout(p=0.5)
        self.lin2 = nn.Linear(h1, h2)
        self.drop2 = nn.Dropout(p=0.5)
        self.lin3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = torchf.relu(self.drop1(self.lin1(x)))
        x = torchf.relu(self.drop2(self.lin2(x)))
        x = torchf.relu(self.lin3(x))
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class BreakerDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
            self.input_vectors = dataset[0]
            self.labels = dataset[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_vectors[idx], self.labels[idx]


def train(generation: int, num_epoch):
    learning_rate = 0.01
    train_dataset = BreakerDataset(f"{dataset_path(generation - 1)}/train.pickle")
    test_dataset = BreakerDataset(f"{dataset_path(generation - 1)}/test.pickle")

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    net = Evaluator()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print_period = 20
            if i % print_period == print_period - 1:
                print(f"[{epoch + 1}, {i + 1:4d}] loss: {running_loss / print_period:.3f}")
                running_loss = 0.0

    print("Finished Training")
    net.save(evaluator_path(generation))

    inputs1, labels1 = test_dataset[:]
    inputs2, labels2 = train_dataset[:]
    net.eval()
    with torch.no_grad():
        outputs1 = net(torch.FloatTensor(np.asarray(inputs1)).to(device)).cpu()
        outputs2 = net(torch.FloatTensor(np.asarray(inputs2)).to(device)).cpu()
    plot.axes().axline((0, 0), slope=1, color="red")
    plot.scatter(labels2, outputs2, marker=1, label='train')
    plot.scatter(labels1, outputs1, marker=0, label="test")
    plot.title(f"EPOCH: {num_epoch}, lr: {learning_rate}")
    plot.title(f"EPOCH: {num_epoch}, lr: {learning_rate}")
    plot.xlabel("ground truth")
    plot.ylabel("prediction")
    plot.legend()
    plot.show()


if __name__ == '__main__':
    # torch.device("cpu")
    train(1, 100)
