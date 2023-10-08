import torch
import torch.nn as nn


class CubeNet(nn.Module):
    def __init__(self, numLayers):
        super(CubeNet, self).__init__()
        shapeIn = 3 ** 2 * 6 * 6
        self.resBlocks = nn.ModuleList()

        self.firstBlock = nn.Sequential(
            nn.Linear(shapeIn, 5000), #le réseau de neurones linéaire
            nn.BatchNorm1d(5000), #le réseau de neurones de normalisation
            nn.ReLU(), #le réseau de neurones qui donne 0 si la valeur d'entrée est négative et la valeur sinon
            nn.Linear(5000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        )

        for _ in range(numLayers):
            self.resBlocks.append(ResidualBlock(1000))

        self.lastBlock = nn.Linear(1000, 1)
        

    def forward(self, states):

        out = self.firstBlock(states.float())

        for block in self.resBlocks:
            out = block(out)

        out = self.lastBlock(out)

        return out.squeeze([1])


class ResidualBlock(nn.Module):
    def __init__(self, shapeIn):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(shapeIn, shapeIn),
            nn.BatchNorm1d(shapeIn),
            nn.ReLU(),
            nn.Linear(shapeIn, shapeIn),
            nn.BatchNorm1d(shapeIn),
        )

        self.ReLU = nn.ReLU()

    def forward(self, states):
        return self.ReLU(self.layers(states) + states)
