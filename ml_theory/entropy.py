import torch
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch.nn as nn

from collections import defaultdict

options = 8


# calculating entropy

def entropy(probs) -> float:
    total = sum(probs * np.log2(probs))
    return -total


coin = np.array([0.5, 0.5])
coin_bias = np.array([0.2, 0.8])
dice = np.ones(6) * 1 / 6
lottery = np.ones(1_000_000) * 1 / 1_000_000

ent_coin = entropy(coin).round(2)
ent_coin_bias = entropy(coin_bias).round(2)
ent_dice = entropy(dice).round(2)
ent_lottery = entropy(lottery).round(2)

print(f'entropy of coin: {ent_coin}')
print(f'entropy of coin_bias: {ent_coin_bias}')
# less information gain for knowing outcome of a know bias coin, ie it is less random for the reduction in uncertainity
# is lower
print(f'entropy of dice: {ent_dice}')
print(f'entropy of lottery: {ent_lottery}')  # higher information gain for knowing outcome of lottery
print()


# cross entropy

def cross_entropy(prob_actual: np.ndarray, prob_predicted: np.ndarray) -> float:
    return - sum(prob_actual * np.log2(prob_predicted))


size = 30
poss = np.random.rand(size).reshape(-1, 1)
poss = np.concatenate([(np.ones(size).reshape(-1, 1) - poss), poss], axis=1)
poss = poss[poss[:, 1].argsort()][::-1]

for probs in poss:
    ent = cross_entropy(coin_bias, probs)
    print(f'entropy: {ent.round(2)}'.ljust(14) + f'{probs.round(2)}' + f'{coin_bias}')
