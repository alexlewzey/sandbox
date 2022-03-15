import torch
import numpy as np
import pandas as pd
import math
from dstk import dviztk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import torch.nn as nn

from collections import defaultdict

# logs #################################################################################################################

# understanding intercepts
np.log(np.e), np.log(1)
np.log2(2), np.log(1)


# entropy ##############################################################################################################
# Skewed Probability Distribution (unsurprising): Low entropy.
# Balanced Probability Distribution (surprising): High entropy.

def entropy(probs: np.ndarray) -> float:
    return -sum(probs * np.log2(probs))


coin = np.array([0.5, 0.5])
coin_bias = np.array([0.2, 0.8])
dice = np.ones(6) / 6
dice_8 = np.ones(8) / 8
lottery = np.ones(1_000_000) / 1_000_000

e_coin = round(entropy(coin), 2)
e_coin_bias = round(entropy(coin_bias), 2)
e_dice = round(entropy(dice), 2)
e_dice_8 = round(entropy(dice_8), 2)
e_lottery = round(entropy(lottery), 2)

print(f'entropy of coin: {e_coin} - {-(0.5 * np.log2(0.5)) * 2}')
print(f'entropy of coin_bias: {e_coin_bias}')
print(f'entropy of dice: {e_dice}')
print(f'entropy of dice_8: {e_dice_8}')
print(f'entropy of lottery: {e_lottery}')


# 1 bit of information if reducing your uncertaintiy by half
# less information gain for knowing outcome of a know bias coin,
# ie it is less random for the reduction in uncertainity is lower
# higher information gain for knowing outcome of lottery

# cross entropy ########################################################################################################

def cross_entropy(prob_actual: np.ndarray, prob_predicted: np.ndarray) -> float:
    return - sum(prob_actual * np.log2(prob_predicted))


size = 30
poss = np.random.rand(size).reshape(-1, 1)
poss = np.concatenate([(np.ones(size).reshape(-1, 1) - poss), poss], axis=1)
poss = poss[poss[:, 1].argsort()][::-1]

for probs in poss:
    ent = cross_entropy(coin_bias, probs)
    print(f'entropy: {ent.round(2)}'.ljust(14) + f'{probs.round(2)}' + f'{coin_bias}')

# negative log curve
x = np.linspace(0, 1, 100)
y = -np.log2(x)
plt.plot(x, y)
plt.title('negative log loss')
dviztk.plt_fig_save_open()

# binary example of cross entropy loss
actual = np.array([0, 1])
pred_1 = np.array([0.11, 0.94])
pred_2 = np.array([0.95, 0.11])
cross_entropy(actual, pred_1), cross_entropy(actual, pred_2)

# multiclass example of cross entropy loss
actual = np.array([0, 1, 0])
pred_1 = np.array([0.11, 0.94, 0.05])
pred_2 = np.array([0.95, 0.11, 0.45])
cross_entropy(actual, pred_1), cross_entropy(actual, pred_2), -np.log2(pred_1[1]), - np.log2(pred_2[1])

