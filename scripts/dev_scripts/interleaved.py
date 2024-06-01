import numpy as np
import matplotlib.pyplot as plt

n_time = 900
n_segs = 20
n_len = n_time / n_segs
sinds = np.linspace(0, n_time - n_len, n_segs).astype(int)
itest = (sinds[:, np.newaxis] + np.arange(0, n_len * 0.5, 1, int)).flatten()
itrain = np.ones(n_time, "bool")
itrain[itest] = 0
itest = ~itrain

plt.plot(itrain)
plt.title("train times")
plt.show()
