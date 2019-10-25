import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

theoretical_val = pd.read_csv('theo.dms')
simulated_val = pd.read_csv('simu.dms')
n = np.arange(1, 30, 1).tolist()

plt.plot(n,theoretical_val,label="theoretical")
plt.plot(n,simulated_val,label="simulated")
plt.legend()
plt.xlabel("number of coin tosses")
plt.ylabel("probability of Alice winning")
plt.show()
