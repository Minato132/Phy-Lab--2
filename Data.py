import numpy as np
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt

with open('senriched.txt') as file:
    sig = pd.read_csv(file, skiprows = 2, header = None)

with open('bgdata.txt') as file1:
    bg = pd.read_csv(file1, skiprows = 2, header = None)

sig_totstart = sig.iat[14951, 2]
bg_totstart = bg.iat[5962, 2]

#creating bins of size 50 ns
bins = np.arange(0, max(sig[0]), 10)

bg_bin = pd.cut(bg[0], bins = bins, include_lowest = True)
bg_binsize = bg_bin.value_counts(sort = False)
bg_binsize = bg_binsize / bg_totstart

sig_bin = pd.cut(sig[0], bins = bins, include_lowest= True)
sig_binsize = sig_bin.value_counts(sort = False)
sig_binsize = sig_binsize / sig_totstart

final_bins = sig_binsize - bg_binsize

final_bins = final_bins[final_bins >= 0]

x = np.arange(len(final_bins))

plt.scatter(x, np.log(final_bins))
plt.show()

