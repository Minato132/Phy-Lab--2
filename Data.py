import numpy as np
import pandas as pd
import statistics as stats

with open('senriched.txt') as file:
    sig = pd.read_csv(file, skiprows = 2, header = None)

with open('bgdata.txt') as file1:
    bg = pd.read_csv(file1, skiprows = 2, header = None)

sig_totstart = sig.iat[14951, 2]
bg_totstart = bg.iat[5962, 2]

#creating bins of size 100 ns
bins = np.arange(0, max(sig[0]), 10)

bin = pd.cut(sig[0], bins = bins, include_lowest = True, labels = False)
#bin_count = sig['bins'].value_counts(sort = False, retbins = True)

print(bin)
