import numpy as np
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt
import scipy.optimize as opt

with open('senriched.txt') as file:
    sig = pd.read_csv(file, skiprows = 2, header = None)

with open('background2.csv') as file:
    bg = pd.read_csv(file, skiprows = 3, header = None)

bg = bg.loc[:,[0, 1, 2]]

sig_totstart = sig.iat[14951, 2]
bg_totstart = bg.iat[10220, 2]

#creating bins of size 90 ns
bins = np.arange(0, max(sig[0]), 9)

#binning background
bg_bin = pd.cut(bg[0], bins = bins, include_lowest = True)
bg_binsize = bg_bin.value_counts(sort = False)
bg_binsize = bg_binsize / bg_totstart
scale = np.arange(len(bg_binsize)) * 100 / 1000

# plt.figure()
# plt.scatter(scale, bg_binsize)
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Count Rate')
# plt.title('Background noise distribution')


#binning signal
sig_bin = pd.cut(sig[0], bins = bins, include_lowest= True)
sig_binsize = sig_bin.value_counts(sort = False)
sig_binsize = sig_binsize / sig_totstart
sscale = np.arange(len(sig_binsize)) * 100 / 1000
# plt.figure()
# plt.scatter(sscale, sig_binsize)
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Count Rate')
# plt.title('Signal enriched distribution')
# plt.show()

#subtracting the bins
final_bin = sig_binsize - bg_binsize 

#creating the time scale that everything is on
time = np.arange(len(final_bin)) * 100 / 1000


#combining the time and the final bins
index = np.arange(len(time))
data = pd.DataFrame(final_bin)
data.rename(columns = {0 : "counts"}, inplace = True)
data['time'] = time
data['index'] = index
data['error'] = np.abs(data['counts'] * .1)
data['err_prop'] = np.abs((1/np.log(10)) * (data['error'] / data['counts']))
data.set_index('index', inplace = True)


#showing the linear parts through log of data
lin_data = data.loc[1:, :]
lin_data = lin_data.loc[lin_data['time'] < 10]
lin_data['counts'] = -np.log10(lin_data['counts'])
lin_data.dropna(inplace = True)

#fit function for linear
def lin(x, a, b):
    y = ((a*x) + b)
    return y

a, b = opt.curve_fit(lin, lin_data['time'], lin_data['counts'], method = 'trf')
plt.figure()
plt.scatter(data['time'], -np.log10(data['counts']), marker = 'x', color = 'green')
plt.errorbar(data['time'], -np.log10(data['counts']), yerr = data['err_prop'], fmt = ',', color = 'purple')
plt.plot(lin_data['time'], lin(lin_data['time'], *a), color = 'Red') 
plt.xlabel('Time (microseconds)')
plt.ylabel('Log of Normalized True Counts')
plt.title('Linearization of Signal Counts')




#Finding the life-time of a muon
#fit function for exp plot 
def func(x, a, b):
    y = (a * np.exp(-x / b))
    return y 

exp_data = data.loc[1:, :]
exp_data = exp_data.loc[exp_data['time'] < 40]
c, d = opt.curve_fit(func, exp_data['time'], exp_data['counts'])


#Doing the chi-square by hand
exp_data['expect'] = func(exp_data['time'], *c)

counts = exp_data['counts'].to_numpy() 
fitteddata = exp_data['expect'].to_numpy()

chi_square = 0
for i in range(len(counts)):
    chi_square += ((counts[i] - fitteddata[i]) ** 2) / fitteddata[i]
print(chi_square)



# print(*c)
# print(np.sqrt(d[1,1]))
# plt.figure()
# plt.plot(exp_data['time'], func(exp_data['time'], *c), color = 'red')
# plt.errorbar(exp_data['time'], exp_data['counts'], yerr = exp_data['error'], fmt = ',', color = 'purple')
# plt.scatter(exp_data['time'], exp_data['counts'], marker = 'x', color = 'green')
# plt.xlabel('Time (microseconds)')
# plt.ylabel('Normalized Count Rate')
# plt.title('Fitted curve of the True signal')
# plt.show()
