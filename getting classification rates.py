import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval})



lost_fault = df['Per Lost Fault Error']
large_fault = df['Per Large Fault Error']
total_length = df['Total Length Error']


# for i in range(0,100, 1):
#     print(str(str(i)+"% :"),"Total Length:" , np.round(len(total_length[total_length < i])/len(total_length), 3), end=" | ")
#     print("Large Fault", np.round(len(large_fault[large_fault < i])/len(large_fault), 3), end=" | ")
#     print("Lost Fault", np.round(len(lost_fault[lost_fault < i])/len(lost_fault), 3))

# plt.style.use('dark_background')
fig, ax = plt.subplots(1, 3, figsize=(15,5))
fig.tight_layout(pad=3.0)
colors = ('r','g','b')
names = ('Lost Fault', 'Large Fault', 'Total Length')
for index, i in enumerate((lost_fault, large_fault, total_length)):
    lost_fault_list = np.sort(i.values.tolist())

    critical_point = int(np.trunc(len(lost_fault_list)/20))

    lost_fault_list_strings = [str(np.round(i, 4)) for i in lost_fault_list]
    lost_fault_list_strings[critical_point] = str(">>> " + str(lost_fault_list_strings[critical_point]) + " <<<")
    print(lost_fault_list_strings)
    for i in lost_fault_list:
        ax[index].axvline(x=i, color='w', alpha=0.06)
    ax[index].axvline(x=lost_fault_list[critical_point], color=colors[index], alpha=0.8, label=str("5% of " + names[index]))
    ax[index].set_title(names[index] + " Method")
    ax[1].set_xlabel("Critical Point (%)")
    ax[0].set_ylabel("Brightness = Frequency Density")
    ax[index].set_xlim(0, 50)
    ax[index].set_ylim(0, 50)
    ax[index].set_xticks(np.arange(0, 55, 5))
    ax[index].set_xticks(np.arange(0, 50, 1), minor=True)
    ax[index].set_yticks([])
    ax[index].set_facecolor('black')
    # mu, std = stats.norm.fit(lost_fault_list)

    # lost_fault_list = lost_fault_list[lost_fault_list < 3*std]

    # ax[index].hist(lost_fault_list, bins=25, density=True, alpha=0.6, color='g')
fig.legend(bbox_to_anchor=(0.99,0.91), loc='upper right', ncol=1, borderaxespad=0., fontsize=10)
plt.show()