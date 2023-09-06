import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import scipy.stats as stats


import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval})


# for i in df[['Integral Error', 'Modulus Error']].iterrows():
#     plt.plot(i[1][0], color='red')
#     plt.plot(i[1][1], color='blue')
#     plt.show()


final_integral_error = df['Integral Error'].apply(lambda x: x[-2]*100)

final_modulus_error = df['Modulus Error'].apply(lambda x: x[-2]*100)

final_intersect_error = df['Intersect Error'].apply(lambda x: x[-2]*100)

bad_index = final_integral_error[(final_integral_error > 100) | (final_integral_error < 0)] + final_modulus_error[(final_modulus_error > 100) | (final_modulus_error < 0)]
final_integral_error = final_integral_error.drop(bad_index.index)
final_modulus_error = final_modulus_error.drop(bad_index.index)
final_intersect_error = final_intersect_error.drop(bad_index.index)
lost_fault = df['Per Lost Fault Error'].drop(bad_index.index)
large_fault = df['Per Large Fault Error'].drop(bad_index.index)
total_fault = df['Total Length Error'].drop(bad_index.index)


# plt.scatter(final_integral_error, lost_fault, color='black', marker='x', s=2)
# plt.xlabel('Integral Error (%)')
# plt.ylabel('Per Lost Fault Error (%)')
# plt.title(str(stats.pearsonr(final_integral_error, lost_fault).correlation))
# plt.show()
# plt.scatter(final_integral_error, large_fault, color='black', marker='x', s=2)
# plt.xlabel('Integral Error (%)')
# plt.ylabel('Per Large Fault Error (%)')
# plt.title(str(stats.pearsonr(final_integral_error, large_fault).correlation))

# plt.show()
# plt.scatter(final_integral_error, total_fault, color='black', marker='x', s=2)
# plt.xlabel('Integral Error (%)')
# plt.ylabel('Total Length Error (%)')
# plt.title(str(stats.pearsonr(final_integral_error, total_fault).correlation))
# plt.show()

plt.rcParams['figure.dpi'] = 150
fig, ax = plt.subplots(3,3)

integral_color = 'red'
modulus_color = 'lightblue'
intersect_color = 'lightgreen'

lost_color = 'magenta'
large_color = 'cyan'
total_color = 'yellow'

ax[0,0].scatter(final_integral_error, lost_fault, color='black', marker='.', s=2)
integral_lost_slope, integral_lost_intercept  = (stats.linregress(final_integral_error, lost_fault).slope, stats.linregress(final_integral_error, lost_fault).intercept)
ax[0,0].axline((0,integral_lost_intercept), slope=integral_lost_slope)
ax[0,0].plot()
ax[0,0].set_xlabel('Integral Error (%)', backgroundcolor=integral_color)
ax[0,0].set_ylabel('Per Lost Fault Error (%)', backgroundcolor=lost_color)
ax[0,0].set_title("Correlation = " + str(np.round(stats.pearsonr(final_integral_error, lost_fault).correlation, 3)))
ax[0,1].scatter(final_integral_error, large_fault, color='black', marker='.', s=2)
integral_large_slope, integral_large_intercept  = (stats.linregress(final_integral_error, large_fault).slope, stats.linregress(final_integral_error, large_fault).intercept)
ax[0,1].axline((0,integral_large_intercept), slope=integral_large_slope)
ax[0,1].set_xlabel('Integral Error (%)', backgroundcolor=integral_color)
ax[0,1].set_ylabel('Per Large Fault Error (%)', backgroundcolor=large_color)
ax[0,1].set_title("Correlation = " + str(np.round(stats.pearsonr(final_integral_error, large_fault).correlation, 3)))
ax[0,2].scatter(final_integral_error, total_fault, color='black', marker='.', s=2)
integral_total_slope, integral_total_intercept  = (stats.linregress(final_integral_error, total_fault).slope, stats.linregress(final_integral_error, total_fault).intercept)
ax[0,2].axline((0,integral_total_intercept), slope=integral_total_slope)
ax[0,2].set_xlabel('Integral Error (%)', backgroundcolor=integral_color)
ax[0,2].set_ylabel('Total Length Error (%)', backgroundcolor=total_color)
ax[0,2].set_title("Correlation = " + str(np.round(stats.pearsonr(final_integral_error, total_fault).correlation, 3)))
ax[1,0].scatter(final_modulus_error, lost_fault, color='black', marker='.', s=2)
modulus_lost_slope, modulus_lost_intercept  = (stats.linregress(final_modulus_error, lost_fault).slope, stats.linregress(final_modulus_error, lost_fault).intercept)
ax[1,0].axline((0,modulus_lost_intercept), slope=modulus_lost_slope)
ax[1,0].set_xlabel('Modulus Error (%)', backgroundcolor=modulus_color)
ax[1,0].set_ylabel('Per Lost Fault Error (%)', backgroundcolor=lost_color)
ax[1,0].set_title("Correlation = " + str(np.round(stats.pearsonr(final_modulus_error, lost_fault).correlation, 3)))
ax[1,1].scatter(final_modulus_error, large_fault, color='black', marker='.', s=2)
modulus_large_slope, modulus_large_intercept  = (stats.linregress(final_modulus_error, large_fault).slope, stats.linregress(final_modulus_error, large_fault).intercept)
ax[1,1].axline((0,modulus_large_intercept), slope=modulus_large_slope)
ax[1,1].set_xlabel('Modulus Error (%)', backgroundcolor=modulus_color)
ax[1,1].set_ylabel('Per Large Fault Error (%)', backgroundcolor=large_color)
ax[1,1].set_title("Correlation = " + str(np.round(stats.pearsonr(final_modulus_error, large_fault).correlation, 3)))
ax[1,2].scatter(final_modulus_error, total_fault, color='black', marker='.', s=2)
modulus_total_slope, modulus_total_intercept  = (stats.linregress(final_modulus_error, total_fault).slope, stats.linregress(final_modulus_error, total_fault).intercept)
ax[1,2].axline((0,modulus_total_intercept), slope=modulus_total_slope)
ax[1,2].set_xlabel('Modulus Error (%)', backgroundcolor=modulus_color)
ax[1,2].set_ylabel('Total Length Error (%)', backgroundcolor=total_color)
ax[1,2].set_title("Correlation = " + str(np.round(stats.pearsonr(final_modulus_error, total_fault).correlation, 3)))
ax[2,0].scatter(final_intersect_error, lost_fault, color='black', marker='.', s=2)
intersect_lost_slope, intersect_lost_intercept  = (stats.linregress(final_intersect_error, lost_fault).slope, stats.linregress(final_intersect_error, lost_fault).intercept)
ax[2,0].axline((0,intersect_lost_intercept), slope=intersect_lost_slope)
ax[2,0].set_xlabel('Intersect Error (%)', backgroundcolor=intersect_color)
ax[2,0].set_ylabel('Per Lost Fault Error (%)', backgroundcolor=lost_color)
ax[2,0].set_title("Correlation = " + str(np.round(stats.pearsonr(final_intersect_error, lost_fault).correlation, 3)))
ax[2,1].scatter(final_intersect_error, large_fault, color='black', marker='.', s=2)
intersect_large_slope, intersect_large_intercept  = (stats.linregress(final_intersect_error, large_fault).slope, stats.linregress(final_intersect_error, large_fault).intercept)
ax[2,1].axline((0,intersect_large_intercept), slope=intersect_large_slope)
ax[2,1].set_xlabel('Intersect Error (%)', backgroundcolor=intersect_color)
ax[2,1].set_ylabel('Per Large Fault Error (%)', backgroundcolor=large_color)
ax[2,1].set_title("Correlation = " + str(np.round(stats.pearsonr(final_intersect_error, large_fault).correlation, 3)))
ax[2,2].scatter(final_intersect_error, total_fault, color='black', marker='.', s=2)
intersect_total_slope, intersect_total_intercept  = (stats.linregress(final_intersect_error, total_fault).slope, stats.linregress(final_intersect_error, total_fault).intercept)
ax[2,2].axline((0,intersect_total_intercept), slope=intersect_total_slope)
ax[2,2].set_xlabel('Intersect Error (%)', backgroundcolor=intersect_color)
ax[2,2].set_ylabel('Total Length Error (%)', backgroundcolor=total_color)
ax[2,2].set_title("Correlation = " + str(np.round(stats.pearsonr(final_intersect_error, total_fault).correlation, 3)))

for subplot in ax.flatten():
    subplot.set_xlim(0,100)
    subplot.set_ylim(0,50)
    subplot.set_xticks(np.arange(0, 120, 20))
    subplot.set_yticks(np.arange(0, 60, 10))
    subplot.set_aspect('equal')
    subplot.margins(x=0, y=0)
    subplot.grid()

fig.tight_layout()
plt.show()


# print(np.sqrt(np.var(final_integral_error)))
# print(np.sqrt(np.var(final_modulus_error)))
# print(np.sqrt(np.var(final_intersect_error)))




# print(stats.pearsonr(final_integral_error, final_intersect_error).correlation)

# plt.scatter(final_integral_error, final_intersect_error, color='black', marker='x', s=2)
# plt.axline((0,0), slope=1, color='red')
# # plt.axvline(np.median(final_integral_error), color='blue', label=str('Integral Error Median: ' + str(np.round(np.median(final_integral_error), 3)) + '%'))
# # plt.axhline(np.median(final_intersect_error), color='blue', label=str('Intersect Error Median: ' + str(np.round(np.median(final_intersect_error), 3)) + '%'))
# # plt.axvline(np.mean(final_integral_error), color='green', label=str('Integral Error Mean: ' + str(np.round(np.mean(final_integral_error), 3)) + '%'))
# # plt.axhline(np.mean(final_intersect_error), color='green', label=str('Intersect Error Mean: ' + str(np.round(np.mean(final_intersect_error), 3)) + '%'))
# plt.xlabel('Integral Error (%)')
# plt.ylabel('Interesct Error (%)')
# plt.xticks(np.arange(0, 110, 10))
# plt.yticks(np.arange(0, 110, 10))
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.margins(x=0, y=0)
# plt.show()

print(np.corrcoef(final_integral_error, final_modulus_error)[0,1])
# plt.scatter(final_integral_error, final_modulus_error, color='black', marker='x', s=2)
# # plt.axvline(np.median(final_integral_error), color='blue', label=str('Integral Error Median: ' + str(np.round(np.median(final_integral_error), 3)) + '%'))
# # plt.axhline(np.median(final_modulus_error), color='blue', label=str('Modulus Error Median: ' + str(np.round(np.median(final_modulus_error), 3)) + '%'))
# # plt.axvline(np.mean(final_integral_error), color='green', label=str('Integral Error Mean: ' + str(np.round(np.mean(final_integral_error), 3)) + '%'))
# # plt.axhline(np.mean(final_modulus_error), color='green', label=str('Modulus Error Mean: ' + str(np.round(np.mean(final_modulus_error), 3)) + '%'))
# plt.xlabel('Integral Error (%)')
# plt.ylabel('Modulus Error (%)')
# plt.xticks(np.arange(0, 110, 10))
# plt.yticks(np.arange(0, 110, 10))
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.margins(x=0, y=0)
# # plt.legend()
# plt.show()

############################################################################################################################################

# max_length = df['Integral Error'].apply(lambda x: len(x)).max()
# lengths = df['Integral Error'].apply(lambda x: len(x)).value_counts().sort_index()




# df['Integral Error'] = df['Integral Error'].apply(lambda x: np.asarray(x)).apply(lambda x: np.concatenate([x, np.zeros(max_length - len(x))]))

# integral_error_total = sum(df['Integral Error'].to_list())


# lengths_array = np.zeros(max_length)
# for length in lengths.keys():
#     for i in range(length):
#         lengths_array[i] += lengths[length]
# print(lengths_array)
# print(integral_error_total)
# for idx, count in enumerate(lengths_array):
#     idx = int(idx)
#     integral_error_total[idx] = integral_error_total[idx] / count


# print(integral_error_total)

