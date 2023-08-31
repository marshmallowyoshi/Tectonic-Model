import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import re


import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval})


# for i in df[['Integral Error', 'Modulus Error']].iterrows():
#     plt.plot(i[1][0], color='red')
#     plt.plot(i[1][1], color='blue')
#     plt.show()


final_integral_error = df['Integral Error'].apply(lambda x: x[-1]*100)

final_modulus_error = df['Modulus Error'].apply(lambda x: x[-1]*100)

final_intersect_error = df['Intersect Error'].apply(lambda x: x[-1])

bad_index = final_integral_error[(final_integral_error > 100) | (final_integral_error < 0)] + final_modulus_error[(final_modulus_error > 100) | (final_modulus_error < 0)]
final_integral_error = final_integral_error.drop(bad_index.index)
final_modulus_error = final_modulus_error.drop(bad_index.index)
final_intersect_error = final_intersect_error.drop(bad_index.index)


print(np.corrcoef(final_integral_error, final_modulus_error)[0,1])
# print(np.median(final_integral_error))
# print(np.median(final_modulus_error))
plt.scatter(final_integral_error, final_modulus_error, color='red', marker='x')
plt.axvline(np.median(final_integral_error), color='blue', label=str('Integral Error Median: ' + str(np.round(np.median(final_integral_error), 3)) + '%'))
plt.axhline(np.median(final_modulus_error), color='blue', label=str('Modulus Error Median: ' + str(np.round(np.median(final_modulus_error), 3)) + '%'))
plt.axvline(np.mean(final_integral_error), color='green', label=str('Integral Error Mean: ' + str(np.round(np.mean(final_integral_error), 3)) + '%'))
plt.axhline(np.mean(final_modulus_error), color='green', label=str('Modulus Error Mean: ' + str(np.round(np.mean(final_modulus_error), 3)) + '%'))
plt.xlabel('Integral Error (%)')
plt.ylabel('Modulus Error (%)')
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 110, 10))
plt.xlim(0,100)
plt.ylim(0,100)
plt.margins(x=0, y=0)
plt.legend()
plt.show()













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