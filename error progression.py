import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import scipy.stats as stats


import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval, 'Sample Counts':ast.literal_eval})


integral_errors = df['Integral Error']
modulus_errors = df['Modulus Error']
intersect_errors = df['Intersect Error']
sample_rates = df['Sample Counts']
dataset = df['Dataset']

for idx, val in enumerate(integral_errors):
    x = list(sample_rates[idx])
    y = list(integral_errors[idx][:-1])
    y2 = list(modulus_errors[idx][:-1])
    y3 = list(intersect_errors[idx][:-1])
    fig, ax = plt.subplots()
    ax.plot(x,y, color='blue', label = 'Integral Error')
    ax.plot(x,y2, color='red', label = 'Modulus Error')
    ax.plot(x,y3, color='green', label = 'Intersect Error')
    ax.set_title(dataset[idx])
    ax.set_xlabel('Total Samples')
    ax.set_ylabel('Error (%)')
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    plt.show()