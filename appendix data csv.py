import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import scipy.stats as stats
import re


import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval})
df.rename(columns={'Per Lost Fault Error': 'Lost Fault', 'Per Large Fault Error': 'Large Fault', 'Total Length Error': 'Total Length'}, inplace=True)
df['Dataset'] = df['Dataset'].apply(lambda x: re.sub(r'.csv', '', x))
df['Lost Fault'] = df['Lost Fault'].apply(lambda x: np.round(x, 4))
df['Large Fault'] = df['Large Fault'].apply(lambda x: np.round(x, 4))
df['Total Length'] = df['Total Length'].apply(lambda x: np.round(x, 4))
output = df[['Dataset', 'Lost Fault', 'Large Fault', 'Total Length']]
output.to_csv('minimum sample rates.csv', index=False)