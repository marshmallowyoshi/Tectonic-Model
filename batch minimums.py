import pandas as pd
import os
import algorithm as alg
import matplotlib.pyplot as plt

def batchMinimum(rawData):
    errorData = []
    for key, value in rawData.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                error = alg.findMinimumSampleRate(value2[0], value2[1], name=key + ' ' + key2, plot=False, verbose=False, resampling_kind='quadratic', peakRateThreshold=0.023)
                errorData.append((key + ' ' + key2, error))

        else:
            error = alg.findMinimumSampleRate(value[0], value[1], name=key, plot=False, verbose=False, resampling_kind='quadratic', peakRateThreshold=0.023)
            errorData.append((key, error))
    return errorData

rawData = pd.read_csv('final_dataframe.csv')
print(rawData.head())
# batchResults = batchMinimum(rawData)
# df = pd.DataFrame(batchResults, columns=['Dataset', 'Error'])

# df[['Per Fault Error', 'Total Length Error']] = pd.DataFrame(df['Error'].tolist(), index=df.index)

# df.to_csv('results peakRateThreshold=0.023.csv')
