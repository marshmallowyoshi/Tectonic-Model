import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt
from scipy import integrate

name = 'R2 H3.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

# MODULUS ERROR


allSegments, polynomialRegression = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=False)

x_2 = alg.reduceDataDecimate(x, 2)
y_2 = alg.reduceDataDecimate(y, 2)

allSegments2, polynomialRegression2 = alg.newFullSegmentAnalysis(x_2,y_2, name=name, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=False)


area = alg.integralDifference(polynomialRegression[0][0], polynomialRegression2[0][0], (allSegments[0][0],allSegments[0][2]))
print(area)
