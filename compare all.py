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
x_2 = alg.reduceDataDecimate(x, 8)
y_2 = alg.reduceDataDecimate(y, 8)
allSegments2, polynomialRegression2 = alg.newFullSegmentAnalysis(x_2,y_2, name=name, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=False)
area = alg.integralDifference(polynomialRegression[0][0], polynomialRegression2[0][0], (allSegments[0][0],allSegments[0][2]))
allAreas = alg.allSegmentsIntegral(allSegments, allSegments2, polynomialRegression, polynomialRegression2)

totalDiff = 100 * np.asarray(allAreas)[:,1].sum()/np.asarray(allAreas)[:,2].sum()
print(np.round(totalDiff,2), "%", sep="")

# numPoints = 100
# for i in range(len(allSegments)):
#     xPlot = np.linspace(0, allSegments[i][2]-allSegments[i][0], numPoints)
#     f = lambda x: polynomialRegression[i][0].coef_[2]*x**3 + polynomialRegression[i][0].coef_[1]*x**2 + polynomialRegression[i][0].coef_[0]*x + polynomialRegression[i][0].intercept_
#     yPlot = f(xPlot)
#     xPlot = xPlot + allSegments[i][0]
#     plt.plot(xPlot, yPlot, color='red')

#     xPlot = np.linspace(0, allSegments2[i][2]-allSegments2[i][0], numPoints)
#     f = lambda x: polynomialRegression2[i][0].coef_[2]*x**3 + polynomialRegression2[i][0].coef_[1]*x**2 + polynomialRegression2[i][0].coef_[0]*x + polynomialRegression2[i][0].intercept_
#     yPlot = f(xPlot)
#     xPlot = xPlot + allSegments2[i][0]
#     plt.plot(xPlot, yPlot, color='blue')
# plt.show()

alg.cubicPlot(allSegments, polynomialRegression)
alg.cubicPlot(allSegments2, polynomialRegression2, color='blue')
title = "Modulus Error: " + str(np.round(totalDiff,2)) + "%"

plt.title(title)
plt.show()