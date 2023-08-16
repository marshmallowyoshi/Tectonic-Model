import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt
from scipy import integrate
import scipy

name = 'R2 H3'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

# MODULUS ERROR


allSegments, polynomialRegression = alg.fullSegmentAnalysis(x,y, name=name, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=False)
x_2 = alg.reduceDataDecimate(x, 4)
y_2 = alg.reduceDataDecimate(y, 4)
allSegments2, polynomialRegression2 = alg.fullSegmentAnalysis(x_2,y_2, name=name, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=False)
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
xResampled, yResampled = alg.resampleToPeak(x,y)
x_2Resampled, y_2Resampled = alg.resampleToPeak(x_2,y_2)
alg.cubicPlot(allSegments, polynomialRegression, label='Original', color='red')
alg.cubicPlot(allSegments2, polynomialRegression2, label='Reduced', color='blue')
plt.plot(np.arange(250), y_2Resampled, color='black', label='Raw Data')
# plt.plot(np.arange(2500),y_2Resampled, color='green', label='Reduced Data')
# for i in range(len(allSegments)):
#     xPlot1 = np.linspace(0, allSegments[i][2]-allSegments[i][0], 250)
#     xPlot2 = np.linspace(0, allSegments2[i][2]-allSegments2[i][0], 250)
#     f1 = lambda x: polynomialRegression[i][0].coef_[2]*x**3 + polynomialRegression[i][0].coef_[1]*x**2 + polynomialRegression[i][0].coef_[0]*x + polynomialRegression[i][0].intercept_
#     f2 = lambda x: polynomialRegression2[i][0].coef_[2]*x**3 + polynomialRegression2[i][0].coef_[1]*x**2 + polynomialRegression2[i][0].coef_[0]*x + polynomialRegression2[i][0].intercept_

#     yPlot1 = f1(xPlot1)
#     yPlot2 = f2(xPlot2)
#     plt.fill_between(xPlot1 + allSegments[i][0], yPlot1,yPlot2, color='green', alpha=0.5)
title = "Integral Error: " + str(np.round(totalDiff,2)) + "%"

plt.title(title)
plt.legend()
plt.show()
# f = scipy.interpolate.interp1d(x, y)
# # plt.plot(x_2,y_2)
# # plt.plot(x,y, label='Original')
# # plt.plot(np.linspace(np.min(x),np.max(x),2500), f(np.linspace(np.min(x),np.max(x),2500)), label='Interpolated')
# # plt.plot(np.linspace(np.min(x), np.max(x), len(yResampled)), yResampled, label='Resampled')
# plt.legend() 
# plt.show()
