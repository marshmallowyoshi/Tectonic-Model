import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt

name = 'R2 H3'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

# MODULUS ERROR


allSegments, polynomialRegression = alg.full_segment_analysis(x,y, name=name, peak_rate_threshold=0.03, join=True, segment_mode='poly', plot=False)
x_2, y_2 = alg.resample_to_peak(x, y, sample_rate_override=len(x)/4)
allSegments2, polynomialRegression2 = alg.full_segment_analysis(x_2,y_2, name=name, peak_rate_threshold=0.03, join=True, segment_mode='poly', plot=False)
area = alg.integral_difference(polynomialRegression[0][0], polynomialRegression2[0][0], (allSegments[0][0],allSegments[0][2]), (allSegments2[0][0],allSegments2[0][2]))
allAreas = alg.all_segments_integral(allSegments, allSegments2, polynomialRegression, polynomialRegression2)

totalDiff = allAreas[1]*100
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
xResampled, yResampled = alg.resample_to_peak(x,y)
x_2Resampled, y_2Resampled = alg.resample_to_peak(x_2,y_2)
alg.cubic_plot(allSegments, polynomialRegression, label='Original', color='red')
alg.cubic_plot(allSegments2, polynomialRegression2, label='Reduced', color='blue')

modulus_error = alg.modulus_error(x, y, x_2, y_2)[1]*100
print(modulus_error)
# plt.plot(np.arange(250), y_2Resampled, color='black', label='Raw Data')
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
