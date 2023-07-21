import algorithm as alg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math
import scipy.special as sp
import scipy.optimize as opt
from scipy import stats

name = 'R2 H3.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

peakRateThreshold = 0.03
contWav = alg.waveletGen(x,y)

############################################################################################################################################

# allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=False)
# contWav = alg.waveletGen(x,y)
# plt.plot(np.arange(250),signal.resample(y, 250) * 0.8 * contWav[0].shape[0] / np.max(y), color='black', linewidth=0.5)
# alg.regressionPlot(allSegments, allCurves, y, contWav)
# plt.show()

############################################################################################################################################

allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
plt.show()

############################################################################################################################################

# sampleFactor = 1
# segmentCount = 0
# allSegmentsFull = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat')
# xNew = x
# yNew = y
# while True:
#     sampleFactor = sampleFactor+1
#     # xNew = signal.decimate(x, int(sampleFactor))
#     # yNew = signal.decimate(y, int(sampleFactor))
    
#     print('Number of Samples: ', len(xNew))
#     allSegments = alg.newFullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat', plot=False)
#     if len(allSegments) < len(allSegmentsFull) or len(xNew) < 4:
#         xPrevious = alg.reduceDataDecimate(x, sampleFactor-2)
#         yPrevious = alg.reduceDataDecimate(y, sampleFactor-2)
#         print('Starting Samples: ', len(x),'Minimum Samples: ', len(xPrevious))
#         percentage = 100 * (len(x)/len(xPrevious)) / np.min([i[2]-i[0] for i in allSegmentsFull])
#         oldSegments = alg.newFullSegmentAnalysis(xPrevious,yPrevious, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)[0]
#         newPercentage = np.min([np.abs(np.asarray(signal.resample(x, 250))-signal.resample(xPrevious, 250)[i[2]]).argmin()-np.abs(np.asarray(signal.resample(x, 250))-signal.resample(xPrevious, 250)[i[0]]).argmin() for i in oldSegments])
#         print(np.asarray(signal.resample(x, 250)))
#         # print('Sample as Percentage of Fault Length: ', percentage, '%')

#         plt.scatter(np.linspace(0, 249, len(xPrevious)),29*yPrevious/np.max(yPrevious), color='black', s=3)
#         plt.show()

#         allSegments = alg.newFullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
#         plt.scatter(np.linspace(0, 249, len(xNew)),29*yNew/np.max(yNew), color='black', s=3)
#         plt.show()
#         break
#     xNew = alg.reduceDataDecimate(x, sampleFactor)
#     yNew = alg.reduceDataDecimate(y, sampleFactor)