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
allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=False)
contWav = alg.waveletGen(x,y)
plt.plot(np.arange(250),signal.resample(y, 250) * 0.8 * contWav[0].shape[0] / np.max(y), color='black', linewidth=0.5)
alg.regressionPlot(allSegments, allCurves, y, contWav)
plt.show()

############################################################################################################################################

# allSegments, allCurves = alg.newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
# plt.show()

