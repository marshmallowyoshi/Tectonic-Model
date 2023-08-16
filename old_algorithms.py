import numpy as np
from scipy.signal import find_peaks, savgol_filter, resample, decimate
import matplotlib.pyplot as plt
from matplotlib import cm
import pywt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad

def firstSegment(x, y, contWav):

    # minimumLength = np.sqrt((np.max(y) - np.min(y))**2 + (x[np.argmax(y)] - x[np.argmin(y)])**2)/20


    minMax = minMaxArray(contWav)
    maxIndex = np.argwhere(minMax==1)
    minIndex = np.argwhere(minMax==-1)

    # find maximum highest throw value that corresponds with a peak in wavelength
    maxDist = maxIndex.T[1]
    yWavMaxInd = [y[i] for i in maxDist]
    waveletPeakIdx = maxDist[np.argmax(yWavMaxInd)]
    segPeak = waveletPeakIdx

    
    # maxY = np.argmax(y)
    # waveletsOnMaxY = contWav[0][:,maxY]
    # peakFrequencyIndex = np.argmax(waveletsOnMaxY)
    # peakFrequency = contWav[1][peakFrequencyIndex]
    # segPeak= peakFrequencyIndex



    firstSegFrequencyIndex = np.argmax(contWav[0][:,segPeak])
    firstSegFrequencyAmplitude = contWav[0][firstSegFrequencyIndex,:]

    localMinima = np.insert(np.asarray([0,len(x)-1]),1,find_peaks(np.asarray(savgol_filter(firstSegFrequencyAmplitude, 5, 3, mode='nearest')) *-1,width=8)[0])
    # newMinima = []
    # for i in localMinima:
    #     length = np.sqrt((x[segPeak] - x[i])**2 + (y[segPeak] - y[i])**2)
    #     if length < minimumLength:
    #         newMinima.append(i)
    # localMinima = np.asarray(newMinima)
    if len(localMinima) < 3:
        for i in np.flip(contWav[0], axis=0):
            localMinima = np.insert(np.asarray([0,len(x)-1]),1,find_peaks(np.asarray(i) *-1,width=8)[0])
            # for i in localMinima:
            #     length = np.sqrt((x[segPeak] - x[i])**2 + (y[segPeak] - y[i])**2)
            #     if length < minimumLength:
            #         newMinima.append(i)
            if len(localMinima) < 3:
                continue
            else:
                break
    # if len(localMinima) < 1:
    #     localMinima = np.asarray([0,len(x)-1])
    firstMinimum = localMinima[(np.abs(localMinima-segPeak)).argmin()]
    if firstMinimum < segPeak:
        localMinima = [localMinima[i] for i in range(len(localMinima)) if localMinima[i] > segPeak]
        if len(localMinima) == 0:
            secondMinimum = len(x)-1
    elif firstMinimum > segPeak:
        localMinima = [localMinima[i] for i in range(len(localMinima)) if localMinima[i] < segPeak]
        if len(localMinima) == 0:
            secondMinimum = 0
    else:
        print('Error: Zero sized segment')
        return None, None, None
    if len(localMinima) == 0:
        pass
    else:
        secondMinimum = localMinima[(np.abs(localMinima-segPeak)).argmin()]


    # firstMinimum = gradientDescent(y, firstMinimum)
    # secondMinimum = gradientDescent(y, secondMinimum)
    lower = np.min([firstMinimum,secondMinimum])
    upper = np.max([firstMinimum,secondMinimum])
    lower = lower + np.argmin(y[lower:segPeak])
    upper = segPeak + np.argmin(y[segPeak:upper]) 

    firstMinimum = lower
    secondMinimum = upper
    return firstMinimum, segPeak, secondMinimum

def nextSegment(x, y, firstMinimum, secondMinimum):
    if firstMinimum < 4:
        nextFirstMinimum1, nextPeak1, nextSecondMinimum1 = None, None, None
    else:
        newy1 = y[:firstMinimum]
        newx1 = x[:firstMinimum]
        contWav1 = waveletGen(newx1, newy1)
        nextFirstMinimum1, nextPeak1, nextSecondMinimum1 = firstSegment(newx1, newy1, contWav1)

    if secondMinimum > len(y)-4:
        nextFirstMinimum2, nextPeak2, nextSecondMinimum2 = None, None, None
    else:
        newy2 = y[secondMinimum:]
        newx2 = x[secondMinimum:]
        contWav2 = waveletGen(newx2, newy2)
        nextFirstMinimum2, nextPeak2, nextSecondMinimum2 = firstSegment(newx2, newy2, contWav2)
        if nextFirstMinimum2 == None:
            return ([nextFirstMinimum1, nextPeak1, nextSecondMinimum1], [nextFirstMinimum2, nextPeak2, nextSecondMinimum2])
        nextFirstMinimum2+=secondMinimum
        nextPeak2+=secondMinimum
        nextSecondMinimum2+=secondMinimum
    return ([nextFirstMinimum1, nextPeak1, nextSecondMinimum1], [nextFirstMinimum2, nextPeak2, nextSecondMinimum2])

def oldFullSegmentAnalysis(x, y, bands=(1,31), sampling_period=0.1, wtype='mexh'):
    contWav = waveletGen(x, y, bands, sampling_period, wtype)
    firstMinimum, peak, secondMinimum = firstSegment(x, y, contWav) 
    band1 = [None, None, None]
    band2 = [firstMinimum, peak, secondMinimum]

    allSegments = [[firstMinimum, peak, secondMinimum]]
    while True:
        
        if None not in band1:
            allSegments.append(band1)
        if None not in band2:
            allSegments.append(band2)

        newSegment = nextSegment(x, y, np.min(np.asarray(allSegments)), np.max(np.asarray(allSegments)))
        band1 = newSegment[0]
        band2 = newSegment[1]
        if None in band1 and None in band2:
            break


    waveletPlot(x,y,contWav)
    for seg in allSegments:
        segmentPlot(seg, y, contWav)
    title = 'Full Segment Analysis\nConstants: wtype=' + str(wtype) + ' sampling_period=' + str(sampling_period) + ' bands=' + str(bands[1]-bands[0])
    plt.title(title)

def fullSegmentAnalysis(x, y, bands=(1,31), sampling_period=0.1, wtype='mexh', name='Unknown'):
    allSegments = []
    ignoredRanges = []
    IndexInSegment = np.full(len(y), False)
    while True:
        


        allIndexInAllSegments = []
        for segment in allSegments:
            allIndexInAllSegments += list(np.arange(segment[0], segment[2]))
        for ignoredRange in ignoredRanges:
            allIndexInAllSegments += list(np.arange(ignoredRange[0], ignoredRange[-1]))

        for index, value in enumerate(IndexInSegment):
            if index in allIndexInAllSegments:
                IndexInSegment[index] = True

        # print(IndexInSegment)

        rangesToScan = []
        startPoints = []
        endPoints = []
        previous = True
        for index, value in enumerate(IndexInSegment):
            if value == False:
                if previous == True:
                    startPoints.append(index)
                if index == len(IndexInSegment) - 1:
                    endPoints.append(index)
            if value == True:
                if previous == False:
                    endPoints.append(index)
            previous = value



        for i in np.arange(len(startPoints)):
            rangesToScan.append((startPoints[i], endPoints[i]))
        significantRangesToScan = [i for i in rangesToScan if i[1] - i[0] > int(0.1 * len(x))]

        if len(significantRangesToScan) == 0:
            break
            print("NO MORE SEGMENTS")


        for range in significantRangesToScan:
            rangeX = x[range[0]:range[1]]
            rangeY = y[range[0]:range[1]]
            contWav = waveletGen(x[range[0]:range[1]], y[range[0]:range[1]])
            newSegmentRange = firstSegment(x[range[0]:range[1]], y[range[0]:range[1]], contWav)
            if None in newSegmentRange:
                ignoredRanges.append(range)
            else:
                trueNewSegmentRange = [newSegmentRange[0] + range[0], newSegmentRange[1] + range[0], newSegmentRange[2] + range[0]]
                allSegments.append(trueNewSegmentRange)

    waveletPlot(x, y, waveletGen(x,y))
    for seg in allSegments:
        segmentPlot(seg, y, waveletGen(x,y))
    plt.title('Full Segment Analysis of '+ name + ' fault \nConstants: wtype=' + str(wtype) + ' sampling_period=' + str(sampling_period) + ' bands=' + str(bands[1]-bands[0]))
