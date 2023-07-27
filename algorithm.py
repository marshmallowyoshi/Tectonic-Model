import numpy as np
from scipy.signal import find_peaks, savgol_filter, resample, decimate
import matplotlib.pyplot as plt
from matplotlib import cm
import pywt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad

def waveletGen(x,y, bands=(1,31), sampling_period=0.1, wtype='mexh'):
    scales = np.arange(bands[0],bands[1])
    contWav = pywt.cwt(y, scales, wtype, sampling_period=sampling_period)
    return contWav

def gradientDescent(y, startIdx):
    newIdx = startIdx

    while True:
        if newIdx >= len(y)-1:
            oldIdx = newIdx
            return newIdx
        else:
            right = -y[newIdx] + y[newIdx+1]
         
        if newIdx <= 0:
            oldIdx = newIdx
            return newIdx
        else:
            left = -y[newIdx] + y[newIdx-1]

        if left > 0 and right > 0:
            return newIdx
        if left < right:
            if left >= 0:
                return newIdx
            else:
                newIdx = newIdx - 1
        elif left > right:
            if right >= 0:
                return newIdx
            else:
                newIdx = newIdx + 1
        else:
            newIdx = newIdx + 1

def minMaxArray(contWav):
    newArray = np.empty(contWav[0].shape)
    newArray[:] = np.nan
    for i in range(contWav[0].shape[0]):
        maxIndex = np.argmax(contWav[0][i])
        minIndex = np.argmin(contWav[0][i])
        newArray[i,maxIndex] = 1
        newArray[i,minIndex] = -1
    return newArray

def segmentWaveletCorrelator(x, y, contWav):
    pass

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

def normalizeThrow(x, y, contWav):
    # newSamplesDistance = np.linspace(np.min(x), np.max(x), contWav[0].shape[1])
    # newTest = np.interp(newSamplesDistance, newSamplesDistance, y)
    newTestNorm = np.asarray(y) * 0.8 * contWav[0].shape[0] / np.max(y)
    return newTestNorm

def waveletPlot(x, y, contWav):
    newTestNorm = normalizeThrow(x, y, contWav)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(contWav[0].shape[1]), newTestNorm, color='black')
    im = ax.imshow(contWav[0], interpolation='nearest', aspect='auto', origin='lower')
    plt.colorbar(im)
    plt.yticks(np.arange(contWav[0].shape[0])[::-1],np.round(contWav[1],3))
    plt.xlabel('Inline No.')
    plt.ylabel('Wavelet Wavelength')
    return fig, ax

def segmentPlot(segmentLocations, y, contWav):
    newTestNorm = np.asarray([y[segmentLocations[0]], y[segmentLocations[1]], y[segmentLocations[2]]]) * (0.8 * contWav[0].shape[0]/np.max(y))
    plt.plot([segmentLocations[0],segmentLocations[1],segmentLocations[2]], newTestNorm, color='pink')
    plt.stem([segmentLocations[0],segmentLocations[1],segmentLocations[2]], newTestNorm, linefmt='pink',markerfmt=' ')

def waveletPlot3D(x, y, contWav):
    wavX = np.reshape(np.asarray([i for (i,j), value in np.ndenumerate(contWav[0])]), (contWav[0].shape[0],contWav[0].shape[1]))
    wavY = np.reshape(np.asarray([j for (i,j), value in np.ndenumerate(contWav[0])]), (contWav[0].shape[0],contWav[0].shape[1]))
    wavZ = contWav[0]


    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(wavX, wavY, wavZ, cmap=cm.coolwarm)
    return fig, ax

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

def newFirstSegment(x, y, contWav, fullx, fully, fullContWav, offset, peakRateThreshold=0.03):
    # find dips and peaks for each wavelet wavelength
    dips = [list(find_peaks(wavelength*-1)[0]) for wavelength in fullContWav[0]]
    peaks = [list(find_peaks(wavelength)[0]) for wavelength in fullContWav[0]]
    offsetEnd = offset+len(y)
    # remove dips and peaks that are outside of the scan range
    allowedDips = [[j for j in i if offsetEnd > j > offset] for i in dips]
    allowedPeaks = [[j for j in i if offsetEnd > j > offset] for i in peaks]

    for i in range(len(allowedDips)):
        # add start and end of scan range to dips when needed
        if contWav[0][i][0] < contWav[0][i][1]:
            allowedDips[i] = [offset] + allowedDips[i]
        if contWav[0][i][-1] < contWav[0][i][-2]:
            allowedDips[i] = allowedDips[i] + [offsetEnd]

    # getting peak height and filtering unusable wavelengths
    fullPeakRate = [len(i)/len(fully) for i in peaks]
    allowedWavelengths = [i for i in range(len(fullPeakRate)) if fullPeakRate[i] < peakRateThreshold and len(allowedPeaks[i]) > 0]
    fullSegmentWavelengths = [i for i in allowedWavelengths if len(allowedDips[i]) > 1 and len(allowedPeaks[i]) > 0]
    if fullSegmentWavelengths == []:
        return None, None, None
    peakHeight = [(idx, value[np.argmax([fully[j] for j in value])], np.max([fully[j] for j in value])) for idx, value in enumerate(allowedPeaks) if len(value) > 0]

    firstSegFrequencyIndex = fullSegmentWavelengths[0]
    for i in peakHeight:
        # segPeak is found by getting peak on the wavelength chosen
        if i[0] == firstSegFrequencyIndex:
            segPeak = i[1] - offset

    firstSegFrequencyAmplitude = fullContWav[0][firstSegFrequencyIndex,offset:offsetEnd]
    localMinima = np.insert(np.asarray([0,len(x)-1]),1,find_peaks(np.asarray(savgol_filter(firstSegFrequencyAmplitude, 5, 3, mode='nearest')) *-1,width=8)[0])
    
    # if len(localMinima) < 3:
    #     for i in np.flip(contWav[0], axis=0):
    #         localMinima = np.insert(np.asarray([0,len(x)-1]),1,find_peaks(np.asarray(i) *-1,width=8)[0])
    #         if len(localMinima) < 3:
    #             continue
    #         else:
    #             break

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
        print('Warning: Zero sized segment')
        return None, None, None
    if len(localMinima) == 0:
        pass
    else:
        secondMinimum = localMinima[(np.abs(localMinima-segPeak)).argmin()]


    firstMinimum = gradientDescent(savgol_filter(y, 11, 3, mode='nearest'), firstMinimum)
    secondMinimum = gradientDescent(savgol_filter(y, 11, 3, mode='nearest'), secondMinimum)
    # segPeak = gradientDescent(savgol_filter(y*-1, 11, 3, mode='nearest'), segPeak)
    # lower = np.min([firstMinimum,secondMinimum])
    # upper = np.max([firstMinimum,secondMinimum])
    lower = np.min([firstMinimum,segPeak,secondMinimum])
    upper = np.max([firstMinimum,segPeak,secondMinimum])
    segPeak = np.argmax(y[lower:upper]) + lower
    if segPeak != lower and segPeak != upper:
        lower = lower + np.argmin(y[lower:segPeak])
        upper = segPeak + np.argmin(y[segPeak:upper]) 
    else:
        print('Warning: Bad Segment Shape')
        return None, None, None
    firstMinimum = lower
    secondMinimum = upper
    return firstMinimum, segPeak, secondMinimum

def resampleToPeak(x, y, wtype='mexh', bands=(1,31), sampling_period=0.1):
    peakWavelength = pywt.scale2frequency(wtype, bands[0])/sampling_period
    return resample(x, int(peakWavelength*100)), resample(y, int(peakWavelength*100))

def newFullSegmentAnalysis(x, y, bands=(1,31), sampling_period=0.1, wtype='mexh', name='Unknown', peakRateThreshold=0.03, join=False, segmentMode='flat', degree=3, plot=False):
    peakWavelength = pywt.scale2frequency(wtype, bands[0])/sampling_period
    
    x = resample(x, int(peakWavelength*100))
    y = resample(y, int(peakWavelength*100))
    
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
            contWav = waveletGen(x[range[0]:range[1]], y[range[0]:range[1]], bands=bands, sampling_period=sampling_period, wtype=wtype)
            fullContWav = waveletGen(x, y, bands=bands, sampling_period=sampling_period, wtype=wtype)
            newSegmentRange = newFirstSegment(x[range[0]:range[1]], y[range[0]:range[1]], contWav, x, y, fullContWav, range[0], peakRateThreshold=peakRateThreshold)
            if None in newSegmentRange:
                ignoredRanges.append(range)
            else:
                trueNewSegmentRange = [newSegmentRange[0] + range[0], newSegmentRange[1] + range[0], newSegmentRange[2] + range[0]]
                allSegments.append(trueNewSegmentRange)

    # sort segments
    allSegments = sorted(allSegments, key=lambda a: a[1])
    # if no segments take a segment across entire range
    if len(allSegments) == 0:
        allSegments = [[0,np.argmax(y),len(x)-1]]
    else:
        # fill in gaps between segments
        if join == True:
            newSegments = []
            if allSegments[0][0] != 0:
                interMinimum = np.argmin(y[:allSegments[0][0]])
            else:
                interMinimum = allSegments[0][0]
            for i in np.arange(len(allSegments)-1):
                previousMinimum = interMinimum
                if allSegments[i][2] < allSegments[i+1][0]:
                    interMinimum = np.argmin(y[allSegments[i][2]:allSegments[i+1][0]]) + allSegments[i][2]
                else:
                    interMinimum = allSegments[i][2]
                newSegments.append([previousMinimum, allSegments[i][1], interMinimum])
            if allSegments[-1][2] != len(x)-1:
                newSegments.append([interMinimum, allSegments[-1][1], np.argmin(y[allSegments[-1][2]:])+allSegments[-1][2]])
            else:
                newSegments.append([interMinimum, allSegments[-1][1], allSegments[-1][2]])
            allSegments = newSegments

    if plot == True:
        waveletPlot(x, y, waveletGen(x,y))
        if segmentMode == 'flat':
            for seg in allSegments:
                segmentPlot(seg, y, waveletGen(x,y))
        elif segmentMode == 'poly':
            regressionPlot(allSegments, allSegmentsPolynomialRegression(y, allSegments, degree=degree), y, waveletGen(x,y))
        elif segmentMode == 'all':
            for seg in allSegments:
                segmentPlot(seg, y, waveletGen(x,y))
            regressionPlot(allSegments, allSegmentsPolynomialRegression(y, allSegments, degree=degree), y, waveletGen(x,y))
        else:
            print('Warning: segmentMode not recognised, no segments plotted')
        plt.title('Full Segment Analysis of '+ name + ' fault \nConstants: wtype=' + str(wtype) + ' sampling_period=' + str(sampling_period) + ' bands=' + str(bands[1]-bands[0]) + ' peakRateThreshold=' + str(peakRateThreshold))
    if segmentMode == 'poly' or segmentMode == 'all':
        return allSegments, allSegmentsPolynomialRegression(y, allSegments, degree=degree)
    else:
        return allSegments

def segmentPolynomialRegression(y, start, peak, end, degree=3):
    segmentYThrow = y[start:end+1]

    segmentXThrow = np.arange(0, len(segmentYThrow), 1)

    xWeightsStart = np.full(100,segmentXThrow[0])
    yWeightsStart = np.full(100,segmentYThrow[0])
    xWeightsEnd = np.full(100,segmentXThrow[-1])
    yWeightsEnd = np.full(100,segmentYThrow[-1])
    xWeightsPeak = np.full(10,segmentXThrow[peak-start])
    yWeightsPeak = np.full(10,segmentYThrow[peak-start])

    
    segmentXThrowWeighted = np.concatenate((xWeightsStart, segmentXThrow, xWeightsEnd))
    segmentYThrowWeighted = np.concatenate((yWeightsStart, segmentYThrow, yWeightsEnd))
    segmentXThrowWeighted = np.insert(segmentXThrowWeighted, peak-start+100, xWeightsPeak)
    segmentYThrowWeighted = np.insert(segmentYThrowWeighted, peak-start+100, yWeightsPeak)

    poly = PolynomialFeatures(degree, include_bias=False)

    poly_features = poly.fit_transform(np.asarray(segmentXThrowWeighted).reshape(-1,1))

    poly_reg_model = LinearRegression()
    return poly_reg_model.fit(poly_features, segmentYThrowWeighted), segmentXThrowWeighted, poly_features

def allSegmentsPolynomialRegression(y, allSegments, degree=3):
    return [segmentPolynomialRegression(y, segment[0], segment[1], segment[2], degree=degree) for segment in allSegments]

def regressionPlot(allSegments, segmentRegression, y, contWav):
    for index, segment in enumerate(allSegments):
        y_predicted = segmentRegression[index][0].predict(segmentRegression[index][2])
        plt.plot(segmentRegression[index][1] + segment[0], y_predicted * 0.8 * contWav[0].shape[0] / np.max(y), color='red')
    return None

def reduceDataResample(x, factor):
    return resample(x, int(len(x)/factor))

def reduceDataDecimate(x, factor):
    return decimate(x, int(factor))

def closestArg(array, val):
    return np.abs(np.asarray(array)-val).argmin()

def findMinimumSampleRate(x, y, samplingMode='decimate', bands=(1,31), sampling_period=0.1, wtype='mexh', name='Unknown', peakRateThreshold=0.03, degree=3, plot=False):
    allSegmentsFull = newFullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat')
    xNew = x
    yNew = y

    sampleFactor = 1
    segmentCount = 0
    while True:
        # Gradual increase in sample reduction
        sampleFactor = sampleFactor+1
        
        print('Number of Samples: ', len(xNew))
        allSegments = newFullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat', plot=False)
        if len(allSegments) < len(allSegmentsFull) or len(xNew) < 4:
            # reproduce previous sample rate
            if samplingMode == 'decimate':
                xPrevious = reduceDataDecimate(x, sampleFactor-2)
                yPrevious = reduceDataDecimate(y, sampleFactor-2)
            else:
                xPrevious = reduceDataResample(x, sampleFactor-2)
                yPrevious = reduceDataResample(y, sampleFactor-2)
            print('Starting Samples: ', len(x),'Minimum Samples: ', len(xPrevious))
            # Find smallest segment length
            oldSegments = newFullSegmentAnalysis(xPrevious,yPrevious, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)[0]
            xPreviousRescaled = np.linspace(0,250,len(xPrevious))
            allSizes = [closestArg(xPreviousRescaled, i[2])-closestArg(xPreviousRescaled, i[0]) for i in oldSegments]
            percentage = 100/np.min(allSizes)

            print('Sample as Percentage of Fault Length: ', percentage, '%')

            if plot == True:
                # One plot at minimum sample rate and another at the step below
                plt.scatter(np.linspace(0, 249, len(xPrevious)),29*yPrevious/np.max(yPrevious), color='black', s=3)
                plt.title("Minimum Sample Rate")
                plt.show()

                allSegments = newFullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
                plt.scatter(np.linspace(0, 249, len(xNew)),29*yNew/np.max(yNew), color='black', s=3)
                plt.title("Below Minimum Sample Rate")
                plt.show()
            break
        # go to next sample rate
        if samplingMode == 'decimate':
            xNew = reduceDataDecimate(x, sampleFactor)
            yNew = reduceDataDecimate(y, sampleFactor)
        else:
            xNew = reduceDataResample(x, sampleFactor)
            yNew = reduceDataResample(y, sampleFactor)
    return percentage

def integralDifference(poly1, poly2, bounds):
    """ Returns the percentage difference between the integrals of two polynomials
    Output:
    [0] : Percentage difference between the integrals of the two polynomials
    [1] : Difference between the integrals of the two polynomials
    [2] : Integral of the first polynomial
    [3] : Integral of the second polynomial"""
    coeffs_1 = np.asarray(list(poly1.coef_)[::-1] + [poly1.intercept_])
    coeffs_2 = np.asarray(list(poly2.coef_)[::-1] + [poly2.intercept_])

    x_segment = np.linspace(bounds[0], bounds[1], 100)
    difference = coeffs_1 - coeffs_2

    
    areaDifference = quad(lambda x: cubicAbs(x, difference[0], difference[1], difference[2], difference[3]), bounds[0], bounds[1])
    difference_1 = quad(lambda x: cubicAbs(x, coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3]), bounds[0], bounds[1])
    difference_2 = quad(lambda x: cubicAbs(x, coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3]), bounds[0], bounds[1])
    
    areaDifferencePercentage = 100 * areaDifference[0]/np.max([difference_1[0], difference_2[0]])

    return (areaDifferencePercentage, areaDifference[0], difference_1[0], difference_2[0])

def cubicAbs(x,cube,square,linear,constant):
    return np.abs(cube*x**3 + square*x**2 + linear*x + constant)