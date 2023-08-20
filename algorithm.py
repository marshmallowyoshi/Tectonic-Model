"""
Contains all functions required to extract fault data from input signals.
"""


import numpy as np
import pywt
from scipy.signal import find_peaks, savgol_filter, resample
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


import matplotlib.pyplot as plt
from matplotlib import cm




def waveletGen(x,y, bands=(1,31), sampling_period=0.1, wtype='mexh'):
    """ Performs a continuous wavelet transform on a signal.
    Input:
    x : time or distance axis
    y : signal amplitude
    bands : tuple of (min,max) range of bands to use
    sampling_period : sampling period of the signal
    wtype : wavelet type to use"""
    
    scales = np.arange(bands[0],bands[1])
    contWav = pywt.cwt(y, scales, wtype, sampling_period=sampling_period)
    return contWav

def gradientDescent(y, startIdx):
    """ Finds the index in y of a local minimum starting from index startIdx"""
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

def normalizeThrow(x, y, contWav):
    """ Normalize throw data to wavelet data for plotting purposes"""
    newTestNorm = np.asarray(y) * 0.8 * contWav[0].shape[0] / np.max(y)
    return newTestNorm

def waveletPlot(x, y, contWav):
    """ Plots a wavelet transform of a signal with signal overlayed"""
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
    """ Plots segment predictions from three input points for each segment"""
    newTestNorm = np.asarray([y[segmentLocations[0]], y[segmentLocations[1]], y[segmentLocations[2]]]) * (0.8 * contWav[0].shape[0]/np.max(y))
    plt.plot([segmentLocations[0],segmentLocations[1],segmentLocations[2]], newTestNorm, color='red')
    plt.stem([segmentLocations[0],segmentLocations[1],segmentLocations[2]], newTestNorm, linefmt='red',markerfmt=' ')

def waveletPlot3D(contWav):
    """ 3D surface plot of a continuous wavelet transform"""
    wavX = np.reshape(np.asarray([i for (i,j), value in np.ndenumerate(contWav[0])]), (contWav[0].shape[0],contWav[0].shape[1]))
    wavY = np.reshape(np.asarray([j for (i,j), value in np.ndenumerate(contWav[0])]), (contWav[0].shape[0],contWav[0].shape[1]))
    wavZ = contWav[0]


    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(wavX, wavY, wavZ, cmap=cm.copper)
    return fig, ax

def firstSegment(x, y, contWav, fullx, fully, fullContWav, offset, peakRateThreshold=0.03):
    """ Detect a fault segment within a range of a signal."""
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

def resampleToPeak(x, y, wtype='mexh', bands=(1,31), sampling_period=0.1, samples_per_period=100, sample_rate_override=None):
    """ Apply interpolation to match a signal to the peak wavelength of a wavelet transform.
    When sample_rate_override is used, the signal is resampled to the specified number of samples."""
    peakWavelength = pywt.scale2frequency(wtype, bands[0])/sampling_period
    if sample_rate_override != None:
        totalSamples = int(sample_rate_override)
    else:
        totalSamples = int(peakWavelength*samples_per_period)
    f = interp1d(x, y)
    return np.linspace(np.min(x), np.max(x), totalSamples), f(np.linspace(np.min(x), np.max(x), totalSamples))

def fullSegmentAnalysis(x, y, bands=(1,31), sampling_period=0.1, wtype='mexh', name='Unknown', peakRateThreshold=0.03, join=False, segmentMode='flat', degree=3, plot=False, title='long', samples_per_period=100):
    """ The main function. Takes in fault shape data in a form with fault throw on the y axis, and distance along fault on the x axis.
        Returns a list of estimated segments, with their start, peak and end points. 
        Poly mode also returns a list of polynomial regression models for each segment."""
    x, y = resampleToPeak(x, y, wtype=wtype, bands=bands, sampling_period=sampling_period, samples_per_period=samples_per_period)
    # x = resample(x, int(peakWavelength*100))
    # y = resample(y, int(peakWavelength*100))
    
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


        for range in significantRangesToScan:
            rangeX = x[range[0]:range[1]]
            rangeY = y[range[0]:range[1]]
            contWav = waveletGen(x[range[0]:range[1]], y[range[0]:range[1]], bands=bands, sampling_period=sampling_period, wtype=wtype)
            fullContWav = waveletGen(x, y, bands=bands, sampling_period=sampling_period, wtype=wtype)
            newSegmentRange = firstSegment(x[range[0]:range[1]], y[range[0]:range[1]], contWav, x, y, fullContWav, range[0], peakRateThreshold=peakRateThreshold)
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
        if title=='long':
            plt.title('Full Segment Analysis of '+ name + ' fault \nConstants: wtype=' + str(wtype) + ' sampling_period=' + str(sampling_period) + ' bands=' + str(bands[1]-bands[0]) + ' peakRateThreshold=' + str(peakRateThreshold))
        elif title=='short':
            plt.title('Full Segment Analysis of '+ name + ' fault')
    if segmentMode == 'poly' or segmentMode == 'all':
        return allSegments, allSegmentsPolynomialRegression(y, allSegments, degree=degree)
    else:
        return allSegments

def segmentPolynomialRegression(y, start, peak, end, degree=3):
    """ Obtain a cubic function that fits a segment of a signal"""
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
    """ Repeat polynomial regression for all input segments"""
    return [segmentPolynomialRegression(y, segment[0], segment[1], segment[2], degree=degree) for segment in allSegments]

def regressionPlot(allSegments, segmentRegression, y, contWav):
    """ Plot results of regression"""
    for index, segment in enumerate(allSegments):
        y_predicted = segmentRegression[index][0].predict(segmentRegression[index][2])
        plt.plot(segmentRegression[index][1] + segment[0], y_predicted * 0.8 * contWav[0].shape[0] / np.max(y), color='red')
    return None

def reduceDataResample(x, factor):
    """ Resample a signal to a lower sample rate"""
    return resample(x, int(len(x)/factor))

def reduceDataDecimate(x, factor):
    """ Decimate a signal to a lower sample rate"""
    return x[0::int(factor)]

def closestArg(array, val):
    return np.abs(np.asarray(array)-val).argmin()

def findMinimumSampleRate(x, y, samplingMode='interpolate', bands=(1,31), sampling_period=0.1, wtype='mexh', name='Unknown', peakRateThreshold=0.03, degree=3, plot=False):
    allSegmentsFull = fullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat')
    xNew = x
    yNew = y

    sampleFactor = 1
    segmentCount = 0
    while True:
        # Gradual increase in sample reduction
        sampleFactor = sampleFactor+1
        
        # print('Number of Samples: ', len(xNew))
        allSegments = fullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='flat', plot=False)
        if len(allSegments) < len(allSegmentsFull) or len(xNew) < 4:
            # reproduce previous sample rate
            if samplingMode == 'decimate':
                xPrevious = reduceDataDecimate(x, sampleFactor-2)
                yPrevious = reduceDataDecimate(y, sampleFactor-2)
            elif samplingMode == 'resample':
                xPrevious = reduceDataResample(x, sampleFactor-2)
                yPrevious = reduceDataResample(y, sampleFactor-2)
            elif samplingMode == 'interpolate':
                #interpolate mode
                xPrevious, yPrevious = resampleToPeak(x, y, wtype=wtype, bands=bands, sampling_period=sampling_period, sample_rate_override=int(len(x) * 0.9**(sampleFactor-1)))
            else:
                print('Warning: samplingMode not recognised')

            print('Starting Samples: ', len(x),'Minimum Samples: ', len(xPrevious))
            # Find smallest segment length
            oldSegments = fullSegmentAnalysis(xPrevious,yPrevious, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)[0]
            xPreviousRescaled = np.linspace(0,250,len(xPrevious))
            allSizes = [closestArg(xPreviousRescaled, i[2])-closestArg(xPreviousRescaled, i[0]) for i in oldSegments]
            percentage = 100/np.max(allSizes)
            percentageTotal = 100/np.sum(allSizes)

            print('Sample as Percentage of Fault Length: ', percentage, '%')
            print('Sample as percentage of Total: ', percentageTotal, '%')

            if plot == True:
                # One plot at minimum sample rate and another at the step below
                plt.scatter(np.linspace(0, 249, len(xPrevious)),29*yPrevious/np.max(yPrevious), color='black', s=3)
                plt.title("Minimum Sample Rate")
                plt.show()

                allSegments = fullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
                plt.scatter(np.linspace(0, 249, len(xNew)),29*yNew/np.max(yNew), color='black', s=3)
                plt.title("Below Minimum Sample Rate")
                plt.show()
            break
        # go to next sample rate
        if samplingMode == 'decimate':
            xNew = reduceDataDecimate(x, sampleFactor)
            yNew = reduceDataDecimate(y, sampleFactor)
        elif samplingMode == 'resample':
            xNew = reduceDataResample(x, sampleFactor)
            yNew = reduceDataResample(y, sampleFactor)
        elif samplingMode == 'interpolate':
            #interpolate mode
            xNew, yNew = resampleToPeak(x, y, wtype=wtype, bands=bands, sampling_period=sampling_period, sample_rate_override=int(len(x) * 0.9**(sampleFactor-1)))
        else:
            print('Error: samplingMode not recognised')
        # fullSegmentAnalysis(xNew,yNew, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=True)
        # plt.show()
    return (percentage, percentageTotal)

def integralDifference(poly1, poly2, bounds1, bounds2):
    """ Returns the percentage difference between the integrals of two polynomials
    Output:
    [0] : Percentage difference between the integrals of the two polynomials
    [1] : Difference between the integrals of the two polynomials
    [2] : Integral of the first polynomial
    [3] : Integral of the second polynomial"""

    # Limit range to smallest bounding box
    bounds = (np.max([bounds1[0], bounds2[0]]), np.min([bounds1[1], bounds2[1]]))

    if bounds1[0] != bounds[0]:
        offset1 = -1*bounds[0] + bounds1[0]
    else:
        offset1 = 0
    if bounds2[0] != bounds[0]:
        offset2 = -1*bounds[0] + bounds2[0]
    else:
        offset2 = 0

    coeffs_1 = np.asarray(list(poly1.coef_)[::-1] + [poly1.intercept_])
    coeffs_2 = np.asarray(list(poly2.coef_)[::-1] + [poly2.intercept_])


    
    areaDifference = quad(lambda x: cubicAbsDiff(x, coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3], coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3], offset1=offset1, offset2=offset2), 0, bounds[1]-bounds[0])
    difference_1 = quad(lambda x: cubicAbs(x, coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3], offset=offset1), 0, bounds[1]-bounds[0])
    difference_2 = quad(lambda x: cubicAbs(x, coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3], offset=offset2), 0, bounds[1]-bounds[0])
    
    areaDifferencePercentage = 100 * areaDifference[0]/np.max([difference_1[0], difference_2[0]])

    return (areaDifferencePercentage, areaDifference[0], difference_1[0], difference_2[0])

def cubicAbs(x,cube,square,linear,constant,offset=0):
    """ Returns the absolute value of a cubic polynomial"""
    return np.abs(cube*(x+offset)**3 + square*(x+offset)**2 + linear*(x+offset) + constant)

def cubicAbsDiff(x,cube1,square1,linear1,constant1,cube2,square2,linear2,constant2,offset1=0,offset2=0):
    """ Returns the absolute value of the difference between two cubic polynomials with offset values"""
    return np.abs(cube1*(x+offset1)**3 + square1*(x+offset1)**2 + linear1*(x+offset1) + constant1 - (cube2*(x+offset2)**3 + square2*(x+offset2)**2 + linear2*(x+offset2) + constant2))

def allSegmentsIntegral(allSegments, allSegments2, segmentRegression, segmentRegression2):
    return [integralDifference(segmentRegression[i][0], segmentRegression2[i][0], (allSegments[i][0], allSegments[i][2]), (allSegments2[i][0], allSegments2[i][2])) for i in range(len(allSegments))]

def cubicPlot(allSegments, polynomialRegression, numPoints=100, color='red', label='Segments'):
    for i in range(len(allSegments)):
        xPlot = np.linspace(0, allSegments[i][2]-allSegments[i][0], numPoints)
        f = lambda x: polynomialRegression[i][0].coef_[2]*x**3 + polynomialRegression[i][0].coef_[1]*x**2 + polynomialRegression[i][0].coef_[0]*x + polynomialRegression[i][0].intercept_
        yPlot = f(xPlot)
        xPlot = xPlot + allSegments[i][0]
        if i == 0:
            plt.plot(xPlot, yPlot, color=color, label=label)
        else:
            plt.plot(xPlot, yPlot, color=color)
    return None