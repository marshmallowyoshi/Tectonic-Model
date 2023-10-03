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




def wavelet_generator(y, bands=(1,31), sampling_period=0.1, wtype='mexh'):
    """ Performs a continuous wavelet transform on a signal.
    Input:
    x : time or distance axis
    y : signal amplitude
    bands : tuple of (min,max) range of bands to use
    sampling_period : sampling period of the signal
    wtype : wavelet type to use"""
    scales = np.arange(bands[0],bands[1])
    cont_wavelets = pywt.cwt(y, scales, wtype, sampling_period=sampling_period)
    return cont_wavelets

def gradient_descent(y, start_idx: int):
    """ Finds the index in y of a local minimum starting from index startIdx"""
    new_idx = start_idx

    while True:
        if new_idx >= len(y)-1:
            return new_idx
        right = -y[new_idx] + y[new_idx+1]
        if new_idx <= 0:
            return new_idx
        left = -y[new_idx] + y[new_idx-1]

        if left > 0 and right > 0:
            return new_idx
        if left < right:
            if left >= 0:
                return new_idx
            new_idx = new_idx - 1
        elif left > right:
            if right >= 0:
                return new_idx
            new_idx = new_idx + 1
        else:
            new_idx = new_idx + 1

def min_max_array(cont_wavelets):
    new_array = np.empty(cont_wavelets[0].shape)
    new_array[:] = np.nan
    for i in range(cont_wavelets[0].shape[0]):
        max_idx = np.argmax(cont_wavelets[0][i])
        min_idx = np.argmin(cont_wavelets[0][i])
        new_array[i,max_idx] = 1
        new_array[i,min_idx] = -1
    return new_array

def normalize_throw(y, cont_wavelets):
    """ Normalize throw data to wavelet data for plotting purposes"""
    y_normalized = np.asarray(y) * 0.8 * cont_wavelets[0].shape[0] / np.max(y)
    return y_normalized

def wavelet_plot(y, cont_wavelets, skip_wavelets=False):
    """ Plots a wavelet transform of a signal with signal overlayed"""
    if skip_wavelets is False:
        y_normalized = normalize_throw(y, cont_wavelets)
    else:
        y_normalized = y
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(cont_wavelets[0].shape[1]), y_normalized, color='black')
    if skip_wavelets is False:
        im = ax.imshow(cont_wavelets[0], interpolation='nearest', aspect='auto', origin='lower')
        plt.colorbar(im)
        plt.yticks(np.arange(cont_wavelets[0].shape[0])[::-1],np.round(cont_wavelets[1],3))
        plt.xlabel('Inline No.')
        plt.ylabel('Wavelet Wavelength')
    return fig, ax

def segment_plot(segment_locations, y, cont_wavelets):
    """ Plots segment predictions from three input points for each segment"""
    y_normalized = np.asarray([y[segment_locations[0]], y[segment_locations[1]], y[segment_locations[2]]]) * (0.8 * cont_wavelets[0].shape[0]/np.max(y))
    plt.plot([segment_locations[0],segment_locations[1],segment_locations[2]], y_normalized, color='red')
    plt.stem([segment_locations[0],segment_locations[1],segment_locations[2]], y_normalized, linefmt='red',markerfmt=' ')

def wavelet_plot_3d(cont_wavelets):
    """ 3D surface plot of a continuous wavelet transform"""
    x = np.reshape(np.asarray([i for (i,j), value in np.ndenumerate(cont_wavelets[0])]), (cont_wavelets[0].shape[0],cont_wavelets[0].shape[1]))
    y = np.reshape(np.asarray([j for (i,j), value in np.ndenumerate(cont_wavelets[0])]), (cont_wavelets[0].shape[0],cont_wavelets[0].shape[1]))
    z = cont_wavelets[0]


    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(x, y, z, cmap=cm.copper)
    return fig, ax

def first_segment(x, y, cont_wavelets, y_full, fullcont_wavelets, offset, peak_rate_threshold=0.03, show_warnings=False):
    """ Detect a fault segment within a range of a signal."""
    # find dips and peaks for each wavelet wavelength
    dips = [list(find_peaks(wavelength*-1)[0]) for wavelength in fullcont_wavelets[0]]
    peaks = [list(find_peaks(wavelength)[0]) for wavelength in fullcont_wavelets[0]]
    offset_end = offset+len(y)
    # remove dips and peaks that are outside of the scan range
    allowed_dips = [[j for j in i if offset_end > j > offset] for i in dips]
    allowed_peaks = [[j for j in i if offset_end > j > offset] for i in peaks]

    for i in range(len(allowed_dips)):
        # add start and end of scan range to dips when needed
        if cont_wavelets[0][i][0] < cont_wavelets[0][i][1]:
            allowed_dips[i] = [offset] + allowed_dips[i]
        if cont_wavelets[0][i][-1] < cont_wavelets[0][i][-2]:
            allowed_dips[i] = allowed_dips[i] + [offset_end]

    # getting peak height and filtering unusable wavelengths
    full_peak_rate = [len(i)/len(y_full) for i in peaks]
    allowed_wavelengths = [i for i in range(len(full_peak_rate)) if full_peak_rate[i] < peak_rate_threshold and len(allowed_peaks[i]) > 0]
    full_segment_wavelengths = [i for i in allowed_wavelengths if len(allowed_dips[i]) > 1 and len(allowed_peaks[i]) > 0]
    if full_segment_wavelengths == []:
        return None, None, None
    peak_height = [(idx, value[np.argmax([y_full[j] for j in value])], np.max([y_full[j] for j in value])) for idx, value in enumerate(allowed_peaks) if len(value) > 0]

    segment_frequency_idx = full_segment_wavelengths[0]
    for i in peak_height:
        # segment_peak is found by getting peak on the wavelength chosen
        if i[0] == segment_frequency_idx:
            segment_peak = i[1] - offset

    segment_frequency_amplitude = fullcont_wavelets[0][segment_frequency_idx,offset:offset_end]
    local_minima = np.insert(np.asarray([0,len(x)-1]),1,find_peaks(np.asarray(savgol_filter(segment_frequency_amplitude, 5, 3, mode='nearest')) *-1,width=8)[0])

    first_minimum = local_minima[(np.abs(local_minima-segment_peak)).argmin()]
    if first_minimum < segment_peak:
        local_minima = [local_minima[i] for i in range(len(local_minima)) if local_minima[i] > segment_peak]
        if len(local_minima) == 0:
            second_minimum = len(x)-1
    elif first_minimum > segment_peak:
        local_minima = [local_minima[i] for i in range(len(local_minima)) if local_minima[i] < segment_peak]
        if len(local_minima) == 0:
            second_minimum = 0
    else:
        if show_warnings is True:
            print('Warning: Zero sized segment')
        return None, None, None
    if len(local_minima) == 0:
        pass
    else:
        second_minimum = local_minima[(np.abs(local_minima-segment_peak)).argmin()]


    first_minimum = gradient_descent(savgol_filter(y, 11, 3, mode='nearest'), first_minimum)
    second_minimum = gradient_descent(savgol_filter(y, 11, 3, mode='nearest'), second_minimum)

    lower = np.min([first_minimum,segment_peak,second_minimum])
    upper = np.max([first_minimum,segment_peak,second_minimum])
    segment_peak = np.argmax(y[lower:upper]) + lower
    if segment_peak != lower and segment_peak != upper:
        lower = lower + np.argmin(y[lower:segment_peak])
        upper = segment_peak + np.argmin(y[segment_peak:upper]) 
    else:
        if show_warnings is True:
            print('Warning: Bad Segment Shape')
        return None, None, None
    first_minimum = lower
    second_minimum = upper
    return first_minimum, segment_peak, second_minimum

def resample_to_peak(x, y, wtype='mexh', bands=(1,31), sampling_period=0.1, samples_per_period=100, sample_rate_override=None, kind='linear'):
    """ Apply interpolation to match a signal to the peak wavelength of a wavelet transform.
    When sample_rate_override is used, the signal is resampled to the specified number of samples."""
    peak_wavelength = pywt.scale2frequency(wtype, bands[0])/sampling_period
    if sample_rate_override != None:
        total_samples = int(sample_rate_override)
    else:
        total_samples = int(peak_wavelength*samples_per_period)
    f = interp1d(x, y, kind=kind)
    return np.linspace(np.min(x), np.max(x), total_samples), f(np.linspace(np.min(x), np.max(x), total_samples))

def full_segment_analysis(x, y, 
                        bands=(1,31), sampling_period=0.1, wtype='mexh',
                        name='Unknown',
                        peak_rate_threshold=0.03,
                        join=False,
                        segment_mode='flat',
                        degree=3,
                        plot=False,
                        title='long',
                        samples_per_period=100,
                        resampling_kind='linear',
                        show_warnings=False,
                        skip_wavelets=False):
    """ The main function. Takes in fault shape data in a form with fault throw on the y axis, and distance along fault on the x axis.
        Returns a list of estimated segments, with their start, peak and end points. 
        Poly mode also returns a list of polynomial regression models for each segment."""
    
    x, y = resample_to_peak(x, y, wtype=wtype, bands=bands, sampling_period=sampling_period, samples_per_period=samples_per_period, kind=resampling_kind)
    
    all_segments = []
    ignored_ranges = []
    idx_in_segment = np.full(len(y), False)
    while True:
        
        idx_in_all_segments = []
        for segment in all_segments:
            idx_in_all_segments += list(np.arange(segment[0], segment[2]))
        for ignored_range in ignored_ranges:
            idx_in_all_segments += list(np.arange(ignored_range[0], ignored_range[-1]))

        for idx, value in enumerate(idx_in_segment):
            if idx in idx_in_all_segments:
                idx_in_segment[idx] = True

        ranges_to_scan = []
        start_points = []
        endPoints = []
        previous = True
        for idx, value in enumerate(idx_in_segment):
            if value == False:
                if previous == True:
                    start_points.append(idx)
                if idx == len(idx_in_segment) - 1:
                    endPoints.append(idx)
            if value == True:
                if previous == False:
                    endPoints.append(idx)
            previous = value



        for i in np.arange(len(start_points)):
            ranges_to_scan.append((start_points[i], endPoints[i]))
        significant_ranges_to_scan = [i for i in ranges_to_scan if i[1] - i[0] > int(0.1 * len(x))]

        if len(significant_ranges_to_scan) == 0:
            break


        for span in significant_ranges_to_scan:
            cont_wavelets = wavelet_generator(y[span[0]:span[1]], bands=bands, sampling_period=sampling_period, wtype=wtype)
            fullcont_wavelets = wavelet_generator(y, bands=bands, sampling_period=sampling_period, wtype=wtype)
            new_segment_range = first_segment(x[span[0]:span[1]], y[span[0]:span[1]], cont_wavelets, y, fullcont_wavelets, span[0], peak_rate_threshold=peak_rate_threshold, show_warnings=show_warnings)
            if None in new_segment_range:
                ignored_ranges.append(span)
            else:
                true_new_segment_range = [new_segment_range[0] + span[0], new_segment_range[1] + span[0], new_segment_range[2] + span[0]]
                all_segments.append(true_new_segment_range)

    # sort segments
    all_segments = sorted(all_segments, key=lambda a: a[1])
    # if no segments take a segment across entire range
    if len(all_segments) == 0:
        all_segments = [[0,np.argmax(y),len(x)-1]]
    else:
        # fill in gaps between segments
        if join == True:
            new_segments = []
            if all_segments[0][0] != 0:
                inter_minimum = np.argmin(y[:all_segments[0][0]])
            else:
                inter_minimum = all_segments[0][0]
            for i in np.arange(len(all_segments)-1):
                previousMinimum = inter_minimum
                if all_segments[i][2] < all_segments[i+1][0]:
                    inter_minimum = np.argmin(y[all_segments[i][2]:all_segments[i+1][0]]) + all_segments[i][2]
                else:
                    inter_minimum = all_segments[i][2]
                new_segments.append([previousMinimum, all_segments[i][1], inter_minimum])
            if all_segments[-1][2] != len(x)-1:
                new_segments.append([inter_minimum, all_segments[-1][1], np.argmin(y[all_segments[-1][2]:])+all_segments[-1][2]])
            else:
                new_segments.append([inter_minimum, all_segments[-1][1], all_segments[-1][2]])
            all_segments = new_segments

    if plot == True:
        wavelet_plot(y, wavelet_generator(y), skip_wavelets=skip_wavelets)
        if segment_mode == 'flat':
            for seg in all_segments:
                segment_plot(seg, y, wavelet_generator(y))
        elif segment_mode == 'poly':
            regression_plot(all_segments, all_segments_polynomial_regression(y, all_segments, degree=degree), y, wavelet_generator(y), skip_wavelets=skip_wavelets)
        elif segment_mode == 'all':
            for seg in all_segments:
                segment_plot(seg, y, wavelet_generator(y))
            regression_plot(all_segments, all_segments_polynomial_regression(y, all_segments, degree=degree), y, wavelet_generator(y))
        else:
            print('Warning: segment_mode not recognised, no segments plotted')
        if title=='long':
            plt.title('Full Segment Analysis of '+ name + ' fault \nConstants: wtype=' + str(wtype) + ' sampling_period=' + str(sampling_period) + ' bands=' + str(bands[1]-bands[0]) + ' peak_rate_threshold=' + str(peak_rate_threshold))
        elif title=='short':
            plt.title('Full Segment Analysis of '+ name + ' fault')
    if segment_mode == 'poly' or segment_mode == 'all':
        return all_segments, all_segments_polynomial_regression(y, all_segments, degree=degree)
    else:
        return all_segments

def segment_polynomial_regression(y, start, peak, end, degree=3):
    """ Obtain a cubic function that fits a segment of a signal"""
    segment_throw_y = y[start:end+1]

    segment_throw_x = np.arange(0, len(segment_throw_y), 1)

    weights_start_x = np.full(100,segment_throw_x[0])
    weights_start_y = np.full(100,segment_throw_y[0])
    weights_end_x = np.full(100,segment_throw_x[-1])
    weights_end_y = np.full(100,segment_throw_y[-1])
    weights_peak_x = np.full(10,segment_throw_x[peak-start])
    weights_peak_y = np.full(10,segment_throw_y[peak-start])

    
    segment_throw_x_weighted = np.concatenate((weights_start_x, segment_throw_x, weights_end_x))
    segment_throw_y_weighted = np.concatenate((weights_start_y, segment_throw_y, weights_end_y))
    segment_throw_x_weighted = np.insert(segment_throw_x_weighted, peak-start+100, weights_peak_x)
    segment_throw_y_weighted = np.insert(segment_throw_y_weighted, peak-start+100, weights_peak_y)

    poly = PolynomialFeatures(degree, include_bias=False)

    poly_features = poly.fit_transform(np.asarray(segment_throw_x_weighted).reshape(-1,1))

    poly_reg_model = LinearRegression()
    return poly_reg_model.fit(poly_features, segment_throw_y_weighted), segment_throw_x_weighted, poly_features

def all_segments_polynomial_regression(y, all_segments, degree=3):
    """ Repeat polynomial regression for all input segments"""
    return [segment_polynomial_regression(y, segment[0], segment[1], segment[2], degree=degree) for segment in all_segments]

def regression_plot(all_segments, segment_regression, y, cont_wavelets, skip_wavelets=False):
    """ Plot results of regression"""
    for idx, segment in enumerate(all_segments):
        y_predicted = segment_regression[idx][0].predict(segment_regression[idx][2])
        if skip_wavelets is False:
            plt.plot(segment_regression[idx][1] + segment[0], y_predicted * 0.8 * cont_wavelets[0].shape[0] / np.max(y), color='red')
        else:
            plt.plot(segment_regression[idx][1] + segment[0], y_predicted, color='red')
    return None

def reduce_data_resample(x, factor):
    """ Resample a signal to a lower sample rate"""
    return resample(x, int(len(x)/factor))

def reduce_data_decimate(x, factor):
    """ Decimate a signal to a lower sample rate"""
    return x[0::int(factor)]

def closest_idx(array, val):
    return np.abs(np.asarray(array)-val).argmin()

def find_minimum_sample_rate(x, y, 
                          sampling_mode='interpolate', 
                          bands=(1,31), 
                          samples_per_period=100,
                          sampling_period=0.1, 
                          wtype='mexh', 
                          name='Unknown', 
                          peak_rate_threshold=0.03,
                          plot=False, 
                          verbose=False, 
                          resampling_kind='linear',
                          get_error=False):
    all_segments_full, segment_regression_full = full_segment_analysis(x,y, 
                                          bands=bands, 
                                          samples_per_period=samples_per_period,
                                          sampling_period=sampling_period, 
                                          wtype=wtype,
                                          name=name, 
                                          peak_rate_threshold=peak_rate_threshold, 
                                          join=True, 
                                          segment_mode='poly')
    x_new = x.copy()
    y_new = y.copy()

    sample_factor = 1
    if get_error is True:
        integral_error = []
        modulus_error = []
        modulus_error.append(modulus_error(x, y, x_new, y_new))
        integral_error.append(all_segments_integral(all_segments_full, all_segments_full, segment_regression_full, segment_regression_full))
    sample_rates = []
    # sample_rates.append(int(len(x) * 0.9**(sample_factor-1)))
    while True:
        
        
        # Gradual increase in sample reduction
        sample_factor = sample_factor+1

        if get_error is True:
            x_temp, y_temp = resample_to_peak(x, y, 
                                        wtype=wtype, 
                                        bands=bands, 
                                        samples_per_period=samples_per_period,
                                        sampling_period=sampling_period, 
                                        sample_rate_override=int(len(x) * 0.9**(sample_factor-2)), 
                                        kind=resampling_kind)
            modulus_error.append(modulus_error(x, y, x_temp, y_temp))
        sample_rates.append(int(len(x) * 0.9**(sample_factor-2)))

        # print('Number of Samples: ', len(x_new))
        all_segments, segments_regression = full_segment_analysis(x_new,y_new, 
                                          bands=bands, 
                                          samples_per_period=samples_per_period,
                                          sampling_period=sampling_period, 
                                          wtype=wtype,
                                          name=name, 
                                          peak_rate_threshold=peak_rate_threshold, 
                                          join=True, 
                                          segment_mode='poly', 
                                          plot=False, 
                                          resampling_kind=resampling_kind)
        # integral error
        if get_error is True:
            integral_error.append(all_segments_integral(all_segments_full, all_segments, segment_regression_full, segments_regression))

        if verbose is True:
            print('Number of Segments: ', len(all_segments))
        if len(all_segments) < len(all_segments_full) or len(x_new) < 4:
            # reproduce previous sample rate
            if sampling_mode == 'decimate':
                x_previous = reduce_data_decimate(x, sample_factor-2)
                y_previous = reduce_data_decimate(y, sample_factor-2)
            elif sampling_mode == 'resample':
                x_previous = reduce_data_resample(x, sample_factor-2)
                y_previous = reduce_data_resample(y, sample_factor-2)
            elif sampling_mode == 'interpolate':
                #interpolate mode
                x_previous, y_previous = resample_to_peak(x, y, 
                                                      wtype=wtype, 
                                                      bands=bands, 
                                                      samples_per_period=samples_per_period,
                                                      sampling_period=sampling_period, 
                                                      sample_rate_override=int(len(x) * 0.9**(sample_factor-3)), 
                                                      kind=resampling_kind)
            else:
                print('Warning: sampling_mode not recognised')

            if verbose is True:
                print('Starting Samples: ', len(x),'Minimum Samples: ', len(x_previous))
            # Find smallest segment length
            old_segments = full_segment_analysis(x_previous,y_previous, 
                                              bands=bands, 
                                              samples_per_period=samples_per_period,
                                              sampling_period=sampling_period, 
                                              wtype=wtype,
                                              name=name, 
                                              peak_rate_threshold=peak_rate_threshold, 
                                              join=True, 
                                              segment_mode='poly', 
                                              plot=plot, 
                                              resampling_kind=resampling_kind,
                                              skip_wavelets=True)[0]
            x_previous_rescaled = np.linspace(0,250,len(x_previous))
            all_sizes = [closest_idx(x_previous_rescaled, i[2])-closest_idx(x_previous_rescaled, i[0])+2 for i in old_segments]
            percentage = 100/np.max(all_sizes)
            percentage_small = 100/np.min(all_sizes)
            percentage_total = 100/np.sum(all_sizes)

            if verbose is True:
                print('Sample as Percentage of Largest Fault Length: ', percentage, '%')
                print('Sample as percentage of Lost Fault Length: ', percentage_small, '%')
                print('Sample as percentage of Total: ', percentage_total, '%')

            if plot == True:
                # One plot at minimum sample rate and another at the step below
                plt.stem(np.linspace(0,old_segments[-1][-1],len(y_previous)),y_previous)
                plt.title("Minimum Sample Rate")
                plt.show()

                all_segments = full_segment_analysis(x_new,y_new, 
                                                  bands=bands, 
                                                  samples_per_period=samples_per_period,
                                                  sampling_period=sampling_period, 
                                                  wtype=wtype,
                                                  name=name, 
                                                  peak_rate_threshold=peak_rate_threshold, 
                                                  join=True, 
                                                  segment_mode='poly', 
                                                  plot=True, 
                                                  resampling_kind=resampling_kind,
                                                  skip_wavelets=True)
                plt.stem(np.linspace(0,all_segments[0][-1][-1],len(y_new)),y_new)
                plt.title("Below Minimum Sample Rate")
                plt.show()
            break
        # go to next sample rate
        if sampling_mode == 'decimate':
            x_new = reduce_data_decimate(x, sample_factor)
            y_new = reduce_data_decimate(y, sample_factor)
        elif sampling_mode == 'resample':
            x_new = reduce_data_resample(x, sample_factor)
            y_new = reduce_data_resample(y, sample_factor)
        elif sampling_mode == 'interpolate':
            #interpolate mode
            x_new, y_new = resample_to_peak(x, y, 
                                        wtype=wtype, 
                                        bands=bands, 
                                        samples_per_period=samples_per_period,
                                        sampling_period=sampling_period, 
                                        sample_rate_override=int(len(x) * 0.9**(sample_factor-1)), 
                                        kind=resampling_kind)
        else:
            print('Error: sampling_mode not recognised')
    
    
    if get_error is True:
        intersect_error = [i[2] if i is not None else 0 for i in integral_error]
        integral_error = [i[1] if i is not None else 0 for i in integral_error]
        modulus_error = [i[1] if i is not None else 0 for i in modulus_error]
        return (percentage, percentage_small, percentage_total, sample_rates, integral_error, modulus_error, intersect_error)
    else:
        return (percentage, percentage_small, percentage_total, sample_rates)

def integral_difference(poly1, poly2, bounds1, bounds2):
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


    
    area_difference = quad(lambda x: cubic_abs_difference(x, coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3], coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3], offset1=offset1, offset2=offset2), 0, bounds[1]-bounds[0])
    difference_1 = quad(lambda x: cubic_abs(x, coeffs_1[0], coeffs_1[1], coeffs_1[2], coeffs_1[3], offset=offset1), 0, bounds[1]-bounds[0])
    difference_2 = quad(lambda x: cubic_abs(x, coeffs_2[0], coeffs_2[1], coeffs_2[2], coeffs_2[3], offset=offset2), 0, bounds[1]-bounds[0])
    
    area_difference_ratio = area_difference[0]/np.max([difference_1[0], difference_2[0]])

    return (area_difference_ratio, area_difference[0], difference_1[0], difference_2[0])

def cubic_abs(x, cube, square, linear, constant, offset=0):
    """ Returns the absolute value of a cubic polynomial"""
    return np.abs(cube*(x+offset)**3 + square*(x+offset)**2 + linear*(x+offset) + constant)

def cubic_abs_difference(x, cube1, square1, linear1, constant1, cube2, square2, linear2, constant2, offset1=0, offset2=0):
    """ Returns the absolute value of the difference between two cubic polynomials with offset values"""
    return np.abs(cube1*(x+offset1)**3 + square1*(x+offset1)**2 + linear1*(x+offset1) + constant1 - (cube2*(x+offset2)**3 + square2*(x+offset2)**2 + linear2*(x+offset2) + constant2))

def all_segments_integral(all_segments, all_segments2, segment_regression, segment_regression2, include_unmatched=True):
    all_segments_match, all_segments2_match, segment_regression_match, segment_regression2_match, intersect_error = fault_matchup(all_segments, all_segments2, segment_regression, segment_regression2)

    # Error for unmatched faults
    if include_unmatched is True:
        unmatched = []
        if len(segment_regression_match) != len(segment_regression):
            for idx, curve in enumerate(segment_regression):
                if curve not in segment_regression_match:
                    coeffs_1 = np.asarray(list(curve[0].coef_)[::-1] + [curve[0].intercept_])
                    unmatched.append(quad(lambda x, coeffs = coeffs_1, i = idx: cubic_abs(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], offset=all_segments[i][0]), 0, all_segments[idx][2]-all_segments[idx][0])[0])
        if len(segment_regression2_match) != len(segment_regression2):
            for idx, curve in enumerate(segment_regression2):
                if curve not in segment_regression2_match:
                    coeffs_1 = np.asarray(list(curve[0].coef_)[::-1] + [curve[0].intercept_])
                    unmatched.append(quad(lambda x, coeffs = coeffs_1, i = idx: cubic_abs(x, coeffs[0], coeffs[1], coeffs[2], coeffs[3], offset=all_segments2[i][0]), 0, all_segments2[idx][2]-all_segments2[idx][0])[0])


    # integral differences
    all_differences = [integral_difference(segment_regression_match[i][0], segment_regression2_match[i][0], (all_segments_match[i][0], all_segments_match[i][2]), (all_segments2_match[i][0], all_segments2_match[i][2])) for i in range(len(all_segments_match))]

    # split up list
    all_errors = [i[1] for i in all_differences] + unmatched
    all_areas = [i[2] for i in all_differences] + unmatched
    

    return (all_differences, np.sum(all_errors)/np.sum(all_areas), intersect_error) 

def cubic_plot(all_segments, polynomial_regression, point_count=100, color='red', label='Segments'):
    for i in range(len(all_segments)):
        x_plot = np.linspace(0, all_segments[i][2]-all_segments[i][0], point_count)
        f = lambda x: polynomial_regression[i][0].coef_[2]*x**3 + polynomial_regression[i][0].coef_[1]*x**2 + polynomial_regression[i][0].coef_[0]*x + polynomial_regression[i][0].intercept_
        y_plot = f(x_plot)
        x_plot = x_plot + all_segments[i][0]
        fig, ax = plt.subplots(1, 1)
        if i == 0:
            ax.plot(x_plot, y_plot, color=color, label=label)
        else:
            ax.plot(x_plot, y_plot, color=color)
    return fig, ax

def modulus_error(x, y, x2, y2):
    x = np.asarray(x)
    y = np.asarray(y)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)

    x = x[x <= np.max(x2)]


    y = y[0:len(x)]
    f = interp1d(x2, y2, kind='linear')
    y2 = f(x)
    x2 = x


    x_diff = x
    y_diff = np.abs(y-y2)
    modulus_error = np.trapz(y_diff, x_diff)
    return (modulus_error, modulus_error/np.trapz(y, x))

def fault_matchup(all_segments, all_segments2, segment_regression, segment_regression2):
    if len(all_segments) == len(all_segments2):
        intersect_error = np.mean([np.abs(all_segments[i][0] - all_segments2[i][0])/np.max([all_segments[i][2]-all_segments[i][0],all_segments2[i][2]-all_segments[i][0]]) for i in range(len(all_segments))])
    else:
        new_length = np.min([len(all_segments), len(all_segments2)])

        start_points = [i[0] for i in all_segments]
        start_points2 = [i[0] for i in all_segments2]

        new_segments = {key: None for key in range(new_length)}
        new_segments2 = {key: None for key in range(new_length)}
        
        new_segment_regression = {key: None for key in range(new_length)}
        new_segment_regression2 = {key: None for key in range(new_length)}

        intersect_error = np.zeros(new_length)

        if len(all_segments) > len(all_segments2):
            start_points_temp = start_points.copy()
            for idx, point in enumerate(start_points2):
                new_point = closest_idx(start_points, point)
                new_segments[idx] = all_segments[new_point]
                new_segments2[idx] = all_segments2[idx]
                new_segment_regression[idx] = segment_regression[new_point]
                new_segment_regression2[idx] = segment_regression2[idx]
                start_points_temp[new_point] = np.nan
                intersect_error[idx] = np.abs(new_segments[idx][0] - new_segments2[idx][0])/np.max([new_segments[idx][2]-new_segments[idx][0], new_segments2[idx][2]-new_segments2[idx][0]])

            for point in start_points_temp:
                if np.isnan(point) == False:
                    intersect_error = np.concatenate((intersect_error, np.ones(1)))
            

        if len(all_segments) < len(all_segments2):
            start_points2_temp = start_points2.copy()
            for idx, point in enumerate(start_points):
                new_point = closest_idx(start_points2, point)
                new_segments2[idx] = all_segments2[new_point]
                new_segments[idx] = all_segments[idx]
                new_segment_regression2[idx] = segment_regression2[new_point]
                new_segment_regression[idx] = segment_regression[idx]
                start_points2_temp[new_point] = np.nan
                intersect_error[idx] = np.abs(new_segments[idx][0] - new_segments2[idx][0])/np.max([new_segments[idx][2]-new_segments[idx][0], new_segments2[idx][2]-new_segments2[idx][0]])

            for point in start_points2_temp:
                if np.isnan(point) == False:
                    intersect_error = np.concatenate((intersect_error, np.ones(1)))
        intersect_error = np.mean(intersect_error)
        


        all_segments = list(new_segments.values())
        all_segments2 = list(new_segments2.values())
        segment_regression = list(new_segment_regression.values())
        segment_regression2 = list(new_segment_regression2.values())
            

    return all_segments, all_segments2, segment_regression, segment_regression2, intersect_error