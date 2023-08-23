import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import algorithm as alg

def f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

name = 'BF1 H2'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

initial_peakRateThreshold = 0.03

initial_samples_per_period = 1000

contWav = alg.waveletGen(x,y)
allSegments, polynomialRegression = alg.fullSegmentAnalysis(x,y, name=name, peakRateThreshold=initial_peakRateThreshold, join=True, segmentMode='poly', plot=False, title='short', samples_per_period=initial_samples_per_period, bands=(1,31), sampling_period=1)

for idx, val in enumerate(polynomialRegression):
    coeffs = np.asarray(list(val[0].coef_)[::-1] + [val[0].intercept_])
    
    if idx == 0:
        curves = f(np.linspace(0, allSegments[idx][2]-allSegments[idx][0], 1000), *coeffs)
    else:
        curves = np.concatenate((curves, f(np.linspace(0, allSegments[idx][2]-allSegments[idx][0], 1000), *coeffs)))

x_segment = [i for sublist in [[i[0],i[1],i[2]] for i in allSegments] for i in sublist]
y_segment, x_segment = alg.resampleToPeak(np.ones(len(x_segment)), x_segment, sample_rate_override=len(x))

fig, ax = plt.subplots()
line, = ax.plot(np.arange(contWav[0].shape[1]), y, color='black')
line2, = ax.plot(np.linspace(0, contWav[0].shape[1], curves.shape[0]), curves, color='red')

line3, = ax.plot(x_segment, y_segment, color='blue')

ax.set_xlabel('Distance (m)')

fig.subplots_adjust(left=0.25, bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(axfreq, 'Peak Rate Threshold', 0.01, 0.1, valinit=initial_peakRateThreshold)

axamp = fig.add_axes([0.1, 0.25, 0.0225, 0.65])
amp_slider = Slider(axamp, 'Samples per Period', 100, 5000, valinit=initial_samples_per_period, orientation='vertical')


def update(val):
    peakRateThreshold = freq_slider.val
    samples_per_period = amp_slider.val
    allSegments, polynomialRegression = alg.fullSegmentAnalysis(x,y, name=name, peakRateThreshold=peakRateThreshold, join=True, segmentMode='poly', plot=False, title='short', samples_per_period=samples_per_period, bands=(1,31), sampling_period=1)
    for idx, val in enumerate(polynomialRegression):
        coeffs = np.asarray(list(val[0].coef_)[::-1] + [val[0].intercept_])
        
        if idx == 0:
            curves = f(np.linspace(0, allSegments[idx][2]-allSegments[idx][0], 1000), *coeffs)
        else:
            curves = np.concatenate((curves, f(np.linspace(0, allSegments[idx][2]-allSegments[idx][0], 1000), *coeffs)))

    line2.set_xdata(np.linspace(0, contWav[0].shape[1], curves.shape[0]))
    line2.set_ydata(curves)
    fig.canvas.draw_idle()

freq_slider.on_changed(update)
amp_slider.on_changed(update)

resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()