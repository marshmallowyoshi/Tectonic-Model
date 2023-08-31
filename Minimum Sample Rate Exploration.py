import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt

name = 'R2 H2'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

output = alg.findMinimumSampleRate(x,y, 
                        plot=True, 
                        verbose=True, 
                        peakRateThreshold=0.04, 
                        samples_per_period=1300, 
                        sampling_period=1, 
                        bands=(1,31),
                        get_error=True,
                        resampling_kind='quadratic',)

integral_errors = output[4]
modulus_errors = output[5]

intersect_errors = output[6]
sample_rates = output[3]

fig, ax = plt.subplots()
ax.plot(sample_rates, integral_errors, label='Integral Error', color='red')
ax.plot(sample_rates,modulus_errors, label='Modulus Error', color='blue')
ax.plot(sample_rates,intersect_errors, label='Intersect Error', color='green')
ax.legend()
ax.invert_xaxis()
ax.set_xlabel('Total Samples')
ax.set_ylabel('Error (%)')
ax.set_title(name)
plt.show()