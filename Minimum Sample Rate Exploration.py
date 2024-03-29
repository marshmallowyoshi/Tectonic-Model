import numpy as np
import pandas as pd
import algorithm as alg
import matplotlib.pyplot as plt

name = 'part4'
name = 'data/' + name + '.csv'
df = pd.read_csv(name)
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])

############################################################################################################################################

output = alg.find_minimum_sample_rate(x,y, 
                        plot=False, 
                        verbose=True, 
                        peak_rate_threshold=0.04, 
                        samples_per_period=1300, 
                        sampling_period=1, 
                        bands=(1,31),
                        get_error=True,
                        resampling_kind='quadratic',)

integral_errors = output[4][1:]
modulus_errors = output[5][1:]

intersect_errors = output[6][1:]
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