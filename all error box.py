import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import plotly.graph_objects as go

import algorithm as alg

df = pd.read_csv('results with error progression kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv', 
                 converters={'Integral Error':ast.literal_eval,'Modulus Error':ast.literal_eval, 'Intersect Error':ast.literal_eval, 'Sample Counts':ast.literal_eval})

final_integral_error = df['Integral Error'].apply(lambda x: x[-1]*100)

final_modulus_error = df['Modulus Error'].apply(lambda x: x[-1]*100)

final_intersect_error = df['Intersect Error'].apply(lambda x: x[-1]*100)

sample_rate_change = df['Sample Counts'].apply(lambda x: 100*(1-(x[-1]/x[0])))

bad_index = final_integral_error[(final_integral_error > 100) | (final_integral_error <= 0)] + final_modulus_error[(final_modulus_error > 100) | (final_integral_error <= 0)] + sample_rate_change[sample_rate_change == 0]
final_integral_error = final_integral_error.drop(bad_index.index)
final_modulus_error = final_modulus_error.drop(bad_index.index)
final_intersect_error = final_intersect_error.drop(bad_index.index)
sample_rate_change = sample_rate_change.drop(bad_index.index)




print('integral', np.mean(final_integral_error))
print('modulus', np.mean(final_modulus_error))
print('intersect', np.mean(final_intersect_error))
print('sample rate', np.mean(sample_rate_change))

fig = go.Figure()

fig.add_box(y=final_integral_error, name='Integral Error')
fig.add_box(y=final_modulus_error, name='Modulus Error')
fig.add_box(y=final_intersect_error, name='Intersection Error')
fig.add_box(y=sample_rate_change, name='Sample Rate Reduction (%)')
fig.update_layout(yaxis_title='Error (%)', xaxis_title='Error Type', template='plotly_white', width=2000, height=1250, font_size=32, showlegend=False,)
fig.show()