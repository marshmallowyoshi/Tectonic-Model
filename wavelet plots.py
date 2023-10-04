import algorithm as alg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go

df = pd.read_csv('R2 H3.csv')
x = np.asarray(df.iloc[:,0])
y = np.asarray(df.iloc[:,3])
y = signal.savgol_filter(y, 11, 3)

contWav = alg.wavelet_generator(x,y)


fig = go.Figure()

for step in np.arange(contWav[0].shape[0]):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color = "#00CED1", width = 2),
            name = "wavelength = " + str(contWav[1][::-1][step]),
            x=np.arange(contWav[0].shape[1]),
            y=np.log(contWav[0][step]-np.min(contWav[0]))*20
            )
    )


fig.data[10].visible = True

steps = []

for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to wavelength: " + str(contWav[1][::-1][i])}],
    label=str(np.round(contWav[1][::-1][i],3))+"m" 
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Wavelength: "},
    pad={"t": 50},
    steps=steps

)]

fig.update_layout(
    sliders=sliders
)

fig.show()