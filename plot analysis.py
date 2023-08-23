import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
df = pd.read_csv('results kind=quadratic peakratethreshold=0.04 samplesperperiod=1300 samplingperiod=1.csv')
rawdf = pd.read_csv('final_dataframe.csv')

# plt.scatter(np.arange(len(df['Total Length Error'])),df['Total Length Error'])
# plt.xticks(np.arange(len(df['Total Length Error'])), df['Dataset'], rotation=90)
# plt.show()

# print("Total Length Error: mean:", df['Total Length Error'].mean(), "sd:", df['Total Length Error'].std(), "q1", df['Total Length Error'].quantile(0.25), "q3", df['Total Length Error'].quantile(0.75))
# print("Per Fault Error: ", df['Per Fault Error'].mean(), "sd:", df['Per Fault Error'].std(), "q1", df['Per Fault Error'].quantile(0.25), "q3", df['Per Fault Error'].quantile(0.75))
df.rename(columns={'Total Length Error': 'Total Length', 'Per Large Fault Error': 'Large Fault', 'Per Lost Fault Error': 'Lost Fault'}, inplace=True)

df_pfe = df.filter(['Dataset', 'Large Fault'], axis=1)
df_lfe = df.filter(['Dataset', 'Lost Fault'], axis=1)
df_tle = df.filter(['Dataset', 'Total Length'], axis=1)

dfc = pd.concat([df_pfe, df_lfe, df_tle])
dfm = pd.melt(dfc, id_vars=['Dataset'], value_vars=['Total Length', 'Large Fault', 'Lost Fault'])

fig = go.Figure()

categories = dfm['variable'].unique().tolist()

for category in categories:
    fig.add_trace(go.Violin(x = dfm['variable'][dfm['variable'] == category],
                            y = dfm['value'][dfm['variable'] == category],
                            name = category,
                            box_visible=True,
                            meanline_visible=True,
                            spanmode='hard',
                            ))
fig.update_layout(template='plotly_white', 
                  width=2000, height=1250, 
                  yaxis_title='Minimum Sample Rate (%)', xaxis_title='Method of Minimum Sample Rate Calculation',
                  font_size=32,
                  showlegend=False,)
fig.show()
# fig.add_trace(go.Violin(dfm, x='value', y='variable', 
#                 box=True, points='all', 
#                 hover_data=dfm.columns, 
#                 color='variable', 
#                 title='Violin Plot of Error Distribution', 
#                 template='plotly_white', 
#                 width=2000, height=1250,
#                 spanmode='hard',))
# fig.update_xaxes(title_text='Error (%)')
# fig.show()