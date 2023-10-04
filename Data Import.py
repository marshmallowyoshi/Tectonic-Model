import pandas as pd
import os
import algorithm as alg
import matplotlib.pyplot as plt

def enumerateFiles(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.csv' in file:
                files.append(file)
    return files

def preprocess(df):
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def standardForm(df):
    if df.shape[1] != 2:
        raise Exception("Dataframe must have exactly 2 columns")
    return (df.iloc[:,0].tolist(), df.iloc[:,1].tolist())

def batchPlot(rawData):
    for key, value in rawData.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                plt.plot(value2[0], value2[1])
                plt.title(key + ' ' + key2)
                plt.savefig('figures\\data\\' + key + ' ' + key2 + '.png', dpi='figure', bbox_inches='tight', format=None)
                plt.clf()
        else:
            plt.plot(value[0], value[1])
            plt.title(key)
            plt.savefig('figures\\data\\' + key + '.png', dpi='figure', bbox_inches='tight', format=None)
            plt.clf()

def batchAnalysis(rawData):
    for key, value in rawData.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                alg.full_segment_analysis(value2[0], value2[1], name=key + ' ' + key2, peak_rate_threshold=0.03, join=True, segment_mode='poly', plot=True, title='short', samples_per_period=1000, bands=(1,31), sampling_period=1)
                plt.savefig('figures\\analysis\\' + key + ' ' + key2 + '.png', dpi='figure', bbox_inches='tight', format=None)
                plt.clf()
        else:
            alg.full_segment_analysis(value[0], value[1], name=key, peak_rate_threshold=0.03, join=True, segment_mode='poly', plot=True, title='short', samples_per_period=1000, bands=(1,31), sampling_period=1)
            plt.savefig('figures\\analysis\\' + key + '.png', dpi='figure', bbox_inches='tight', format=None)
            plt.clf()

def batchMinimum(rawData,
                 peak_rate_threshold=0.04,
                 resampling_kind='quadratic',
                 samples_per_period=1300,
                 sampling_period=1):
    peak_rate_threshold = 0.04
    errorData = []
    for key, value in rawData.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                error = alg.find_minimum_sample_rate(value2[0], value2[1],
                                                  name=key + ' ' + key2,
                                                  plot=False,
                                                  verbose=False,
                                                  resampling_kind=resampling_kind,
                                                  peak_rate_threshold=peak_rate_threshold,
                                                  samples_per_period=samples_per_period,
                                                  sampling_period=sampling_period,
                                                  get_error=True)
                errorData.append((key + ' ' + key2, error))

        else:
            error = alg.find_minimum_sample_rate(value[0], value[1],
                                              name=key,
                                              plot=False,
                                              verbose=False,
                                              resampling_kind=resampling_kind,
                                              peak_rate_threshold=peak_rate_threshold,
                                              samples_per_period=samples_per_period,
                                              sampling_period=sampling_period,
                                              get_error=True)
            errorData.append((key, error))
    return errorData

def exportBatchMinimum(peak_rate_threshold=0.04,
                       resampling_kind='quadratic',
                       samples_per_period=1300,
                       sampling_period=1,):
    batchResults = batchMinimum(rawData,
                            peak_rate_threshold=peak_rate_threshold,
                            resampling_kind=resampling_kind,
                            samples_per_period=samples_per_period,
                            sampling_period=sampling_period)
    df = pd.DataFrame(batchResults, columns=['Dataset', 'Error'])

    df[['Per Large Fault Error', 
        'Per Lost Fault Error', 
        'Total Length Error',
        'Sample Counts',
        'Integral Error',
        'Modulus Error',
        'Intersect Error']] = pd.DataFrame(df['Error'].tolist(), index=df.index)
    
    df = df.drop(columns=['Error'])

    # df['Integral Error'] = df['Integral Error'].apply(lambda x: [i[1] for i in x])
    # df['Modulus Error'] = df['Modulus Error'].apply(lambda x: [i[1] for i in x])

    df.to_csv(str('results with error progression ' +
                'kind=' + resampling_kind + 
                ' peakratethreshold=' + str(peak_rate_threshold) + 
                ' samplesperperiod=' + str(samples_per_period) + 
                ' samplingperiod=' + str(sampling_period) + 
                '.csv'))
    return df

dataNames = enumerateFiles('data')

rawData = {name: preprocess(pd.read_csv('data\\'+name)) for name in dataNames}

rawData['BF1 H2.csv'] = rawData['BF1 H2.csv'].filter(['Inline No.','Throw'])
rawData['BF2 H2.csv'] = rawData['BF2 H2.csv'].filter(['Inline No.','Throw'])
rawData['BF3 H2.csv'] = rawData['BF3 H2.csv'].filter(['Inline No.','Throw'])
rawData['BF4 H2.csv'] = rawData['BF4 H2.csv'].filter(['Inline No.','Throw'])

rawData['F9.csv'] = {str(int(profile)): df.filter(['Depth/ms','Throw/ms']) for profile, df in rawData['F9.csv'].groupby('Profile')}

name = 'F22C1'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C2'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C3'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C4'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C5'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C6'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C7'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C8'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C9'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])
name = 'F22C10'
rawData[name + '.csv'] = rawData[name + '.csv'].filter(['Depth/ms','Throw/ms'])

rawData['Keystone all.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['Keystone all.csv'].groupby('Fault No.')}

rawData['KeystoneF6.csv'] = {str(int(profile)): df.filter(['Depth/ms','Throw/ms']) for profile, df in rawData['KeystoneF6.csv'].groupby('Profile')}

rawData['L2 H4-1.csv'] = rawData['L2 H4-1.csv'].filter(['Inline No.','Throw'])

rawData['Linked Key&Poly Central.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['Linked Key&Poly Central.csv'].groupby('Fault No.')}
rawData['Linked Key&Poly North.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['Linked Key&Poly North.csv'].groupby('Fault No.')}
rawData['Linked Key&Poly South.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['Linked Key&Poly South.csv'].groupby('Fault No.')}

rawData['part4.csv'] = rawData['part4.csv'].filter(['Xline','Throw'])

rawData['poly central.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['poly central.csv'].groupby('Fault No.')}
rawData['poly north.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['poly north.csv'].groupby('Fault No.')}
rawData['poly south.csv'] = {str(int(profile)): df.filter(['Z (ms) FW','Throw (m)']) for profile, df in rawData['poly south.csv'].groupby('Fault No.')}

rawData['R2 H2.csv'] = rawData['R2 H2.csv'].filter(['Xline No.','Throw'])
rawData['R2 H3.csv'] = rawData['R2 H3.csv'].filter(['Xline No.','Throw'])
rawData['R3 H3.csv'] = rawData['R3 H3.csv'].filter(['Xline No.','Throw'])
rawData['R4 H3.csv'] = rawData['R4 H3.csv'].filter(['Xline No.','Throw'])
rawData['R5 H2.csv'] = rawData['R5 H2.csv'].filter(['Xline No.','Throw'])

rawData['Roller 1 analysis.csv'] = rawData['Roller 1 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 2 analysis.csv'] = rawData['Roller 2 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 3 analysis.csv'] = rawData['Roller 3 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 4 analysis.csv'] = rawData['Roller 4 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 5 analysis.csv'] = rawData['Roller 5 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 6 analysis.csv'] = rawData['Roller 6 analysis.csv'].filter(['Distance (km)','Throw (m)'])
rawData['Roller 7 analysis.csv'] = rawData['Roller 7 analysis.csv'].filter(['Distance (km)','Throw (m)'])

rawData['S1 H2.csv'] = rawData['S1 H2.csv'].filter(['Inline No.','Throw'])
rawData['S2 H2.csv'] = rawData['S2 H2.csv'].filter(['Inline No.','Throw'])
rawData['S3 H2.csv'] = rawData['S3 H2.csv'].filter(['Inline No.','Throw'])
rawData['S4 H2.csv'] = rawData['S4 H2.csv'].filter(['Xline No.','Throw'])
rawData['S5 H2.csv'] = rawData['S5 H2.csv'].filter(['Xline No.','Throw'])

for key, value in rawData.items():
    if isinstance(value, dict):
        for key2, value2 in value.items():
            value[key2] = standardForm(value2)
    else:
        rawData[key] = standardForm(value)

key = 'BF1 H2.csv'
newdf = pd.DataFrame({'dataset': key, 'x': rawData[key][0], 'y': rawData[key][1]})
print(newdf.head())


final_dataframe = pd.DataFrame({'dataset': [], 'x': [], 'y': []})
for key, value in rawData.items():
    if isinstance(value, dict):
        for key2, value2 in value.items():
            final_dataframe = pd.concat([final_dataframe, pd.DataFrame({'dataset': key + ' ' + key2, 'x': value2[0], 'y': value2[1]})])
    else:
        final_dataframe = pd.concat([final_dataframe, pd.DataFrame({'dataset': key, 'x': value[0], 'y': value[1]})])

final_dataframe.to_csv('final_dataframe.csv')

peak_rate_threshold=0.04
resampling_kind='quadratic'
samples_per_period=1300
sampling_period=1

exportBatchMinimum(peak_rate_threshold=peak_rate_threshold,
                       resampling_kind=resampling_kind,
                       samples_per_period=samples_per_period,
                       sampling_period=sampling_period,)