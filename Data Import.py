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
                alg.fullSegmentAnalysis(value2[0], value2[1], name=key + ' ' + key2, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=True, title='short', samples_per_period=1000, bands=(1,31), sampling_period=1)
                plt.savefig('figures\\analysis\\' + key + ' ' + key2 + '.png', dpi='figure', bbox_inches='tight', format=None)
                plt.clf()
        else:
            alg.fullSegmentAnalysis(value[0], value[1], name=key, peakRateThreshold=0.03, join=True, segmentMode='poly', plot=True, title='short', samples_per_period=1000, bands=(1,31), sampling_period=1)
            plt.savefig('figures\\analysis\\' + key + '.png', dpi='figure', bbox_inches='tight', format=None)
            plt.clf()

def batchMinimum(rawData):
    errorData = []
    for key, value in rawData.items():
        if isinstance(value, dict):
            for key2, value2 in value.items():
                error = alg.findMinimumSampleRate(value2[0], value2[1], name=key + ' ' + key2)
                errorData.append((key + ' ' + key2, error))

        else:
            error = alg.findMinimumSampleRate(value[0], value[1], name=key)
            errorData.append((key, error))
    return errorData

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

with open('rawData.txt', 'w') as f:
    f.write(str(batchMinimum(rawData)))


# x, y = standardForm(rawData['Roller 1 analysis.csv'])
# alg.fullSegmentAnalysis(x,y, plot=True, join=True, segmentMode='poly', peakRateThreshold=0.03)
# plt.show()