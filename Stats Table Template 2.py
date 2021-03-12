import pandas as pd
from sigfig import round
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as scipy
import datetime
import glob
import streamlit as st
from matplotlib.font_manager import FontProperties
# pd.options.display.float_format = "{:.2f}".format
#***program meant for use with IDAT_OUT***
site = 'LUR' #pick site
time_interval = 'Q2' #pick Quarter Q1, Q2, Q3, Q4,
year = 2020 #pick year
file_path = r'E:\IDAT in-out\csv_out' #your file_path (idat_out data should be used)


def import_func(file_path, year, species):
    df = pd.read_csv(r'' + file_path + '\\' + site + '_' + species + '.csv', encoding='utf8', index_col=None, header=0)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['month'] = pd.DatetimeIndex(df['time']).month
    df['year'] = pd.DatetimeIndex(df['time']).year
    df = df.loc[df['year'] == year]
    if time_interval == 'Q1':
        df = df.loc[df['month'] >= 1]
        df = df.loc[df['month'] <= 3]
        df = df.drop(columns=['month', 'year'])
    elif time_interval == 'Q2':
        df = df.loc[df['month'] >= 4]
        df = df.loc[df['month'] <= 6]
        df = df.drop(columns=['month', 'year'])
    elif time_interval == 'Q3':
        df = df.loc[df['month'] >= 7]
        df = df.loc[df['month'] <= 9]
        df = df.drop(columns=['month', 'year'])
    elif time_interval == 'Q4':
        df = df.loc[df['month'] >= 10]
        df = df.loc[df['month'] <= 12]
        df = df.drop(columns=['month', 'year'])
    return df

def sigfig_func(df, column_name, sigfigs): #imput dataframe, column_name(as string), sigfigs(as number)
    list = []
    for ind, row in df.iterrows():
        if row[column_name] == df[column_name].values[0]:
            list.append(row[column_name])
            continue
        else:
            list.append(np.format_float_positional(row[column_name], precision=sigfigs, unique=False, fractional=False, trim='k'))
    return list

if site == 'LUR' or site == 'BSE' or site == 'BLV' or site == 'BRZ': #VOC's
    ethane = import_func(file_path, year, 'ethane')
    ethane = ethane.dropna()
    # voc_len = len(ethane)
    #ethane_stats = ethane.describe(percentiles=[.05, .25, .5, .75, .95])
    # print(ethane_stats.iloc[0])
    # print(voc_len)
    try:
        propane = import_func(file_path, year, 'propane')
        propane = propane.dropna()
        #propane_stats = propane.describe(percentiles=[.05, .25, .5, .75, .95])
    except:
        pass
    try:
        i_butane = import_func(file_path, year, 'i-butane')
        #i_butane_stats = i_butane.describe(percentiles=[.05, .25, .5, .75, .95])
        i_butane = i_butane.dropna()
    except:
        pass
    try:
        n_butane = import_func(file_path, year, 'n-butane')
        #n_butane_stats = n_butane.describe(percentiles=[.05, .25, .5, .75, .95])
        n_butane = n_butane.dropna()
    except:
        pass
    try:
        i_pentane = import_func(file_path, year, 'i-pentane')
        #i_pentane_stats = i_pentane.describe(percentiles=[.05, .25, .5, .75, .95])
        i_pentane = i_pentane.dropna()
    except:
        pass
    try:
        n_pentane = import_func(file_path, year, 'n-pentane')
        #n_pentane_stats = n_pentane.describe(percentiles=[.05, .25, .5, .75, .95])
        n_pentane = n_pentane.dropna()
    except:
        pass
    try:
        n_hexane = import_func(file_path, year, 'n-hexane')
        #n_hexane_stats = n_hexane.describe(percentiles=[.05, .25, .5, .75, .95])
        n_hexane = n_hexane.dropna()
    except:
        pass
    try:
        n_octane = import_func(file_path, year, 'n-octane')
        #n_octane_stats = n_octane.describe(percentiles=[.05, .25, .5, .75, .95])
        n_octane = n_octane.dropna()
    except:
        pass
    try:
        ethene = import_func(file_path, year, 'ethene')
        #ethene_stats = ethene.describe(percentiles=[.05, .25, .5, .75, .95])
        ethene = ethene.dropna()
    except:
        pass
    try:
        acetylene = import_func(file_path, year, 'acetylene')
        #acetylene_stats = acetylene.describe(percentiles=[.05, .25, .5, .75, .95])
        acetylene = acetylene.dropna()
    except:
        pass
    try:
        propene = import_func(file_path, year, 'propene')
        #propene_stats = propene.describe(percentiles=[.05, .25, .5, .75, .95])
        propene = propene.dropna()
    except:
        pass
    try:
        butadiene1_3 = import_func(file_path, year, '1_3-butadiene')
        #butadiene1_3_stats = butadiene1_3.describe(percentiles=[.05, .25, .5, .75, .95])
        butadiene1_3 = butadiene1_3.dropna()
    except:
        pass
    try:
        isoprene = import_func(file_path, year, 'isoprene')
        #isoprene_stats = isoprene.describe(percentiles=[.05, .25, .5, .75, .95])
        isoprene = isoprene.dropna()

    except:
        pass
    try:
        benzene = import_func(file_path, year, 'benzene')
        #benzene_stats = benzene.describe(percentiles=[.05, .25, .5, .75, .95])
        benzene = benzene.dropna()
    except:
        pass
    try:
        toluene = import_func(file_path, year, 'toluene')
        #toluene_stats = toluene.describe(percentiles=[.05, .25, .5, .75, .95])
        toluene = toluene.dropna()
    except:
        pass
    try:
        ethyl_benzene = import_func(file_path, year, 'ethyl-benzene')
        #ethyl_benzene_stats = ethyl_benzene.describe(percentiles=[.05, .25, .5, .75, .95])
        ethyl_benzene = ethyl_benzene.dropna()
    except:
        pass
    try:
        mp_xylene = import_func(file_path, year, 'm&p-xylene')
        #mp_xylene_stats = mp_xylene.describe(percentiles=[.05, .25, .5, .75, .95])
        mp_xylene = mp_xylene.dropna()
    except:
        pass
    try:
        o_xylene = import_func(file_path, year, 'o-xylene')
        #o_xylene_stats = o_xylene.describe(percentiles=[.05, .25, .5, .75, .95])
        o_xylene = o_xylene.dropna()
    except:
        pass
    try:
        acetaldehyde = import_func(file_path, year, 'acetaldehyde')
        #acetaldehyde_stats = acetaldehyde.describe(percentiles=[.05, .25, .5, .75, .95])
        acetaldehyde = acetaldehyde.dropna()
    except:
        pass
    try:
        acetone = import_func(file_path, year, 'acetone')
        #acetone_stats = acetone.describe(percentiles=[.05, .25, .5, .75, .95])
        acetone = acetone.dropna()
    except:
        pass

    voc = pd.merge(ethane, propane, on='time', how='outer')
    #voc['propane'] = voc['propane'].fillna(0.005)
    voc = pd.merge(voc, i_butane, on='time', how='outer')
    # voc['i-butane'] = voc['i-butane'].fillna(0.005)
    voc = pd.merge(voc, n_butane, on='time', how='outer')
    # voc['n-butane'] = voc['n-butane'].fillna(0.005)
    voc = pd.merge(voc, i_pentane, on='time', how='outer')
    voc['i-pentane'] = voc['i-pentane'].fillna(0.005)
    voc = pd.merge(voc, n_pentane, on='time', how='outer')
    voc['n-pentane'] = voc['n-pentane'].fillna(0.005)
    try:
        voc = pd.merge(voc, n_hexane, on='time', how='outer')
        voc['n-hexane'] = voc['n-hexane'].fillna(0.005)
    except:
        pass
    try:
        voc = pd.merge(voc, n_octane, on='time', how='outer')
        voc['n-octane'] = voc['n-octane'].fillna(0.005)
    except:
        pass
    try:
        voc = pd.merge(voc, ethene, on='time', how='outer')
        voc['ethene'] = voc['ethene'].fillna(0.005)
    except:
        pass
    voc = pd.merge(voc, acetylene, on='time', how='outer')
    voc['acetylene'] = voc['acetylene'].fillna(0.005)
    try:
        voc = pd.merge(voc, propene, on='time', how='outer')
        voc['propene'] = voc['propene'].fillna(0.005)
    except:
        pass
    try:
        voc = pd.merge(voc, butadiene1_3, on='time', how='outer')
        voc['1-3-butadiene'] = voc['1-3-butadiene'].fillna(0.005)
    except:
        pass
    try:
        voc = pd.merge(voc, isoprene, on='time', how='outer')
        voc['isoprene'] = voc['isoprene'].fillna(0.01)
    except:
        pass
    voc = pd.merge(voc, benzene, on='time', how='outer')
    voc['benzene'] = voc['benzene'].fillna(0.005)
    voc = pd.merge(voc, toluene, on='time', how='outer')
    voc['toluene'] = voc['toluene'].fillna(0.005)
    try:
        voc = pd.merge(voc, ethyl_benzene, on='time', how='outer')
        voc['ethyl-benzene'] = voc['ethyl-benzene'].fillna(0.003)
    except:
        pass
    try:
        voc = pd.merge(voc, mp_xylene, on='time', how='outer')
        voc['m&p-xylene'] = voc['m&p-xylene'].fillna(0.003)
    except:
        pass
    try:
        voc = pd.merge(voc, o_xylene, on='time', how='outer')
        voc['o-xylene'] = voc['o-xylene'].fillna(0.003)
    except:
        pass
    try:
        voc = pd.merge(voc, acetaldehyde, on='time', how='outer')
        voc['acetaldehyde'] = voc['acetaldehyde'].fillna(0.005)
    except:
        pass
    try:
        voc = pd.merge(voc, acetone, on='time', how='outer')
        voc['acetone'] = voc['acetone'].fillna(0.005)
    except:
        pass

    #voc.to_csv('E:\LUR Data Check\\LUR_VOC_Out\\' + site + ' ' + time_interval + ' ' + str(year) + '.csv', encoding='utf-8')

    voc['bool'] = voc['ethane'].notna()
    voc = voc.loc[voc['ethane'] > .2]
    voc = voc.loc[voc['bool'] == True]
    voc = voc.drop(['bool'], axis=1)
    print(voc)

    # voc = voc.sort_values(by=['propane'], na_position='first')


    print('nerd')
    print(voc)
    voc_stats = voc.describe(percentiles=[.05, .25, .5, .75, .95])
    stats = voc_stats.round(3)
    print(stats)



if site == 'LUR' or site == 'BSE' or site == 'BRZ' or site == 'LMA':  #Methane
    methane = import_func(file_path, year, 'ch4')
    methane_stats = methane.describe(percentiles=[.05, .25, .5, .75, .95])
    methane_stats = methane_stats.round(1)

    stats = pd.merge(stats, methane_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')
    # stats = methane_stats

if site == 'LUR' or site == 'BSE' or site == 'LMA': #Carbon Dioxide
    co2 = import_func(file_path, year, 'co2')
    co2_stats = co2.describe(percentiles=[.05, .25, .5, .75, .95])
    co2_stats = co2_stats.round(1)

    stats = pd.merge(stats, co2_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

if site == 'LUR' or site == 'LMA' or site == 'BSE' or site == 'BRZ':  #Ozone
    ozone = import_func(file_path, year, 'o3')
    for ind, row in ozone.iterrows():
        if row['o3'] < 1:
            ozone.replace(row['o3'], .5, inplace=True)
        else:
            continue
    ozone_stats = ozone.describe(percentiles=[.05, .25, .5, .75, .95])
    ozone_stats = ozone_stats.round(1)

    #ozone.to_csv('E:\LUR Data Check\\LUR_Ozone_Out\\' + site + ' ' + time_interval + ' ' + str(year) + '.csv',
             #  encoding='utf-8')

    stats = pd.merge(stats, ozone_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

    ozone8hr = import_func(file_path, year, 'o3_8hr')
    ozonefix = [ozone8hr['time'].isin(ozone['time'])]
    ozone8hr['bool'] = np.transpose(ozonefix)
    ozone8hr = ozone8hr.loc[ozone8hr['bool'] == True]
    ozone8hr = ozone8hr.drop(['bool'], axis=1)
    ozone8hr_stats = ozone8hr.describe(percentiles=[.05, .25, .5, .75, .95])
    ozone8hr_stats = ozone8hr_stats.round(1)

    stats = pd.merge(stats, ozone8hr_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

if site == 'LUR' or site == 'BSE' or site == 'BRZ': #Nitrogen Oxides
    no = import_func(file_path, year, 'no')
    # no = no.loc[no['no'] > 0]
    for ind, row in no.iterrows():
        if row['no'] < .05:
            no.replace(row['no'], .025, inplace=True)
        else:
            continue
    no_stats = no.describe(percentiles=[.05, .25, .5, .75, .95])
    no_stats = no_stats.round(2)

    stats = pd.merge(stats, no_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

    nox = import_func(file_path, year, 'nox')
    # low_nox = nox.loc[nox['nox'] < .05]
    # nox.replace(to_replace=low_nox['nox'], value=.025, inplace=True)


    for ind, row in nox.iterrows():
        if row['nox'] < .05:
            nox.replace(row['nox'], .025, inplace=True)
        else:
            continue

    nox_stats = nox.describe(percentiles=[.05, .25, .5, .75, .95])
    nox_stats = nox_stats.round(2)

    stats = pd.merge(stats, nox_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

    all_nox = pd.merge(no, nox, on='time', how='outer')
    #all_nox.to_csv('E:\LUR Data Check\\LUR_Nitrogen_Oxide_Out\\' + site + ' ' + time_interval + ' ' + str(year) + '.csv',
            #   encoding='utf-8')

if site == 'BSE': #Hydrogen Sulfides
    h2s = import_func(file_path, year, 'h2s')
    h2s_stats = h2s.describe(percentiles=[.05, .25, .5, .75, .95])
    h2s_stats = h2s_stats.round(2)

    stats = pd.merge(stats, h2s_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

if site == 'LUR' or site == 'BSE': #Particulate Matter
    pm10 = import_func(file_path, year, 'pm10')
    pm10_stats = pm10.describe(percentiles=[.05, .25, .5, .75, .95])
    pm10_stats = pm10_stats.round(2)

    pm2_5 = import_func(file_path, year, 'pm2_5')
    pm2_5_stats = pm2_5.describe(percentiles=[.05, .25, .5, .75, .95])
    pm2_5_stats = pm2_5_stats.round(2)

    pm2_5_24hr = import_func(file_path, year, 'pm2_5_24hr')
    print(pm2_5_24hr)
    pm2_5_in_24hr = [pm2_5_24hr['time'].isin(pm2_5['time'])]
    pm2_5_24hr['bool'] = np.transpose(pm2_5_in_24hr)
    pm2_5_24hr = pm2_5_24hr.loc[pm2_5_24hr['bool'] == True]
    pm2_5_24hr = pm2_5_24hr.drop(['bool'], axis=1)
    pm2_5_24hr_stats = pm2_5_24hr.describe(percentiles=[.05, .25, .5, .75, .95])
    pm2_5_24hr_stats = pm2_5_24hr_stats.round(decimals=2)

    stats = pd.merge(stats, pm10_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')
    stats = pd.merge(stats, pm2_5_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')
    stats = pd.merge(stats, pm2_5_24hr_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')

if site == 'LUR' or site == 'BSE' or site == 'BLV' or site == 'BRZ' or site == 'LMA': #Met DATA
    temp = import_func(file_path, year, 'temp')
    temp['temp'] = temp['temp'].astype(float)
    wsp = import_func(file_path, year, 'wsp')
    solr = import_func(file_path, year, 'solr')

    temp = pd.merge(temp, solr, on='time', how='outer')
    wind = pd.merge(temp, wsp, on='time', how='outer')
    wind = wind.fillna(0)
    wind_stats = wind.describe(percentiles=[.05, .25, .5, .75, .95])
    wind_stats['temp'] = wind_stats['temp'].round(0)
    wind_stats['solr'] = wind_stats['solr'].round(0)
    wind_stats['wsp'] = wind_stats['wsp'].round(1)

    stats = pd.merge(stats, wind_stats, on=stats.index, how='outer')
    stats = stats.set_index('key_0')
CAS_dictionary = {'ethane': '74-84-0', 'ethene': '74-85-1', 'propane': '74-98-6', 'propene': '115-07-1', 'i-butane': '75-28-5',
        'n-butane': '106-97-8', 'i-pentane': '78-78-4', 'n-pentane': '109-66-0', 'acetylene': '74-86-2',
        'toluene': '108-88-3', 'benzene': '71-43-2', 'isoprene': '78-79-5', 'ethyl-benzene': '100-41-4', 'm&p-xylene': '(m)108-38-3, (p)106-42-3', 'o-xylene': '95-47-6',
        'n-heptane': '142-82-5', 'n-hexane': '110-54-3', 'n-octane': '111-65-9', '1_3-butadiene': '106-99-0',
        'acetaldehyde': '75-07-0', 'acetone': '67-64-1', 'wsp': 'NA', 'solr': 'NA',
        'temp': 'NA', 'pm2_5': 'NA', 'pm10': 'NA', 'pm2_5_24hr':'NA', 'ch4': '74-82-8', 'co2': '124-38-9',
        'no': '10102-43-9', 'nox': '11104-93-1', 'o3': '10028-15-6', 'o3_8hr': '10028-15-6', 'h2s': '7783-06-4'}
CAS_list = []
for name in stats.columns:
    for key, value in CAS_dictionary.items():
        if name == key:
            CAS_list.append(value)
print(CAS_list)

unit_dictionary = {'ethane': '(ppb)', 'ethene': '(ppb)', 'propane': '(ppb)', 'propene': '(ppb)', 'i-butane': '(ppb)',
        'n-butane': '(ppb)', 'i-pentane': '(ppb)', 'n-pentane': '(ppb)', 'acetylene': '(ppb)',
        'toluene': '(ppb)', 'benzene': '(ppb)', 'isoprene': '(ppb)', 'm&p-xylene': '(ppb)', 'o-xylene': '(ppb)',
        'n-heptane': '(ppb)', 'n-hexane': '(ppb)', 'n-octane': '(ppb)', '1_3-butadiene': '(ppb)',
        'acetaldehyde': '(ppb)', 'acetone': '(ppb)', 'wsp': '(m/s, 1-min Avg)', 'solr': '(W/m^3, 1-min Avg)',
        'temp': '(Deg C, 1-min Avg)', 'pm2_5': '(μg/m^3, 1-min Avg)', 'pm10': '(μg/m^3, 1-min Avg)', 'ch4': '(ppb, 5-min Avg)', 'co2': '(ppm, 5-min Avg)',
        'no': '(ppb, 1-min Avg)', 'nox': '(ppb, 1-min Avg)', 'o3': '(ppb, 1-min Avg)', 'h2s': '(ppb)'}

print(stats.columns)

fixed_name_list = []
for name in stats.columns:
    for key, value in unit_dictionary.items():
        if str(key) in str(name) and name != 'nox':
            fixed_name_list.append(name + ' ' + value)
        elif str(key) == 'nox' and str(name) == 'nox':
            fixed_name_list.append(name + ' ' + value)
        else:
            continue
print(fixed_name_list)
stats['ethane'] = stats['ethane'].apply('{:.3f}'.format).astype(str)  #Formatting to get correct Decimal points
stats['propane'] = stats['propane'].apply('{:.3f}'.format).astype(str)
stats['i-butane'] = stats['i-butane'].apply('{:.3f}'.format).astype(str)
stats['n-butane'] = stats['n-butane'].apply('{:.3f}'.format).astype(str)
stats['i-pentane'] = stats['i-pentane'].apply('{:.3f}'.format).astype(str)
stats['n-pentane'] = stats['n-pentane'].apply('{:.3f}'.format).astype(str)
stats['n-hexane'] = stats['n-hexane'].apply('{:.3f}'.format).astype(str)
stats['ethene'] = stats['ethene'].apply('{:.3f}'.format).astype(str)
stats['acetylene'] = stats['acetylene'].apply('{:.3f}'.format).astype(str)
stats['propene'] = stats['propene'].apply('{:.3f}'.format).astype(str)
stats['isoprene'] = stats['isoprene'].apply('{:.3f}'.format).astype(str)
stats['benzene'] = stats['benzene'].apply('{:.3f}'.format).astype(str)
stats['ethyl-benzene'] = stats['ethyl-benzene'].apply('{:.3f}'.format).astype(str)
stats['m&p-xylene'] = stats['m&p-xylene'].apply('{:.3f}'.format).astype(str)
stats['o-xylene'] = stats['o-xylene'].apply('{:.3f}'.format).astype(str)
stats['ch4'] = stats['ch4'].apply('{:.1f}'.format).astype(str)
stats['co2'] = stats['co2'].apply('{:.1f}'.format).astype(str)
stats['o3'] = stats['o3'].apply('{:.1f}'.format).astype(str)
stats['o3_8hr'] = stats['o3_8hr'].apply('{:.1f}'.format).astype(str)
stats['no'] = stats['no'].apply('{:.2f}'.format).astype(str)
stats['nox'] = stats['nox'].apply('{:.2f}'.format).astype(str)
stats['pm10'] = stats['pm10'].apply('{:.2f}'.format).astype(str)
stats['pm2_5'] = stats['pm2_5'].apply('{:.2f}'.format).astype(str)
stats['pm2_5_24hr'] = stats['pm2_5_24hr'].apply('{:.2f}'.format).astype(str)
stats['temp'] = stats['temp'].apply('{:.1f}'.format).astype(str)
stats['wsp'] = stats['wsp'].apply('{:.1f}'.format).astype(str)
stats['solr'] = stats['solr'].astype(int)
stats['solr'] = stats['solr'].astype(str)

# for item in stats['ethane']:
#     stats['ethane'].replace(item, str(format(item, '.3f')), inplace=True)


print('we here')
print(stats)
print(stats.dtypes)
stats.columns=fixed_name_list

if site == 'LUR' or site == 'BSE': #particulate matter 24hour rename
    stats = stats.rename(columns={"pm2_5_24hr (μg/m^3, 1-min Avg)": "pm2_5_24hr (μg/m^3)"})

if site == 'LUR' or site == 'LMA' or site == 'BSE' or site == 'BRZ':  # Ozone 8hr rename
    stats = stats.rename(columns={"o3_8hr (ppb, 1-min Avg)": "o3_8hr (ppb)"})


stats = stats.transpose()
stats['CAS No:'] = CAS_list
# stats = stats.astype(object).T
print(stats.dtypes)
stats['count'] = stats['count'].astype(float)
stats['count'] = stats['count'].astype(int)
stats['count'] = stats['count'].astype(str)
print(stats)


#stats.to_csv('E:\\LUR quarterly statistic plots\\Better plots\\finalized\\stats ' + site + ' ' + time_interval + ' ' + str(year) + '.csv',
               #encoding='utf-8')


fig, ax = plt.subplots()
ax.set_axis_off()
table = ax.table(
    cellText=stats.values,
    rowLabels=stats.index,
    colLabels=stats.columns, cellLoc='center',
    loc ='upper center')

for (row, col), cell in table.get_celld().items():
  if (row == 0) or (col == -1):
    cell.set_text_props(fontproperties=FontProperties(weight='bold'))

table.auto_set_column_width(col=list(range(len(stats.columns))))
table.auto_set_font_size(False)
table.set_fontsize(18)
table.scale(1.25, .75)
ax.set_title(site + ' ' + time_interval + ' ' + str(year) + ' Descriptive Statistics',
             fontweight="bold", fontsize=30)

plt.show()
