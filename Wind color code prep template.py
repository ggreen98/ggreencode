import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import numpy as np
import glob
import matplotlib.dates as mdates
from statistics import mean
### IMPORTANT NOTE!!!
# This program is inteded for IDAT_CSV_IN data to make input files for heatplots, windroses, and color coded timeseries plots
#LMA, LUR, and BSE have (ch4 and co2) stored under 'picarro'. BRZ has no picarro and thus no co2, but has methane stored under 'ch4'.
#LUR, BSE, and BRZ have NO & NOx data. BRZ NO & NOx data is stored in '42i'.  LUR and BSE nox is stored in 'campbell'.
#LMA, LUR, BSE, and BRZ all have Ozone data. BRZ o3 data is stored in 'o3'. LMA, LUR, and BSE 'o3' data is stored in '49c'.
#LUR, BRZ, BSE, and BLV have VOC data. For all sites all voc data is stored under 'voc'.
#LUR and BSE have PM stored under 'pm'.
#BSE has H2S stored under 'campbell'.
### IMPORTANT NOTE!!!

Quarter = 'all' #input Quarter you want as a 'string'
Year = 'all' #imput year you want or 'all'
Species = '42i' #voc, picarro, campbell, 49c, o3, pm, 42i, or ch4
SubSpecies = 'nox'#ch4, co2,PM2_5,h2s, ethane, propane, benzene ect... subspices will be plotted during datacheck
Site = 'BRZ' #input the site you want
Directory = 'E:\IDAT in-out\csv_in' #imput file directory as a string''
Saving_Directory = 'E:\Ready for wind plot dir'  #import directory you want to export file to as string'' exclude the last \
Check_data = 'ON' #imput ON or OFF if you want to make a plot to check the data
Problem = 'NO' #imput string YES or NO. If YES it means that data needs to be filtered and filtering code will run (for filtering CH4 and CO2 data)
Filter_Number = range(2) #imput number which speciefies number of times filter runs on the data set
Methane_Filter_Number = 35 #imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for Methane
CO2_Filter_Number = 5 #imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppm for CO2
NO_Filter_Number = 2#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for NO
NOx_Filter_Number = 3#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for NOx
O3_Filter_Number = 5#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for O3
PM10_Filter_Number = 35#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for PM10
PM2_5_Filter_Number = 35#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for PM2.5
H2S_Filter_Number = 1#imput number which speciefies how far apart each data point can be (+- from one point to the next) in ppb for H2S
Ready_For_Export = 'YES' #change imput from NO to YES when data is ready to be exported

#importing data section:

def picking_files_from_dir_func(file_path): #imput file path as string
    path = (r'' + file_path)
    all_files = glob.glob(path + "/*.csv")
    df_list = []
    for filename in all_files:
        if Species in filename and Site in filename:
            df = pd.read_csv(filename, index_col=None, header=1)
            df = pd.DataFrame(df)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df_list.append(df)
        else:
            continue
    return df_list

def picking_wind_files_from_dir_func(file_path): #imput file path as string
    path = (r'' + file_path)
    all_files = glob.glob(path + "/*.csv")
    wind_list = []
    for filename in all_files:
        if 'met' in filename and Site in filename:
            wind = pd.read_csv(filename, index_col=None, header=1)
            wind = pd.DataFrame(wind)
            wind['time'] = pd.to_datetime(wind['time'], unit='s')
            wind_list.append(wind)
        else:
            continue
    return wind_list

data_list = picking_files_from_dir_func(Directory)
data = pd.concat(data_list)
wind_list = picking_wind_files_from_dir_func(Directory)
wind = pd.concat(wind_list)
print('WIND!')



def data_time_set_func(df, Q = Quarter, year = Year): #When you call func input file path as a string
    df['month'] = pd.DatetimeIndex(df['time']).month
    df['year'] = pd.DatetimeIndex(df['time']).year
    if year != 'all':
        df = df.loc[df['year'] == year]
    elif year == 'all':
        print('loading all years')
    if Q == 'Q1':
        df = df.loc[df['month'] >= 1]
        df = df.loc[df['month'] <= 3]
    elif Q == 'Q2':
        df = df.loc[df['month'] >= 4]
        df = df.loc[df['month'] <= 6]
    elif Q == 'Q3':
        df = df.loc[df['month'] >= 7]
        df = df.loc[df['month'] <= 9]
    elif Q == 'Q4':
        df = df.loc[df['month'] >= 10]
        df = df.loc[df['month'] <= 12]
    elif Q == 'Q1 & Q4':
        df = df
    elif Q == 'all':
        df = df
    return df

data = data_time_set_func(data)
print(data)
wind = data_time_set_func(wind)
wind = wind.iloc[1:]
print(wind)

headers = []
for col in wind.columns:
    headers.append(col)
print(headers)

if 'wsp_avg_ms' in headers:
    wind.rename(columns={"wsp_avg_ms": "wsp"}, inplace=True)

if 'wdr_avg' in headers:
    wind.rename(columns={"wdr_avg": "wdr"}, inplace=True)



#finished importanting data
#correcting ch4 and co2 data format

if Species == 'picarro' or Species == 'ch4':
    data = data.rename(columns={"ch4dry": "ch4", "co2dry": "co2"})
    data = data.set_index(data['time'])
    # data = data.loc[data['mpv'] == 1]
    data = data.resample('1Min').mean()
    # data['ch4'] = data['ch4'] * 1000
    data.reset_index(inplace=True)

if Species == 'ch4' and Site == 'BRZ':
    data['time'] = data['time'].dt.round('1min')

#finished correcting ch4 and co2 data format
#fixing NO and NOx data format

if Species == 'campbell' and (SubSpecies == 'no' or SubSpecies == 'nox'):
    nox_data = pd.DataFrame()
    nox_data['time'] = data['time']
    nox_data['no'] = data['no']
    nox_data['nox'] = data['nox']
    data = nox_data
    wind_data = pd.DataFrame()
    wind_data['time'] = wind['time']
    wind_data['wdr'] = wind['wdr']
    wind_data['wsp'] = wind['wsp']
    wind = wind_data

if Species == '42i':
    data = data.loc[data['mode'] == 0]

#finished fixing NO and NOx data format
#fixing Ozone data format

if Species == '49c':
    data = data.loc[data['mode'] == 0]

#finished fixing Ozone data format
#fixing H2S data format

if Species == 'campbell' and SubSpecies == 'h2s':
    h2s_data = pd.DataFrame()
    h2s_data['time'] = data['time']
    h2s_data['h2s'] = data['h2s_corrected']
    data = h2s_data
    wind_data = pd.DataFrame()
    wind_data['time'] = wind['time']
    wind_data['wdr'] = wind['wdr']
    wind_data['wsp'] = wind['wsp']
    wind = wind_data


data = data.loc[data[SubSpecies] > 0]
if Species != 'voc':
    data = data.iloc[1:]

if Check_data == 'ON':
    print(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

    print(wind)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=wind['time'], y=wind['wdr'], data=wind)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=wind['time'], y=wind['wsp'], data=wind)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

### Picarro (Methane & CO2) data QC Section Below:

def methane_filter_func(df):
    df['Processed'] = df['ch4'].sub(df['ch4'].shift())
    df['Processed'].iloc[0] = df['ch4'].iloc[0]
    df = df.loc[df['Processed'] < Methane_Filter_Number]
    df = df.loc[df['Processed'] > -(Methane_Filter_Number)]
    return(df)

def co2_filter_func(df):
    df['Processed'] = df['co2'].sub(df['co2'].shift())
    df['Processed'].iloc[0] = df['co2'].iloc[0]
    df = df.loc[df['Processed'] < CO2_Filter_Number]
    df = df.loc[df['Processed'] > -(CO2_Filter_Number)]
    return(df)

if Problem == 'YES' and SubSpecies == 'ch4':
    data = data.loc[data['ch4'] > 1880]
    for i in range(len(Filter_Number)):
        data = methane_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Problem == 'YES' and SubSpecies == 'co2':
    for i in range(len(Filter_Number)):
        data = co2_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Ready_For_Export == 'YES' and Species == 'picarro' or Species == 'ch4':
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    wind = wind.loc[wind['bool'] == True]
    out_data = pd.merge(data, wind, on='time', how='outer')
    out_data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',encoding='utf-8', index = False) # relative position
    out_data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv')
    print(SubSpecies + ' data successfully exported!')

### Campbell & 42i (NO & NOx) data QC Section Below:

def NO_filter_func(df):
    df['Processed'] = df['no'].sub(df['no'].shift())
    df['Processed'].iloc[0] = df['no'].iloc[0]
    df = df.loc[df['Processed'] < NO_Filter_Number]
    df = df.loc[df['Processed'] > -(NO_Filter_Number)]
    return(df)

def NOx_filter_func(df):
    df['Processed'] = df['nox'].sub(df['nox'].shift())
    df['Processed'].iloc[0] = df['nox'].iloc[0]
    df = df.loc[df['Processed'] < NOx_Filter_Number]
    df = df.loc[df['Processed'] > -(NOx_Filter_Number)]
    return(df)

if Problem == 'YES' and SubSpecies == 'no':
    for i in range(len(Filter_Number)):
        data = NO_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Problem == 'YES' and SubSpecies == 'nox':
    for i in range(len(Filter_Number)):
        data = NOx_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Ready_For_Export == 'YES' and Species == 'campbell' or Species == '42i' or Species == 'met':
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    wind = wind.loc[wind['bool'] == True]
    out_data = pd.merge(data, wind, on='time', how='outer')
    out_data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',encoding='utf-8', index = False) # relative position
    out_data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv')
    print(SubSpecies + ' data successfully exported!')

### 49c & o3 (Ozone) data QC Section Below:

def O3_filter_func(df):
    df['Processed'] = df['o3'].sub(df['o3'].shift())
    df['Processed'].iloc[0] = df['o3'].iloc[0]
    df = df.loc[df['Processed'] < O3_Filter_Number]
    df = df.loc[df['Processed'] > -(O3_Filter_Number)]
    return(df)

if Problem == 'YES' and SubSpecies == 'o3':
    for i in range(len(Filter_Number)):
        data = O3_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Ready_For_Export == 'YES' and Species == '49c' or Species == 'o3':
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    wind = wind.loc[wind['bool'] == True]
    out_data = pd.merge(data, wind, on='time', how='outer')
    out_data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',encoding='utf-8', index = False) # relative position
    out_data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv')
    print(SubSpecies + ' data successfully exported!')

### Particulate Matter data QC Section Below:

def PM10_filter_func(df):
    df['Processed'] = df['PM10'].sub(df['PM10'].shift())
    df['Processed'].iloc[0] = df['PM10'].iloc[0]
    df = df.loc[df['Processed'] < PM10_Filter_Number]
    df = df.loc[df['Processed'] > -(PM10_Filter_Number)]
    return(df)

def PM2_5_filter_func(df):
    df['Processed'] = df['PM2_5'].sub(df['PM2_5'].shift())
    df['Processed'].iloc[0] = df['PM2_5'].iloc[0]
    df = df.loc[df['Processed'] < PM2_5_Filter_Number]
    df = df.loc[df['Processed'] > -(PM2_5_Filter_Number)]
    return(df)

if Problem == 'YES' and SubSpecies == 'PM10':
    for i in range(len(Filter_Number)):
        data = PM10_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Problem == 'YES' and SubSpecies == 'PM2_5':
    for i in range(len(Filter_Number)):
        data = PM2_5_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Ready_For_Export == 'YES' and Species == 'pm':
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    wind = wind.loc[wind['bool'] == True]
    out_data = pd.merge(data, wind, on='time', how='outer')
    out_data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',encoding='utf-8', index = False) # relative position
    out_data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv')
    print(SubSpecies + ' data successfully exported!')

### Hydrogen Sulfide data QC Section Below:

def H2S_filter_func(df):
    df['Processed'] = df['h2s'].sub(df['h2s'].shift())
    df['Processed'].iloc[0] = df['h2s'].iloc[0]
    df = df.loc[df['Processed'] < H2S_Filter_Number]
    df = df.loc[df['Processed'] > -(H2S_Filter_Number)]
    return(df)

if Problem == 'YES' and SubSpecies == 'h2s':
    for i in range(len(Filter_Number)):
        data = H2S_filter_func(data)
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.show()

if Ready_For_Export == 'YES' and Species == 'h2s':
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    wind = wind.loc[wind['bool'] == True]
    out_data = pd.merge(data, wind, on='time', how='outer')
    out_data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',encoding='utf-8', index = False) # relative position
    out_data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv')
    print(SubSpecies + ' data successfully exported!')

### VOC data QC Section Below:

if Species == 'voc':
    data['time'] = data['time'].dt.round('1min')
    bool_list = wind['time'].isin(data['time'])
    wind['bool'] = bool_list
    data = pd.merge(data, wind, on='time', how='outer')
    data = data.set_index(['time'])  # sorting index by date
    data = data.sort_index()
    data = data.reset_index()


    def get_voc_index(df):  # get list of index +- 4 indices away from central voc value
        goodind = []
        reallygoodind = []
        for ind, row in df.iterrows():
            if row['bool'] == True:
                goodind.append(ind)
        for ind, row in df.iterrows():
            for i in goodind:
                if ind > i - 5 and ind < i + 5:
                    reallygoodind.append(ind)
        return reallygoodind


    ind_list = get_voc_index(data)  # list of wind data indecies(sets of 9) that need to be averaged

    print('we here')


    def get_avg_int(df):  # remove all unwanted rows
        df['index'] = df.index
        data2 = [df['index'].isin(ind_list)]
        df['bool2'] = np.transpose(data2)
        df = df.loc[df['bool2'] == True]
        return df

    data = get_avg_int(data)
    data['wdr'] = data['wdr'].astype(float)  # converting wdr and wsp into floating point values for later functions
    data['wsp'] = data['wsp'].astype(float)
    data['x'] = np.sin((data['wdr'] * np.pi / 180))  # converting wdr into east-west and north-south components
    data['y'] = np.cos((data['wdr'] * np.pi / 180))

    def avg_wdr_match_voc_interval(
            df):  # avg wind spead and wind direction in 9 row intervals during voc sampling interval
        avg_wsp_list = []
        avg_list = []
        fixedind = []
        last_ind = None

        for ind, row in df.iterrows():
            if ind - 1 == last_ind:
                xlist.append(row['x'])
                ylist.append(row['y'])
                last_ind = ind
                avgwsplist.append(row['wsp'])
            elif ind == ind_list[0]:
                avgwsplist = []
                xlist = []
                ylist = []
                xlist.append(row['x'])
                ylist.append(row['y'])
                avgwsplist.append(row['wsp'])
                last_ind = ind
            else:
                centerind = (last_ind - 4)
                fixedind.append(centerind)
                xtot = sum(xlist)
                ytot = sum(ylist)
                dev = (xtot / ytot)
                ark = (np.arctan(dev))
                avg = (ark * 57.2958)
                if xtot > 0 and ytot > 0:  # if loop used to determin which quadrent wind direction data fit in
                    avg_list.append(avg)
                elif xtot > 0 and ytot < 0:
                    avg2nd = (avg + 180)
                    avg_list.append(avg2nd)
                elif xtot < 0 and ytot < 0:
                    avg3rd = (avg + 180)
                    avg_list.append(avg3rd)
                elif xtot < 0 and ytot > 0:
                    avg4th = (avg + 360)
                    avg_list.append(avg4th)
                else:
                    avg_list.append(np.nan)
                wsp = mean(avgwsplist)
                avg_wsp_list.append(wsp)
                xlist = []
                ylist = []
                avgwsplist = []
                avgwsplist.append(row['wsp'])
                xlist.append(row['x'])
                ylist.append(row['y'])
                last_ind = ind

        avgdata = pd.DataFrame(fixedind, columns=['index'])
        avgdata['avg_wdr'] = avg_list
        avgdata['avg_wsp'] = avg_wsp_list
        return avgdata

    avgdata = (avg_wdr_match_voc_interval(data))
    data = pd.merge(data, avgdata, on=['index'], how='outer')  # merging data frames
    data = data.loc[data['bool'] == True]

    def win_int_sorting_func(df):
        direction_list = []
        for ind, row in df.iterrows():
            if row['avg_wdr'] <= 30:
                direction_list.append('0-30 deg')
            elif 30 < row['avg_wdr'] <= 60:
                direction_list.append('30-60 deg')
            elif 60 < row['avg_wdr'] <= 90:
                direction_list.append('60-90 deg')
            elif 90 < row['avg_wdr'] <= 120:
                direction_list.append('90-120 deg')
            elif 120 < row['avg_wdr'] <= 150:
                direction_list.append('120-150 deg')
            elif 150 < row['avg_wdr'] <= 180:
                direction_list.append('150-180 deg')
            elif 180 < row['avg_wdr'] <= 210:
                direction_list.append('180-210 deg')
            elif 210 < row['avg_wdr'] <= 240:
                direction_list.append('210-240 deg')
            elif 240 < row['avg_wdr'] <= 270:
                direction_list.append('240-270 deg')
            elif 270 < row['avg_wdr'] <= 300:
                direction_list.append('270-300 deg')
            elif 300 < row['avg_wdr'] <= 330:
                direction_list.append('300-330 deg')
            elif 330 < row['avg_wdr'] <= 360:
                direction_list.append('330-360 deg')
            else:
                direction_list.append('No Wind Data')
        return direction_list


    data['direction'] = win_int_sorting_func(data)
    better_data = pd.DataFrame()
    better_data['time'] = data['time']
    better_data['wdr'] = data['avg_wdr']
    better_data['wsp'] = data['avg_wsp']
    better_data['direction'] = data['direction']
    better_data['ethane'] = data['ethane']
    better_data['ethene'] = data['ethene']
    better_data['propane'] = data['propane']
    better_data['propene'] = data['propene']
    better_data['i-butane'] = data['i-butane']
    better_data['n-butane'] = data['n-butane']
    better_data['i-pentane'] = data['i-pentane']
    better_data['n-pentane'] = data['n-pentane']
    better_data['acetylene'] = data['acetylene']
    better_data['benzene'] = data['benzene']
    better_data['toluene'] = data['toluene']
    data = better_data
    data['benz_toul'] = (data['benzene']/data['toluene'])
    data['pro_eth'] = (data['propane'] / data['ethane'])
    data['i_n_pent'] = (data['i-pentane'] / data['n-pentane'])
    print(data)

    color_dict = dict({'0-30 deg': '#3c02a3',
                       '30-60 deg': '#0247fe',
                       '60-90 deg': '#0291cd',
                       '90-120 deg': '#66b132',
                       '120-150 deg': '#d0e92b',
                       '150-180 deg': '#fffe32',
                       '180-210 deg': '#fcb904',
                       '210-240 deg': '#fb9902',
                       '240-270 deg': '#fd5308',
                       '270-300 deg': '#fe2712',
                       '300-330 deg': '#a7194b',
                       '330-360 deg': '#8601b0',
                       'No Wind Data': '#000000'})

    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    ax1 = sns.scatterplot(x=data['time'], y=data[SubSpecies], data=data,
                          hue=data['direction'], hue_order=['0-30 deg',
                                                            '30-60 deg', '60-90 deg', '90-120 deg', '120-150 deg',
                                                            '150-180 deg', '180-210 deg', '210-240 deg', '240-270 deg',
                                                            '270-300 deg',
                                                            '300-330 deg', '330-360 deg', 'No Wind Data'],
                          palette=color_dict, s=300)
    ax1.set_xlim(data['time'].min(), data['time'].max())
    plt.subplots_adjust(right=0.80)
    ax1.legend(fontsize=20, bbox_to_anchor=(1, 1), loc='upper left', markerscale=2)
    plt.show()

if Ready_For_Export == 'YES' and Species == 'voc':
    data.to_csv(Site + '_' + Quarter + '_' + str(Year) + '_' + SubSpecies + '_Heat_Plot_Ready.csv',
                    encoding='utf-8', index=False)  # relative position
    data.to_csv(r'' + Saving_Directory + '\_' + Site + '_' + Quarter + '_' + str(
        Year) + '_' + 'voc' + '_Heat_Plot_Ready.csv')
    print('VOC data successfully exported!')



