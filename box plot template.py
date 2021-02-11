import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import numpy as np
import glob
import matplotlib.dates as mdates
import re

Quarter = 'Q4' #input Quarter you want as a 'string'
Year = 2020 #imput year you want
file_path = 'E:\IDAT in-out\csv_out\BRZ_o3' #input your file path (it does not matter which site you chose in your file path)
Sites = ['BSE', 'BRZ', 'LMA', 'LUR'] #input the sites you want to include as a series of comma separated strings (input the sites in the order you want them to show up on the box plot)


if 'BRZ' in file_path:
    file_path = file_path.replace('BRZ', 'site')
elif 'BSE' in file_path:
    file_path = file_path.replace('BSE', 'site')
elif 'BLV' in file_path:
    file_path = file_path.replace('BLV', 'site')
elif 'LMA' in file_path:
    file_path = file_path.replace('LMA', 'site')
elif 'LUR' in file_path:
    file_path = file_path.replace('LUR', 'site')
else:
    raise ValueError('Site not recognized')


if 'ch4' in file_path:
    Species = 'Methane'
    species_name = 'ch4'
elif 'co2' in file_path:
    Species = 'CO2'
    species_name = 'co2'
elif 'o3' in file_path:
    Species = 'Ozone'
    species_name = 'o3'
elif 'nox' in file_path:
    Species = 'Nitrogen Oxides'
    species_name = 'nox'
elif 'no' in file_path:
    Species = 'Nitric Oxide'
    species_name = 'no'
elif 'PM2_5' in file_path:
    Species = 'PM 2.5'
    species_name = 'PM2_5'
elif 'PM10' in file_path:
    Species = 'PM 10'
    species_name = 'PM10'
elif 'ethane' in file_path:
    Species = 'Ethane'
    species_name = 'ethane'
elif 'benzene' in file_path:
    Species = 'Benzene'
    species_name = 'benzene'
elif 'propane' in file_path:
    Species = 'Propane'
    species_name = 'propane'
elif 'acetylene' in file_path:
    Species = 'Acetylene'
    species_name = 'acetylene'
else:
    raise ValueError('Species not recognized')

print(species_name)
def load_in_csv_out_func(file_path, Q = Quarter, year = Year): #When you call func input file path as a string
    df = pd.read_csv(r'' + file_path + '.csv', encoding='utf8', index_col=None, header=0)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['month'] = pd.DatetimeIndex(df['time']).month
    df['year'] = pd.DatetimeIndex(df['time']).year
    df = df.loc[df['year'] == year]
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
    elif Q == 'Q1 & Q2':
        df = df.loc[df['month'] >= 1]
        df = df.loc[df['month'] <= 6]
    return df

site_list = []
for site in Sites:
    if site == 'BRZ':
        BRZ = load_in_csv_out_func(file_path.replace('site', site))
        site_list.append(BRZ)
    elif site == 'BSE':
        BSE = load_in_csv_out_func(file_path.replace('site', site))
        site_list.append(BSE)
    elif site == 'BLV':
        BLV = load_in_csv_out_func(file_path.replace('site', site))
        site_list.append(BLV)
    elif site == 'LMA':
        LMA = load_in_csv_out_func(file_path.replace('site', site))
        site_list.append(LMA)
    elif site == 'LUR':
        LUR = load_in_csv_out_func(file_path.replace('site', site))
        site_list.append(LUR)


if len(Sites) == 5:
    data = pd.concat([site_list[0].assign(dataset=Sites[0]), site_list[1].assign(dataset=Sites[1]),
                      site_list[2].assign(dataset=Sites[2]), site_list[3].assign(dataset=Sites[3]),
                      site_list[4].assign(dataset=Sites[4])])
elif len(Sites) == 4:
    data = pd.concat([site_list[0].assign(dataset=Sites[0]), site_list[1].assign(dataset=Sites[1]),
                      site_list[2].assign(dataset=Sites[2]), site_list[3].assign(dataset=Sites[3])])
elif len(Sites) == 3:
    data = pd.concat([site_list[0].assign(dataset=Sites[0]), site_list[1].assign(dataset=Sites[1]),
                      site_list[2].assign(dataset=Sites[2])])
elif len(Sites) == 2:
    data = pd.concat([site_list[0].assign(dataset=Sites[0]), site_list[1].assign(dataset=Sites[1])])
else:
    data = site_list[0]

data.rename(columns={"dataset": "site"}, inplace=True)

data['time'] = data['time'].dt.month
data['time'].replace({1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct",
                      11:"Nov", 12:"Dec"}, inplace=True)

def site_color_assign_func(data): #when you call function input concatonated dataframe
    color_list = []
    site_BRZ = False
    site_BSE = False
    site_BLV = False
    site_LUR = False
    site_LMA = False
    for ind, row in data.iterrows():
        if row['site'] == 'BRZ' and site_BRZ == False:
            color_list.append('#008BE8')
            site_BRZ = True
        elif row['site'] == 'BRZ' and site_BRZ == True:
            continue
        elif row['site'] == 'BSE' and site_BSE == False:
            color_list.append('#0CF215')
            site_BSE = True
        elif row['site'] == 'BSE' and site_BSE == True:
            continue
        elif row['site'] == 'BLV' and site_BLV == False:
            color_list.append('#33806c')
            site_BLV = True
        elif row['site'] == 'BLV' and site_BLV == True:
            continue
        elif row['site'] == 'LUR' and site_LUR == False:
            color_list.append('#B10CF2')
            site_LUR = True
        elif row['site'] == 'LUR' and site_LUR == True:
            continue
        elif row['site'] == 'LMA' and site_LMA == False:
            color_list.append('#E84B10')
            site_LMA = True
        elif row['site'] == 'LMA' and site_LMA == True:
            continue
    return color_list

color = site_color_assign_func(data)

#plotting
sns.set(style="darkgrid")
sns.set(font_scale=4)
sns.set_palette(palette=color)  #Blue(brz "#008BE8"), Purple(lur "#B10CF2"), Green(BSE "#0CF215"), Darkgreen(BLV "#33806c"), Orange(LMA "#E84B10")
ax1 = sns.boxplot(x=data['time'], y=data[species_name], data=data, hue=data['site'],  whis=[5, 95], showmeans=True,  showfliers=False,
                  meanprops={"marker": "o",
                             "markerfacecolor": "white",
                             "markeredgecolor": "black",
                             "markersize": "15"}
                  )

if len(Sites) == 5:
    ax1.set_title(Sites[0] + ', ' + Sites[1] + ', ' + Sites[2] + ', ' + Sites[3] + ', & ' + Sites[4
    ] + ' ' + Species + ' ' + Quarter + ' ' + str(Year))
elif len(Sites) == 4:
    ax1.set_title(Sites[0] + ', ' + Sites[1] + ', ' + Sites[2] + ', & ' + Sites[3] + ' ' + Species + ' ' + Quarter + ' ' + str(Year))
elif len(Sites) == 3:
    ax1.set_title(Sites[0] + ', ' + Sites[1] + ', & ' + Sites[2] + ' ' + Species + ' ' + Quarter + ' ' + str(Year))
elif len(Sites) == 2:
    ax1.set_title(Sites[0] + ' & ' + Sites[1] + ' ' + Species + ' ' + Quarter + ' ' + str(Year))
else:
    ax1.set_title(Sites[0] + ' ' + Species + ' ' + Quarter + ' ' + str(Year))

ax1.set_ylabel(Species + ' (ppb)', labelpad=15)
ax1.set_xlabel('')
ax1.legend(bbox_to_anchor=(1, 1.029))
plt.subplots_adjust(right=0.84)
plt.show()