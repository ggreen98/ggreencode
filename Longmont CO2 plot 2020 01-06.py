import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#importing data as CSV for month 1 LMA
df = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_01_LMA_picarro.csv',encoding='utf8')
df = df.dropna(axis=1, how='any')
df = df.reset_index()
df = df.drop(index=0)
# setting colum headers
df.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df['mpv'] = df['mpv'].astype(float)
df['CO2'] = df['CO2'].astype(float)
df['CH4'] = df['CH4'].astype(float)
df['CH4'] = df['CH4'].values * 1000
df['month'] = 'Jan'

#importing data as CSV for month 2 LMA
df1 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_02_LMA_picarro.csv',encoding='utf8')
df1 = df1.dropna(axis=1, how='any')
df1 = df1.reset_index()
df1 = df1.drop(index=0)
# setting colum headers
df1.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df1['mpv'] = df1['mpv'].astype(float)
df1['CO2'] = df1['CO2'].astype(float)
df1['CH4'] = df1['CH4'].astype(float)
df1['CH4'] = df1['CH4'].values * 1000
df1['month'] = 'Feb'

#importing data as CSV for month 3 LMA
df2 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_03_LMA_picarro.csv',encoding='utf8')
df2 = df2.dropna(axis=1, how='any')
df2 = df2.reset_index()
df2 = df2.drop(index=0)
# setting colum headers
df2.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df2['mpv'] = df2['mpv'].astype(float)
df2['CO2'] = df2['CO2'].astype(float)
df2['CH4'] = df2['CH4'].astype(float)
df2['CH4'] = df2['CH4'].values * 1000
df2['month'] = 'Mar'

#importing data as CSV for month 4 LMA
df3 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_04_LMA_picarro.csv',encoding='utf8')
df3 = df3.dropna(axis=1, how='any')
df3 = df3.reset_index()
df3 = df3.drop(index=0)
# setting colum headers
df3.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df3['mpv'] = df3['mpv'].astype(float)
df3['CO2'] = df3['CO2'].astype(float)
df3['CH4'] = df3['CH4'].astype(float)
df3['CH4'] = df3['CH4'].values * 1000
df3['month'] = 'Apr'

#importing data as CSV for month 5 LMA
df4 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_05_LMA_picarro.csv',encoding='utf8')
df4 = df4.dropna(axis=1, how='any')
df4 = df4.reset_index()
df4 = df4.drop(index=0)
# setting colum headers
df4.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df4['mpv'] = df4['mpv'].astype(float)
df4['CO2'] = df4['CO2'].astype(float)
df4['CH4'] = df4['CH4'].astype(float)
df4['CH4'] = df4['CH4'].values * 1000
df4['month'] = 'May'

#importing data as CSV for month 6 LMA
df5 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_06_LMA_picarro.csv',encoding='utf8')
df5 = df5.dropna(axis=1, how='any')
df5 = df5.reset_index()
df5 = df5.drop(index=0)
# setting colum headers
df5.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
df5['mpv'] = df5['mpv'].astype(float)
df5['CO2'] = df5['CO2'].astype(float)
df5['CH4'] = df5['CH4'].astype(float)
df5['CH4'] = df5['CH4'].values * 1000
df5['month'] = 'Jun'

#importing data as CSV for month 1 LUR
dfu = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_01_LUR_picarro.csv',encoding='utf8')
dfu = dfu.dropna(axis=1, how='any')
dfu = dfu.reset_index()
dfu = dfu.drop(index=0)
# setting colum headers
dfu.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu['mpv'] = dfu['mpv'].astype(float)
dfu['CO2'] = dfu['CO2'].astype(float)
dfu['CH4'] = dfu['CH4'].astype(float)
dfu['CH4'] = dfu['CH4'].values * 1000
dfu['month'] = 'Jan'

#importing data as CSV for month 2 LUR
dfu1 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_02_LUR_picarro.csv',encoding='utf8')
dfu1 = dfu1.dropna(axis=1, how='any')
dfu1 = dfu1.reset_index()
dfu1 = dfu1.drop(index=0)
# setting colum headers
dfu1.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu1['mpv'] = dfu1['mpv'].astype(float)
dfu1['CO2'] = dfu1['CO2'].astype(float)
dfu1['CH4'] = dfu1['CH4'].astype(float)
dfu1['CH4'] = dfu1['CH4'].values * 1000
dfu1['month'] = 'Feb'

#importing data as CSV for month 3 LUR
dfu2 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_03_LUR_picarro.csv',encoding='utf8')
dfu2 = dfu2.dropna(axis=1, how='any')
dfu2 = dfu2.reset_index()
dfu2 = dfu2.drop(index=0)
# setting colum headers
dfu2.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu2['mpv'] = dfu2['mpv'].astype(float)
dfu2['CO2'] = dfu2['CO2'].astype(float)
dfu2['CH4'] = dfu2['CH4'].astype(float)
dfu2['CH4'] = dfu2['CH4'].values * 1000
dfu2['month'] = 'Mar'

#importing data as CSV for month 4 LUR
dfu3 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_04_LUR_picarro.csv',encoding='utf8')
dfu3 = dfu3.dropna(axis=1, how='any')
dfu3 = dfu3.reset_index()
dfu3 = dfu3.drop(index=0)
# setting colum headers
dfu3.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu3['mpv'] = dfu3['mpv'].astype(float)
dfu3['CO2'] = dfu3['CO2'].astype(float)
dfu3['CH4'] = dfu3['CH4'].astype(float)
dfu3['CH4'] = dfu3['CH4'].values * 1000
dfu3['month'] = 'Apr'

#importing data as CSV for month 5 LUR
dfu4 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_05_LUR_picarro.csv',encoding='utf8')
dfu4 = dfu4.dropna(axis=1, how='any')
dfu4 = dfu4.reset_index()
dfu4 = dfu4.drop(index=0)
# setting colum headers
dfu4.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu4['mpv'] = dfu4['mpv'].astype(float)
dfu4['CO2'] = dfu4['CO2'].astype(float)
dfu4['CH4'] = dfu4['CH4'].astype(float)
dfu4['CH4'] = dfu4['CH4'].values * 1000
dfu4['month'] = 'May'

#importing data as CSV for month 6 LUR
dfu5 = pd.read_csv (r'C:\Users\Gabriel Greenberg\Documents\Longmont\20_06_LUR_picarro.csv',encoding='utf8')
dfu5 = dfu5.dropna(axis=1, how='any')
dfu5 = dfu5.reset_index()
dfu5 = dfu5.drop(index=0)
# setting colum headers
dfu5.columns = ['time', 'mpv', 'CO2', 'CH4']
# converting objects to newmerical values
dfu5['mpv'] = dfu5['mpv'].astype(float)
dfu5['CO2'] = dfu5['CO2'].astype(float)
dfu5['CH4'] = dfu5['CH4'].astype(float)
dfu5['CH4'] = dfu5['CH4'].values * 1000
dfu5['month'] = 'Jun'

# merging dataframes before plotting
framesLMA = [df, df1, df2, df3, df4, df5]
framesLUR = [dfu, dfu1, dfu2, dfu3, dfu4, dfu5]
#framesBRZ = []

bbyLMA = pd.concat(framesLMA)
bbyLMA = bbyLMA.loc[bbyLMA['mpv'] == 1]
bbyLUR = pd.concat(framesLUR)
bbyLUR = bbyLUR.loc[bbyLUR['mpv'] == 1]
#bbyBRZ = pd.concat(framesBRZ)

con = pd.concat([bbyLUR.assign(dataset='LUR'), bbyLMA.assign(dataset='LMA')])
con = con.loc[con['CO2'] > 0]
con['Site'] = con['dataset']


# plotting
sns.set(style="darkgrid")
sns.set(font_scale=2)
ax1 = sns.boxplot(x=con['month'], y=con['CO2'], data=con, hue=con['Site'], showfliers=False)
ax1.set_title('CO2 Mixing Ratio at LMA & LUR by Month', fontsize=30)
ax1.set_ylabel('CO2 Mixing Ratio (ppm)', fontsize=25)
ax1.set_xlabel('Month', fontsize=25)
ax1.legend(fontsize = 20)
plt.show()