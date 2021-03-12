import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as scipy
from scipy import stats
import scipy.odr as odr

file_path = 'E:\Ready for wind plot dir\_BRZ_all_all_voc_Heat_Plot_Ready'
X_species = 'n-pentane'
Y_species = 'i-pentane'

if '-' in X_species:
    X_name = X_species[:2] + X_species[2].upper() + X_species[3:]
else:
    X_name = X_species.title()

if '-' in Y_species:
    Y_name = Y_species[:2] + Y_species[2].upper() + Y_species[3:]
else:
    Y_name = Y_species.title()


print(X_name)

if '2019' in file_path:
    year = '2019'
elif '2020' in file_path:
    year = '2020'
elif '2021' in file_path:
    year = '2021'
elif '2022' in file_path:
    year = '2022'
elif 'all' in file_path:
    year = 'all'
else:
    raise ValueError('Year not recognized')

if 'Q1' in file_path:
    quarter = 'Q1'
elif 'Q2' in file_path:
    quarter = 'Q2'
elif 'Q3' in file_path:
    quarter = 'Q3'
elif 'Q4' in file_path:
    quarter = 'Q4'
elif 'all' in file_path:
    quarter = 'all'
else:
    raise ValueError('Quarter not recognized')

if 'LUR' in file_path:
    site = 'LUR'
elif 'LMA' in file_path:
    site = 'LMA'
elif 'BSE' in file_path:
    site = 'BSE'
elif 'BLV' in file_path:
    site = 'BLV'
elif 'BRZ' in file_path:
    site = 'BRZ'
else:
    raise ValueError('Site not recognized')


df = pd.read_csv(r'' + file_path + '.csv', encoding='utf8', index_col=None, header=0)
print(df)
df = df.loc[df[X_species] > 0]
df = df.loc[df[Y_species] > 0]

def odr_data_prep(x_species, y_species):
    data = pd.DataFrame()
    data['x_species'] = x_species
    data['y_species'] = y_species
    data.dropna(inplace=True)
    return data

data = odr_data_prep(df[X_species], df[Y_species]) # imput x species, y species
print(data)
data = data.loc[data['x_species'] > 0]
data = data.loc[data['y_species'] > 0]

sx = data['x_species'].std(axis=0, skipna=True)
sy = data['y_species'].std(axis=0, skipna=True)


linreg = scipy.stats.linregress(data['x_species'], data['y_species'])
print(linreg)


def f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]


def perform_odr(x, y, sx, sy):
    linear = odr.Model(f)
    mydata = odr.Data(x, y, wd=1./np.power(sx,2), we=1./np.power(sy,2))
    myodr = odr.ODR(mydata, linear, beta0=linreg[0:2])
    output = myodr.run()
    output.pprint()
    return output


regression = perform_odr(data['x_species'], data['y_species'], sx, sy)
odr_results = ('ODR Results, Beta: ' + str(regression.beta) + ', Beta Std Error: ' + str(regression.sd_beta)
               + ', Beta Covariance: ' + str(regression.cov_beta) + ', Residual Variance: ' + str(regression.res_var) +
               ', Inverse Condition #: ' + str(regression.inv_condnum))


odr_slope = str(round(regression.beta[0], 3))
odr_intercept = str(round(regression.beta[1], 3))
lin_slope = str(round(linreg[0], 3))
lin_intercept = str(round(linreg[1], 3))


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

if year != 'all':
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    fig = plt.figure()
    ax1 = sns.scatterplot(x=df[X_species], y=df[Y_species], data=df,
    hue=df['direction'], hue_order = ['0-30 deg',
                        '30-60 deg', '60-90 deg', '90-120 deg', '120-150 deg',
                        '150-180 deg', '180-210 deg', '210-240 deg', '240-270 deg', '270-300 deg',
                        '300-330 deg', '330-360 deg', 'No Wind Data'],
                        palette = color_dict, s=350)
    ax1.set_title(site + ' ' + quarter + ' ' + year + ', ' + Y_name + ' - ' + X_name + ' Relationship', fontsize=45)
    ax1.set_ylabel(Y_name + ' (ppb)', labelpad=15, fontsize=45)
    ax1.set_xlabel(X_name + ' (ppb)', labelpad=15, fontsize=45)
    plt.plot(data['x_species'], regression.beta[0] * data['x_species'] + regression.beta[1], color='k',
             label = ('(y = ' + odr_slope + ' * x + ' + odr_intercept + ')'))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(right=0.75)
    handles1, labels1 = ax1.get_legend_handles_labels()
    print(handles1)
    handles1.pop(1)
    labels1.pop(1)
    ax1.legend(fontsize = 30,bbox_to_anchor=(1, 1),labels=labels1, handles=handles1,  loc='upper left', markerscale=3)
    plt.show()
elif year == 'all':
    sns.set(style="darkgrid")
    sns.set(font_scale=4)
    fig = plt.figure()
    ax1 = sns.scatterplot(x=df[X_species], y=df[Y_species], data=df,
    hue=df['direction'], hue_order = ['0-30 deg',
                        '30-60 deg', '60-90 deg', '90-120 deg', '120-150 deg',
                        '150-180 deg', '180-210 deg', '210-240 deg', '240-270 deg', '270-300 deg',
                        '300-330 deg', '330-360 deg', 'No Wind Data'],
                        palette = color_dict, s=350)
    ax1.set_title(site + ' Jun 2017 - Feb 2021' + ', ' + Y_name + ' - ' + X_name + ' Relationship', fontsize=45)
    ax1.set_ylabel(Y_name + ' (ppb)', labelpad=15, fontsize=45)
    ax1.set_xlabel(X_name + ' (ppb)', labelpad=15, fontsize=45)
    plt.plot(data['x_species'], regression.beta[0] * data['x_species'] + regression.beta[1], color='k',
             label = ('(y = ' + odr_slope + ' * x + ' + odr_intercept + ')'))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(right=0.75)
    handles1, labels1 = ax1.get_legend_handles_labels()
    print(handles1)
    handles1.pop(1)
    labels1.pop(1)
    ax1.legend(fontsize = 30,bbox_to_anchor=(1, 1),labels=labels1, handles=handles1,  loc='upper left', markerscale=3)
    plt.show()