import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler as scl



df = pd.read_csv("Train.csv")

#creating an array for the features or variables

df2=df



df2 = df2.drop(labels='Place_ID X Date',axis=1)
df2 = df2.drop(labels='Place_ID',axis=1)
df2 = df2.drop(labels='Date',axis=1)

dataset=df2.columns

dataR=np.array(dataset)


#dropping columns

dd = df.isnull().sum()
pd.set_option('display.max_rows',82)
pd.set_option('display.max_columns',35000)
for row in dataR:
    if dd[row] > 9000:
        df2 = df2.drop([row],axis=1)
        dataR=np.delete(dataR, np.where(dataR==row))


for row in dataR:
    if 'solar_azimuth' in row :
        continue
    if (
    'angle' in row or 'O2_sensor_altitude' in row or 
    'height' in row or 'depth' in row or 'pressure' in row or 
    'strat' in row or 'NO2_cloud' in row or 'NO2_absorb' in row or
    'O3_cloud' in row or 'O3_effect' in row or 'target_' in row or 
    '_CO_H2O_column' in row or '_HCHO_cloud_fraction' in row  or 
    'L3_CLOUD_surface_albedo' in row or 'precipitable_water_entire_atmosphere' in row or
    'SO2_cloud_fraction' in row or 'L3_SO2_SO2_column_number_density_amf' in row or 
    'L3_SO2_SO2_slant_column_number_density' in row or 'L3_SO2_SO2_column_number_density' in row or
    'L3_CO_sensor_altitude' in row 
    ):
        df2 = df2.drop([row],axis=1)
        dataR=np.delete(dataR, np.where(dataR==row))





maxTar = df2['target'].quantile(0.99977)
df2=df2[df2.target < maxTar]

maxAIaz = df2['L3_AER_AI_solar_azimuth_angle'].quantile(0.99857)
print(f"wirkin is {maxAIaz}")
df2=df2[df2.L3_AER_AI_solar_azimuth_angle < maxAIaz]



df2 = df2.replace({'L3_NO2_NO2_slant_column_number_density': 0 , 'L3_CO_CO_column_number_density': 0 ,
                    'L3_NO2_NO2_column_number_density': 0 ,'L3_O3_O3_column_number_density': 0 ,
                    'L3_HCHO_tropospheric_HCHO_column_number_density': 0,
                    'L3_HCHO_HCHO_slant_column_number_density': 0 ,'L3_NO2_tropospheric_NO2_column_number_density': 0,
                    'L3_SO2_absorbing_aerosol_index': 0 ,'L3_HCHO_tropospheric_HCHO_column_number_density_amf': 0,
                    'L3_CLOUD_cloud_fraction': 0 } ,   
                    {'L3_NO2_NO2_slant_column_number_density': np.nan, 'L3_CO_CO_column_number_density': np.nan,
                    'L3_NO2_NO2_column_number_density': np.nan, 'L3_O3_O3_column_number_density': np.nan ,
                    'L3_HCHO_tropospheric_HCHO_column_number_density': np.nan,
                    'L3_HCHO_HCHO_slant_column_number_density': np.nan, 'L3_NO2_tropospheric_NO2_column_number_density': np.nan,
                    'L3_SO2_absorbing_aerosol_index': np.nan, 'L3_HCHO_tropospheric_HCHO_column_number_density_amf':np.nan,
                    'L3_CLOUD_cloud_fraction': np.nan} )


df2=df2.dropna()


dfdisk =df2.describe()
ddsum = df2.isnull().sum()

#clears previous data and writes new data
clear = open ("Description.csv","w")
clear.truncate()
clear.close
clear = open ("CleanedT.csv","w")
clear.truncate()
clear.close
clear = open ("Sum.csv","w")
clear.truncate()
clear.close
dfdisk.to_csv('Description.csv',mode='a')
df2.to_csv('CleanedT.csv',mode='a', index=False)
ddsum.to_csv('Sum.csv',mode='a')


