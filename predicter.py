import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from train import clf



df = pd.read_csv("Test.csv")

#creating an array for the features or variables

df2=df




df2 = df2.drop(labels='Place_ID',axis=1)
df2 = df2.drop(labels='Date',axis=1)

dataset=df2.columns

dataR=np.array(dataset)


#dropping columns

dd = df.isnull().sum()
pd.set_option('display.max_rows',82)
pd.set_option('display.max_columns',35000)
for row in dataR:
    if 'Place_ID X Date' in row:
        continue
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


df2['L3_CO_CO_column_number_density'].fillna(df2['L3_CO_CO_column_number_density'].median(), inplace=True)
df2['L3_HCHO_tropospheric_HCHO_column_number_density'].fillna(df2['L3_HCHO_tropospheric_HCHO_column_number_density'].median(), inplace=True)
df2['L3_HCHO_HCHO_slant_column_number_density'].fillna(df2['L3_HCHO_HCHO_slant_column_number_density'].median(), inplace=True)
df2['L3_NO2_NO2_slant_column_number_density'].fillna(df2['L3_NO2_NO2_slant_column_number_density'].median(), inplace=True)
df2['L3_AER_AI_sensor_altitude'].fillna(df2['L3_AER_AI_sensor_altitude'].mean(), inplace=True)
df2['L3_NO2_tropospheric_NO2_column_number_density'].fillna(df2['L3_NO2_tropospheric_NO2_column_number_density'].median(), inplace=True)
df2['L3_NO2_NO2_column_number_density'].fillna(df2['L3_NO2_NO2_column_number_density'].median(), inplace=True)
df2['u_component_of_wind_10m_above_ground'].fillna(df2['u_component_of_wind_10m_above_ground'].median(), inplace=True)
df2['L3_O3_O3_column_number_density'].fillna(df2['L3_O3_O3_column_number_density'].mean(), inplace=True)
df2['L3_SO2_absorbing_aerosol_index'].fillna(df2['L3_SO2_absorbing_aerosol_index'].median(), inplace=True)
df2['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].fillna(df2['L3_HCHO_tropospheric_HCHO_column_number_density_amf'].median(), inplace=True)
df2['L3_AER_AI_absorbing_aerosol_index'].fillna(df2['L3_AER_AI_absorbing_aerosol_index'].median(), inplace=True)
df2['L3_CLOUD_cloud_fraction'].fillna(df2['L3_CLOUD_cloud_fraction'].median(), inplace=True)
df2['temperature_2m_above_ground'].fillna(df2['temperature_2m_above_ground'].mean(), inplace=True)
df2['specific_humidity_2m_above_ground'].fillna(df2['specific_humidity_2m_above_ground'].median(), inplace=True)
df2['v_component_of_wind_10m_above_ground'].fillna(df2['v_component_of_wind_10m_above_ground'].median(), inplace=True)
df2['relative_humidity_2m_above_ground'].fillna(df2['relative_humidity_2m_above_ground'].median(), inplace=True)
df2['L3_AER_AI_solar_azimuth_angle'].fillna(df2['L3_AER_AI_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_O3_solar_azimuth_angle'].fillna(df2['L3_O3_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_CLOUD_solar_azimuth_angle'].fillna(df2['L3_CLOUD_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_NO2_solar_azimuth_angle'].fillna(df2['L3_NO2_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_HCHO_solar_azimuth_angle'].fillna(df2['L3_HCHO_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_SO2_solar_azimuth_angle'].fillna(df2['L3_SO2_solar_azimuth_angle'].mean(), inplace=True)
df2['L3_CO_solar_azimuth_angle'].fillna(df2['L3_CO_solar_azimuth_angle'].mean(), inplace=True)


df3=pd.DataFrame(data={'Place_ID X Date' :df2['Place_ID X Date']})

df2=df2.drop(labels= 'Place_ID X Date', axis=1)
pred =clf.predict(df2)

nf =pd.DataFrame(data={"target":pred})

ff=pd.concat([df3,nf] ,axis=1,join='inner')


#clears previous data and writes new data

clear = open ("Submission.csv","w")
clear.truncate()
clear.close
ff.to_csv('Submission.csv',mode='a', index=False)
