import numpy as np
import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv


clt = pd.read_csv("CleanedT.csv")

corr_matrx=np.abs(clt.corr())

viewt=corr_matrx['target'].sort_values(ascending=False)

#plt.figure(3) 
#plt.subplot(121) 
#sns.distplot(clt['L3_CO_solar_azimuth_angle']);
#plt.subplot(122) 
#plt['L3_CO_solar_azimuth_angle'].plot.box(figsize=(16,5))
#plt.figure(1)
#sns.scatterplot(data=clt,x='target',y='L3_CO_solar_azimuth_angle')
#plt.show()

clear = open ("corr_table2.csv","w")
clear.truncate()
clear.close
viewt.to_csv('corr_table2.csv',mode='a')



#sns.pairplot( hue= 'target' ,data=clt)

