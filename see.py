import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv



df = pd.read_csv("Test.csv")

#creating an array for the features or variables

df2=df

pd.set_option('display.max_rows',802)
pd.set_option('display.max_columns',35)


df2 = df2.drop(labels='Place_ID',axis=1)
df2 = df2.drop(labels='Date',axis=1)

print(df2.isnull().sum())