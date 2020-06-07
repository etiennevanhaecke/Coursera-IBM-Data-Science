# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import pandas as pd

df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )

print('Data downloaded and read into a dataframe!')

print(df_can.head())
print(df_can.tail())
print(df_can.shape)
print(df_can.info())

# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
# let's rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
# set the country name as index - useful for quickly looking up countries using .loc method
df_can.set_index('Country', inplace=True)
# years that we will be using in this lesson
years = list(map(str, range(1980, 2013)))
# add total column das colunas 1980 a 2013
df_can['Total'] = df_can[years].sum(axis=1)

#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

print("Version de matplotlib: "+mpl.__version__)

#Let's use a pie chart to explore the proportion (percentage) of new immigrants grouped by continents for the entire time period from 1980 to 2013. 
df_contin = df_can.groupby('Continent', axis=0).sum() #Calculo da soma de imigrantes agrupando os paises por continente
df_contin['Total'].plot(kind='pie', autopct='%1.1f%%', startangle=55,
         figsize=(5,6), pctdistance=0.85, explode=[0, 0.07, 0.025, 0, 0, 0])
plt.title("Reparticao imigracao por continente")

#Using a pie chart, explore the proportion (percentage) of new immigrants grouped by continents in the year 2013.
df_contin['2013'].plot(kind='pie', autopct='%1.1f%%', startangle=55,
         figsize=(5,6), pctdistance=0.85, explode=[0.025, 0.07, 0.06, 0, 0, 0])

#Let's plot the box plot for the Japanese immigrants between 1980 - 2013.
df_jap = df_can.loc['Japan', years]
#OU (1ero gera uma serie pandas e o segundo um data frame pandas, mas os 2 permitem usar plot)
df_jap = df_can.loc[['Japan'], years].T
df_jap.plot(kind='box')
plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')
plt.show()

df_jap.describe()

#Compare the distribution of the number of new immigrants from India and China for the period 1980 - 2013.
df_can.loc[['India', 'China'], years].T.plot(kind='box')
df_can.loc[['India', 'China'], years].T.describe()
plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')

df_can.loc[['India', 'China'], years].T.describe()

df_IA = df_can.loc[['India', 'China'], years].T
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(122)
df_IA.plot(kind='box', vert=False, ax=ax1, figsize=(20,6))
df_IA.plot(kind='line', ax=ax2, figsize=(20,6))

#Create a box plot to visualize the distribution of the top 15 countries 
#(based on total immigration) grouped by the decades 1980s, 1990s, and 2000s.
df_top15 = df_can.sort_values(by='Total', ascending=False, axis=0).head(15)
dec80=list(map(str, range(1980, 1990)))
dec90=list(map(str, range(1990, 2000)))
dec00=list(map(str, range(2000, 2010)))
df_new = pd.DataFrame(data={"Dec80": df_top15.loc[:, dec80].sum(axis=1),
                            "Dec90": df_top15.loc[:, dec90].sum(axis=1),
                            "Dec00": df_top15.loc[:, dec00].sum(axis=1)})
df_new.describe()
df_new.plot(kind='box', figsize=(10,6))

#let's visualize the trend of total immigrantion to Canada
# (all countries combined) for the years 1980 - 2013.
df_tot=pd.DataFrame(df_can[years].sum(axis=0)) #Total de todos os paises por ano
df_tot.index=list(map(int, df_tot.index)) #Se muda o tipo de str para int para o index que corresponde a lista no nome das colunas ano
df_tot.reset_index(inplace=True) #para ter uma coluna ano para desenhar o grafico, se reseta o index (novo index serie a parte de 0 e antigo index vira coluna)
df_tot.columns=['Ano', 'Total'] #Se passa um label para as colunas ano e total
df_tot.head()
df_tot.plot(kind='scatter', x='Ano', y='Total', figsize=(10,6))

fit = np.polyfit(x=df_tot.Ano, y=df_tot.Total, deg=1) #Calculo da regressao de nivel 1 da serie
plt.plot(df_tot.Ano, fit[0]*df_tot.Ano+fit[1])

#Let's analyze the effect of this crisis, and compare Argentina's immigration to that of it's neighbour Brazil.
#Let's do that using a bubble plot of immigration from Brazil and Argentina for the years 1980 - 2013. 
#We will set the weights for the bubble as the normalized value of the population for each year.
df_ab = df_can.loc[['Argentina', 'Brazil'], years].T #Recuperacao dos valores por ano dos dois paises
df_ab.index = list(map(int, df_ab.index)) #Se passa o index informado com o nome das colunas anos do DF inicial para o tipo int
df_ab.reset_index(inplace=True, drop=False) #Se reseta o index com uma sequencia, deixando o index anterior do numero dos anos para passar como coluna do DF
df_ab.rename(inplace=True, columns={'index':'Ano'}) #Se renomea esta coluna anos de index (label recuperado do antigo index) a Ano

# Brazil
norm_brazil = (df_ab.loc[:,'Brazil']-df_ab.loc[:,'Brazil'].min()) / (df_ab.loc[:,'Brazil'].max()-df_ab.loc[:,'Brazil'].min())
ax0 = df_ab.plot(kind='scatter',
                    x='Ano',
                    y='Brazil',
                    figsize=(14, 8),
                    alpha=0.5,                  # transparency
                    color='green',
                    s=norm_brazil * 2000 + 10,  # pass in weights multiplicando de maneira arbitraria por 2000 e adicionando 10 para aparecer o valor minimo normalizado a 0
                    xlim=(1975, 2015)
                   )

# Argentina
norm_argentina = (df_ab.loc[:,'Argentina']-df_ab.loc[:,'Argentina'].min()) / (df_ab.loc[:,'Argentina'].max()-df_ab.loc[:,'Argentina'].min())
ax1 = df_ab.plot(kind='scatter',
                    x='Ano',
                    y='Argentina',
                    alpha=0.5,
                    color="blue",
                    s=norm_argentina * 2000 + 10,
                    ax = ax0
                   )

ax0.set_ylabel('Number of Immigrants')
ax0.set_title('Immigration from Brazil and Argentina from 1980 - 2013')
ax0.legend(['Brazil', 'Argentina', 'toto'], loc='upper left', fontsize='x-large')


