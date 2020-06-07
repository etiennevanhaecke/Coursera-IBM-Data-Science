# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:48:01 2020

@author: etienne.vanhaecke
"""

import pandas as pd
import numpy as np
df_survey = pd.read_csv('https://cocl.us/datascience_survey_data', index_col=0)
df_survey

import matplotlib as mpl
import matplotlib.pyplot as plt
df_survey.sort_values(by='Very interested', ascending=False, inplace=True)
tot_respond = 2233 
df_survey_perc = np.round((df_survey / tot_respond * 100), 2)

fig, ax = plt.subplots(figsize=(20,8)) #A multipl 2
labels=df_survey_perc.index
x = np.arange(len(labels))  # the label locations
width = 0.30  # the width of the bars a passar a 0.8

rectVI = ax.bar(x-width, df_survey_perc['Very interested'], width, 
                label='Very interested', color='#5cb85c')
rectSI = ax.bar(x, df_survey_perc['Somewhat interested'], width, 
                label='Somewhat interested', color='#5bc0de')
rectNI = ax.bar(x+width, df_survey_perc['Not interested'], width, 
                label='Not interested', color='#d9534f')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_title("Percentage of Respondents' Interest in Data Science Areas", fontsize=20)
#ax.title.set_fontsize=16
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.tick_params(axis='x', which='both', rotation=90)
ax.tick_params(axis='y', which='both', left=False, labelleft=False)
ax.legend(fontsize=14)
# Hide the left, right and top spines
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height)+'%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14)

autolabel(rectVI)
autolabel(rectSI)
autolabel(rectNI)
fig.tight_layout()
plt.show()


df_crime = pd.read_csv('https://cocl.us/sanfran_crime_dataset') #, index_col=0)
df_crime.head()
df_crime.tail()
df_crime.info()

gr_crime=df_crime[['IncidntNum','PdDistrict']].groupby(by='PdDistrict', as_index=False)
gr_crime=gr_crime.count()
gr_crime.rename(columns={'PdDistrict':'Neighborhood', 'IncidntNum':'Count'}, inplace=True)

import folium
# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42
# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
# display the map of San Francisco
sanfran_map

# download countries geojson file
#!wget --quiet https://cocl.us/sanfran_geojson -O san-francisco.json
    
print('GeoJSON file downloaded!')
sf_geo = r'san-francisco.json' # geojson file

# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
threshold_scale = np.linspace(gr_crime['Count'].min(),
                              gr_crime['Count'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum crime rate

# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
sanfran_map.choropleth(
    geo_data=sf_geo,
    data=gr_crime,
    columns=['Neighborhood', 'Count'],
    key_on='feature.properties.DISTRICT',
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime rate in San Francisco',
    #threshold_scale = threshold_scale,
    reset=True
)
# display map
sanfran_map