# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:40:37 2020

@author: etienne.vanhaecke
"""


import pandas as pd
import numpy as np
URL='https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=945633050'
#This function searches for <table> elements and only for <tr> and <th> rows and <td> elements
#within each <tr> or <th> element in the table. <td> stands for "table data". 
scrapWiki = pd.read_html(URL, match='Postcode')
print('Data downloaded!')

#Renaming of the column Postal Code
postalCode_df = scrapWiki[0]
postalCode_df=postalCode_df.rename(columns={'Postcode':'PostalCode', 'Neighbourhood':'Neighborhood'})
print(postalCode_df.columns)

#Replace empty string by np.nam to have True with the Pandas function isnull
postalCode_df = postalCode_df.replace('', np.nan)

#Only consideration of the row wwith a borough associated
postalCode_df=postalCode_df[(postalCode_df.Borough!='Not assigned') & (~postalCode_df.Borough.isnull())]
postalCode_df.reset_index(drop=True, inplace=True)

#For the cell without neighborhood assigned, use the borough as neighborhood
for ind in postalCode_df[(postalCode_df.Neighborhood == 'Not assigned') | (postalCode_df.Neighborhood.isnull())].index:
    print('index {}: Neighborhood informed with {}'.format(ind, postalCode_df.loc[ind, "Borough"]))
    postalCode_df.loc[ind, "Neighborhood"] = postalCode_df.loc[ind, "Borough"]

#Consultation of an exenple of postal code, M9V. with more that one neighborhood
postalCode_df[postalCode_df.PostalCode=='M9V']
#Grouping of the neighborhood by the postaCode and borough, separating these by a comma
#postalCode_grp = postalCode_df.groupby(by=['PostalCode', 'Borough'], axis=0)
#postalCode_df['Neighborhood']=postalCode_grp['Neighborhood'].transform(lambda x: ', '.join(x))
postalCode_df=postalCode_df.groupby(by=['PostalCode', 'Borough'])['Neighborhood'].apply(', '.join).reset_index()
#Consultation of an exenple of postal code, M9V. with more that one neighborhood
postalCode_df[postalCode_df.PostalCode=='M9V']


'''Observation: In the current page, there is only one row by postal code, compiling the diferent 
neigborbooh with the same postal code in a same row. The part one of the assigment has been done 
spliting the neighborhood of same postal code in distuncts rows to recover afterward the latitude 
and longitude coordenates of each neighborhood.

To do the exercice with the wikipage corresponding at the description of the exercice, it should 
to acess at the wiki page
"https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=945633050"

URL='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
#This function searches for <table> elements and only for <tr> and <th> rows and <td> elements
#within each <tr> or <th> element in the table. <td> stands for "table data". 
scrapWiki = pd.read_html(URL, match='Postal Code')
print('Data downloaded!')

#Renaming of the column Postal Code
postalCode_df = scrapWiki[0]
postalCode_df=postalCode_df.rename(columns={'Postal Code':'PostalCode'})
print(postalCode_df.columns)

#Replace empty string by np.nam to have True with the Pandas function isnull
postalCode_df = postalCode_df.replace('', np.nan)

#Only consideration of the row wwith a borough associated
postalCode_df=postalCode_df[(postalCode_df.Borough!='Not assigned') & (~postalCode_df.Borough.isnull())]
postalCode_df.reset_index(drop=True, inplace=True)

#For the cell without neighborhood assigned, use the borough as neighborhood
for ind in postalCode_df[(postalCode_df.Neighborhood == 'Not assigned') | (postalCode_df.Neighborhood.isnull())].index:
    print('index {}: Neighborhood informed with {}'.format(ind, postalCode_df.loc[ind, "Borough"]))
    postalCode_df.loc[ind, "Neighborhood"] = postalCode_df.loc[ind, "Borough"]

#Consultation of an exenple of postal code, M5A. with more that one neighborhood
postalCode_df[postalCode_df.PostalCode=='M5A']

#For the code postal associated at more of one neighborhood, creation of one row by neighborhood
dico = {} #Dictionary with the columns of the data frame for one neighborhood
rows_list = [] #List of the of neighborhood dicionary t o insert afterward in the data frame
#First loop on each row in this situation to create a list of neighborhood dicionary
for ind in postalCode_df[postalCode_df.Neighborhood.str.contains(',')].index:
    print('Index '+str(ind)+': '+postalCode_df.Neighborhood[ind])
    for neighborhood in postalCode_df.Neighborhood[ind].split(sep=','):
        dico = {} #Needs to use a new memory to preserve the rows already presente in row_list
        dico.update({'PostalCode':postalCode_df.PostalCode[ind],
                     'Borough':postalCode_df.Borough[ind],
                     'Neighborhood':neighborhood.strip()})
        rows_list.append(dico) #Beaward that it's a shallow copy and it needs to have a new dico variable
                               #whereas all the rows will have the same value of the last row appended
        print('-> Preparation of a new row for the neighborhood '+neighborhood.strip())
#Append of the neighborhood list to the data frame
postalCode_df=postalCode_df.append(rows_list)
#Supression of the initials roew with various neighborhoods (indicated by the presence of the separator ",")
postalCode_df = postalCode_df[~postalCode_df.Neighborhood.str.contains(',')]
#Reset of the index to be continuous
postalCode_df.reset_index(drop=True, inplace=True)
'''

#Shape of the final cleaned data frame
print('Shape of postalCode_df: '+str(postalCode_df.shape))

import geocoder # import geocoder
# initialization of the list which will conserve the coordenates of each postal code of Toronto
coordLatLng_lst = []

#Loop on each postal code of Toronto
for ind in postalCode_df.iloc[0:].index:
    postalCode=postalCode_df.loc[ind, "PostalCode"]
    # initialization
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        #g = geocoder.google('{}, Toronto, Ontario'.format(postalCode))
        g = geocoder.arcgis('{}, Toronto, Ontario'.format(postalCode))
        lat_lng_coords = g.latlng
        
    coordLatLng_lst.append(lat_lng_coords)
    print("index {} - Coordenates of the postal code {}: {}".format(ind, postalCode, lat_lng_coords))

#Adition of thes coordenates at the data frame with the new columns Latitude and Longitude
postalCode_df["Latitude"] = np.array(coordLatLng_lst)[:, 0]
postalCode_df["Longitude"] = np.array(coordLatLng_lst)[:, 1]
#Observation: The list is passed as array because the slicing in nested list is not working
#             to recover the individual latitude ans longitude of each postal code coordenates.

#Map of Toronto City
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import folium # map rendering library


address = 'Toronto, Ontario'
geolocator = Nominatim(user_agent="tr_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto City are {}, {}.'.format(latitude, longitude))
# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
