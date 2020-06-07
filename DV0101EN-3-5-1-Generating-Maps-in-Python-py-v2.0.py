# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:35:43 2020

@author: etienne.vanhaecke
"""
import folium

# define the world map
world_map = folium.Map()

# display world map
world_map


# define the world map centered around Canada with a low zoom level
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Terrain')
# display world map
world_map

world_map.save('world_map.html')
# Import the Folium interactive html file
from IPython.display import HTML
HTML('<iframe src=world_map.html width=700 height=450></iframe>')

import webbrowser
webbrowser.open_tab("world_map.html")

# ------------------------------------------------------------------------------------------------
# so let's write a custom temporary-HTML renderer
# pretty much copy-paste of this answer: https://stackoverflow.com/a/38945907/3494126
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer


PORT = 7000
HOST = '127.0.0.1'
SERVER_ADDRESS = '{host}:{port}'.format(host=HOST, port=PORT)
FULL_SERVER_ADDRESS = 'http://' + SERVER_ADDRESS


def TemproraryHttpServer(page_content_type, raw_data):
    """
    A simpe, temprorary http web server on the pure Python 3.
    It has features for processing pages with a XML or HTML content.
    """

    class HTTPServerRequestHandler(BaseHTTPRequestHandler):
        """
        An handler of request for the server, hosting XML-pages.
        """

        def do_GET(self):
            """Handle GET requests"""

            # response from page
            self.send_response(200)

            # set up headers for pages
            content_type = 'text/{0}'.format(page_content_type)
            self.send_header('Content-type', content_type)
            self.end_headers()

            # writing data on a page
            self.wfile.write(bytes(raw_data, encoding='utf'))

            return

    if page_content_type not in ['html', 'xml']:
        raise ValueError('This server can serve only HTML or XML pages.')

    page_content_type = page_content_type

    # kill a process, hosted on a localhost:PORT
    subprocess.call(['fuser', '-k', '{0}/tcp'.format(PORT)])

    # Started creating a temprorary http server.
    httpd = HTTPServer((HOST, PORT), HTTPServerRequestHandler)

    # run a temprorary http server
    httpd.serve_forever()


def run_html_server(html_data=None):

    if html_data is None:
        html_data = """
        <!DOCTYPE html>
        <html>
        <head>
        <title>Page Title</title>
        </head>
        <body>
        <h1>This is a Heading</h1>
        <p>This is a paragraph.</p>
        </body>
        </html>
        """

    # open in a browser URL and see a result
    webbrowser.open(FULL_SERVER_ADDRESS)

    # run server
    TemproraryHttpServer('html', html_data)

# ------------------------------------------------------------------------------------------------


# now let's save the visualization into the temp file and render it
#from tempfile import NamedTemporaryFile
#tmp = NamedTemporaryFile()
#world_map.save(tmp.name)
world_map.save("world_map.html")
with open("world_map.html") as f:
    folium_map_html = f.read()

folium_map_html
run_html_server(folium_map_html)

world_map_html=HTML(world_map._repr_html_())

#=== CODIGO ABAIXO FUNCIONANDO PARA ABRIR A MAPA EM UM BROWSER
filepath = r"C:\Users\etienne.vanhaecke\OneDrive\10 - IBM Data Science\world_map.html"
webbrowser.open('file://' + filepath)

import pandas as pd
df_incidents = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Police_Department_Incidents_-_Previous_Year__2016_.csv')
print('Dataset downloaded and read into a pandas dataframe!')
df_incidents.head()
# get the first 100 crimes in the df_incidents dataframe
limit = 100
df_incidents = df_incidents.iloc[0:limit, :]
# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42
# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
# display the map of San Francisco

def Mostra(regiao_map):
    regiao_map.save("regiao_map.html")
    filepath = r"C:\Users\etienne.vanhaecke\OneDrive\10 - IBM Data Science\regiao_map.html"
    webbrowser.open('file://' + filepath)

Mostra(sanfran_map)

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# loop through the 100 crimes and add each to the incidents feature group
import folium.features
for lat, lng, in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

# add pop-up text to each marker on the map
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)    

# add incidents to map
sanfran_map.add_child(incidents)
Mostra(sanfran_map)

# create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
# loop through the 100 crimes and add each to the map
for lat, lng, label in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.CircleMarker(
        [lat, lng],
        radius=5, # define how big you want the circle markers to be
        color='yellow',
        fill=True,
        popup=label,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(sanfran_map)
Mostra(sanfran_map)

from folium import plugins
# let's start again with a clean copy of the map of San Francisco
sanfran_map = folium.Map(location = [latitude, longitude], zoom_start = 12)
# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(sanfran_map)
# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)
# display map
Mostra(sanfran_map)

df_can = pd.read_excel('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Canada.xlsx',
                     sheet_name='Canada by Citizenship',
                     skiprows=range(20),
                     skipfooter=2)
print('Data downloaded and read into a dataframe!')
# clean up the dataset to remove unnecessary columns (eg. REG) 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
# let's rename the columns so that they make sense
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent','RegName':'Region'}, inplace=True)
# for sake of consistency, let's also make all column labels of type string
df_can.columns = list(map(str, df_can.columns))
# add total column
df_can['Total'] = df_can.sum(axis=1)
# years that we will be using in this lesson - useful for plotting later on
years = list(map(str, range(1980, 2014)))
print ('data dimensions:', df_can.shape)
world_geo = r'world_countries.json' # geojson file
# create a plain world map
world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
# Recuperacao do arquivo JSON com as fronteiras de cada estado com o comando seguinte a 
# lancar do Anaconda PowerShell
#wget "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/world_countries.json" -O world_countries.json
# generate choropleth map using the total immigration of each country to Canada from 1980 to 2013
# create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
import numpy as np
threshold_scale = np.linspace(df_can['Total'].min(),
                              df_can['Total'].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist() # change the numpy array to a list
threshold_scale[-1] = threshold_scale[-1] + 1 # make sure that the last value of the list is greater than the maximum immigration
world_map.choropleth(
    geo_data=world_geo,
    data=df_can,
    columns=['Country', 'Total'],
    key_on='feature.properties.name',
    threshold_scale=threshold_scale,
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Immigration to Canada'
)
# display map
Mostra(world_map)




