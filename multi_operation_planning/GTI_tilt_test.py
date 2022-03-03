import multi_operation_planning
from multi_operation_planning.solar_irradiance import aoi, get_total_irradiance
from multi_operation_planning.solar_position import get_solarposition
from pvlib import atmosphere, solarposition, tools
import csv
from csv import writer, reader
import pandas as pd
import datetime
import os
import sys

class GTI_class:
    def __init__(self,_year,_city,path_test,tilt_revised):
        editable_data_path =os.path.join(path_test, 'editable_values.csv')
        editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
        self.lat = float(editable_data['Latitude'])
        self.lon = float(editable_data['Longitude'])
        self.altitude = float(editable_data['Altitude']) #SLC altitude m
        editable_data['solar_tilt'] = tilt_revised
        self.surf_tilt = float(editable_data['solar_tilt']) #panels tilt degree
        print(self.surf_tilt)
        self.surf_azimuth = float(editable_data['solar_azimuth']) #panels azimuth degree
        self.city = _city
        self.year = _year
        self.info_name =self.city+'_'+str(self.lat)+'_'+str(self.lon)+'_psm3_60_'+str(self.year)+'.csv'
        self.weather_data = pd.read_csv(os.path.join(folder_path,self.info_name), header=None)[2:]
    def process_gti(self):
        DNI= self.weather_data[5]
        DHI = self.weather_data[6]
        GHI = self.weather_data[7]
        dti = pd.date_range(str(self.year)+'-01-01', periods=365*24, freq='H')
        solar_position = get_solarposition(dti, self.lat, self.lon, self.altitude, pressure=None, method='nrel_numpy', temperature=12)
        solar_zenith = solar_position['zenith']
        solar_azimuth =  solar_position['azimuth']
        poa_components_vector = []
        poa_global = []
        for i in range(len(solar_zenith)):
            poa_components_vector.append(get_total_irradiance(self.surf_tilt, self.surf_azimuth,
                                     solar_zenith[i], solar_azimuth[i],
                                    float(DNI[3+i]), float(GHI[3+i]), float(DHI[3+i]), dni_extra=None, airmass=None,
                                     albedo=.25, surface_type=None,
                                     model='isotropic',
                                     model_perez='allsitescomposite1990'))
            poa_global.append(poa_components_vector[i]['poa_global'])
        csv_input = pd.read_csv(os.path.join(folder_path,self.info_name), header=None)[2:]
        poa_global.insert(0,'GTI')
        csv_input['ghi'] = poa_global
        csv_input.to_csv(os.path.join(folder_path,self.info_name), index=False)
        return poa_global
def GTI_results(city_DES,path_test,tilt_revised):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = city_DES
    global folder_path
    folder_path = os.path.join(path_test,str(city_DES))
    for year in range(int(editable_data['starting_year']),int(editable_data['ending_year'])+1):
        print('Calculating the global tilted irradiance on a surface in '+editable_data['city']+' in '+str(year))
        weather_year = GTI_class(year,city,path_test,tilt_revised)
        weather_year.process_gti()
