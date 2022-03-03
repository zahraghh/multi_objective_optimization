import os
import sys
import pandas as pd
import csv
from pathlib import Path
import json
from collections import defaultdict
import multi_operation_planning
from multi_operation_planning import clustring_kmediod_PCA_operation, MILP_two_objective
lbstokg_convert = 0.453592 #1 lb = 0.453592 kg
def results_repdays(path_test):
    components_path = os.path.join(path_test,'Energy Components')
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    num_components = 0
    representative_days_path = os.path.join(path_test,'Scenario Generation',city, 'Operation Representative days')
    renewable_percentage = float(editable_data['renewable percentage'])  #Amount of renewables at the U (100% --> 1,mix of 43% grid--> 0.463, mix of 29% grid--> 0.29, 100% renewables -->0)

    ###System Parameters###
    city = editable_data['city']
    year = int(editable_data['ending_year'])
    lat = float(editable_data['Latitude'])
    lon = float(editable_data['Longitude'])
    city_EF = int(editable_data['city EF'])
    electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE
    num_clusters = int(editable_data['Cluster numbers'])+2
    num_scenarios = int(editable_data['num_scenarios'])

    folder_path = os.path.join(path_test,str(city))
    weather_data = pd.read_csv(os.path.join(folder_path,city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)+'.csv'), header=None)[2:]
    solar_data = weather_data[46]
    wind_data = weather_data[8]
    solar_day =  defaultdict(lambda: defaultdict(list))
    wind_day =  defaultdict(lambda: defaultdict(list))
    data_all_labels,represent_day = clustring_kmediod_PCA_operation.kmedoid_clusters(path_test)
    for key in represent_day:
        day = represent_day[key]
        hour = 0
        for index_in_year in range(2+(day+1)*24,2+(day+2)*24):
            solar_day[key][hour].append(float(solar_data[index_in_year]))
            wind_day[key][hour].append(float(wind_data[index_in_year]))
            hour +=1
    hour =0
    representative_day = {}
    weight_representative_day = {}
    for represent in range(num_clusters):
        representative_day[represent] = pd.read_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(represent)+'.csv'))
        weight_representative_day[represent] = representative_day[represent]['Percent %'][0]/100*365
        E_bat = {}
        df_object = {}
        df_operation = {}
        for hour in range(0,24):
            electricity_demand_now = round(representative_day[represent]['Electricity total (kWh)'][hour],5) #kWh
            heating_demand_now = round(representative_day[represent]['Heating (kWh)'][hour],5) #kWh
            G_T_now = round(solar_day[represent][hour][0],5) #Global Tilted Irradiation (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
            V_wind_now = round(wind_day[represent][hour][0],5) #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
            electricity_EF = round(city_EF*renewable_percentage*lbstokg_convert/1000,5) #kg/kWh
            if hour==0:
                E_bat[hour]=0
            results = MILP_two_objective.Operation(hour, G_T_now,V_wind_now,E_bat[hour], electricity_demand_now, heating_demand_now,electricity_EF)
            E_bat[hour+1] = results[10]
            df_object_hour, df_operation_hour = MILP_two_objective.results_extraction(hour, results,E_bat[hour])
            df_object[hour] = df_object_hour
            df_operation[hour] =df_operation_hour
            #print(hour, df_operation_hour['CHP Operation (kWh)'])
            #print('cost',df_operation_hour['Cost ($)'])
            if hour !=0:
                df_object[hour] = df_object[hour].add(df_object[hour-1])
                df_operation[hour] = df_operation[hour].add(df_operation[hour-1])

        file_name = city+'_Deter_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_MILP_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
        results_path = os.path.join(path_test, file_name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        df_object[hour].to_csv(os.path.join(results_path , str(represent)+'_'+'_represent_objectives.csv'), index=False)
        df_operation[hour].to_csv(os.path.join(results_path, str(represent)+'_'+'_represent_sizing_all.csv'), index=False)
