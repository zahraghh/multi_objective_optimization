### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
from pathlib import Path
import json
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
from pyomo.opt import SolverFactory
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI,uncertainty_analysis_operation,MILP_two_objective
num_scenarios = int(editable_data['num_scenarios'])
lbstokg_convert = 0.453592 #1 l b = 0.453592 kg
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES,path_test)
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
            json.dump(generated_scenario, fp)
    #Do we need to perfrom the two stage stochastic programming using NSGA-II?
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        #(path_test,solar_index,wind_index,CHP_index, boiler_index,battery_index)
        #solar index is the percentage of using rooftop area, where 1 equals to using the whole available rooftop
        components_path = os.path.join(path_test,'Energy Components')
        editable_data_path =os.path.join(path_test, 'editable_values.csv')
        editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
        city = editable_data['city']
        num_components = 0
        representative_days_path = os.path.join(path_test,'Scenario Generation',city, 'Representative days')
        renewable_percentage = float(editable_data['renewable percentage'])  #Amount of renewables at the U (100% --> 1,mix of 43% grid--> 0.463, mix of 29% grid--> 0.29, 100% renewables -->0)
        ###System Parameters## #
        year = int(editable_data['ending_year'])
        electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE
        num_clusters = int(editable_data['Cluster numbers'])+2
        num_scenarios = int(editable_data['num_scenarios'])
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json')) as f:
            scenario_generated = json.load(f)
        for represent in range(num_clusters):
            for day in range(num_scenarios): #,num_scenarios
                E_bat = {}
                df_object = {}
                df_operation = {}
                for hour in range(0,24):
                    data_now = scenario_generated[str(represent)][str(day)][str(hour)]
                    electricity_demand_now =data_now[0] #kWh
                    heating_demand_now = data_now[1] #kWh
                    G_T_now = data_now[2] #Global Tilted Irradiatio n (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
                    V_wind_now = data_now[3] #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
                    electricity_EF = data_now[4]*renewable_percentage*lbstokg_convert/1000 #kg/kWh
                    if hour==0:
                        E_bat[hour]=0
                    results = MILP_two_objective.Operation(hour, G_T_now,V_wind_now,E_bat[hour], electricity_demand_now, heating_demand_now,electricity_EF)
                    E_bat[hour+1] = results[10]
                    df_object_hour, df_operation_hour = MILP_two_objective.results_extraction(hour, results,E_bat[hour])
                    df_object[hour] = df_object_hour
                    df_operation[hour] =df_operation_hour
                    if hour !=0:
                        df_object[hour] = df_object[hour].add(df_object[hour-1])
                        df_operation[hour] = df_operation[hour].add(df_operation[hour-1])
                file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_MILP_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
                results_path = os.path.join(path_test, file_name)
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                df_object[hour].to_csv(os.path.join(results_path , str(represent)+'_'+str(day)+'_represent_objectives.csv'), index=False)
                df_operation[hour].to_csv(os.path.join(results_path, str(represent)+'_'+str(day)+'_represent_sizing_all.csv'), index=False)
