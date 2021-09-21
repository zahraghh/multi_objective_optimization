from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import datetime as dt
import os
import sys
import pandas as pd
import csv
from pathlib import Path
import json
path_test =  os.path.join(sys.path[0])
path_parent= Path(Path(sys.path[0]))
import multi_operation_planning
from multi_operation_planning import Operation_EA_class, uncertainty_analysis_operation
###Decison Variables###
solar_PV_generation = []
wind_turbine_generation = []
components_path = os.path.join(path_test,'Energy Components')
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city = editable_data['city']
renewable_percentage = float(editable_data['renewable percentage'])  #Amount of renewables at the U (100% --> 1,mix of 43% grid--> 0.463, mix of 29% grid--> 0.29, 100% renewables -->0)
representative_day = {}
representative_days_path = os.path.join(path_test,'Scenario Generation',city, 'Operation Representative days')
file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
lbstokg_convert = 0.453592 #1 l b = 0.453592 kg
num_scenarios = int(editable_data['num_scenarios'])

if __name__ == "__main__":
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
    with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json')) as f:
        scenario_generated = json.load(f)
    num_clusters = int(editable_data['Cluster numbers'])+2
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        for represent in range(num_clusters):
            for day in range(num_scenarios):
                E_bat = {}
                df_object = {}
                df_operation = {}
                for hour in range(0,24):
                    data_now = scenario_generated[str(represent)][str(day)][str(hour)]
                    electricity_demand_now =data_now[0] #kWh
                    heating_demand_now = data_now[1] #kWh
                    G_T_now = data_now[2] #Global Tilted Irradiation (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
                    V_wind_now = data_now[3] #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
                    electricity_EF = data_now[4]*renewable_percentage*lbstokg_convert/1000 #kg/kWh
                    #print('rep',represent,'demands',electricity_demand_now,heating_demand_now,G_T_now,V_wind_now,electricity_EF)
                    if hour==0:
                        E_bat[hour]=0
                    problem= Operation_EA_class.Operation_EA(hour,G_T_now,V_wind_now,E_bat[hour], electricity_demand_now,heating_demand_now,electricity_EF)
                    with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
                        algorithm = NSGAII(problem,  population_size = int(editable_data['population_size']), evaluator=evaluator) #5000 iteration
                        algorithm.run(int(editable_data['num_iterations']))
                    E_bat[hour+1] = problem.E_bat_new
                    df_object_hour,df_operation_hour = Operation_EA_class.results_extraction(hour, problem,algorithm,problem.solar_PV_generation,problem.wind_turbine_generation,E_bat[hour])
                    df_object[hour] = df_object_hour.sort_values(by = 'Cost ($)')
                    df_operation[hour] =df_operation_hour.sort_values(by = 'Cost ($)')
                    if hour !=0:
                        df_object[hour] = df_object[hour].add(df_object[hour-1])
                        df_operation[hour] = df_operation[hour].add(df_operation[hour-1])
                file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_EA_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
                results_path = os.path.join(sys.path[0], file_name)
                if not os.path.exists(results_path):
                    os.makedirs(results_path)
                df_object[hour].to_csv(os.path.join(results_path , str(represent)+'_'+str(day)+'_represent_objectives.csv'), index=False)
                df_operation[hour].to_csv(os.path.join(results_path, str(represent)+'_'+str(day)+'_represent_sizing_all.csv'), index=False)
                print(represent,day,df_object[hour])
