### Performing Multi-objective Optimization of Operation planning of District Energy systems ###
### Using MILP Solver (GLPK) ###
import os
import sys
import pandas as pd
import csv
import json
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI, scenario_generation_operation, uncertainty_analysis_operation,MILP_two_objective,MILP_results_repdays
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES,path_test)
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Scenarios Generation/Reduction']=='yes':
        scenario_generation_operation.scenario_generation_results(path_test)
        generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
            json.dump(generated_scenario, fp)
    #Do we need to perfrom the two stage stochastic programming using MILP solver (GLPK)?
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        MILP_results_repdays.results_repdays(path_test)
