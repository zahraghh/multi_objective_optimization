### Performing Multi-objective Optimization of Operation planning of District Energy systems ###
### Using NSGA-II Algorithm ###
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
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI, scenario_generation_operation, uncertainty_analysis_operation,NSGA_two_objectives, uncertainty_analysis_operation
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
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
    if editable_data['Generate Scenarios']=='yes':
        scenario_generation_operation.scenario_generation_results(path_test)
        generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
            json.dump(generated_scenario, fp)
    #Do we need to perfrom the multi-objective optimization of operation planning using NSGA-II?
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        NSGA_two_objectives.NSGA_Operation(path_test)
