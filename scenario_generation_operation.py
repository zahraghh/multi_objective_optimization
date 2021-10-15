import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from scipy import stats
import os
import sys

def scenario_generation_results(path_test,state=None):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    folder_path = os.path.join(sys.path[0],str(city))
    save_path = os.path.join(sys.path[0],'Scenario Generation',city)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #Normal distribution for electricity emissions
    energy_data = pd.read_csv(os.path.join(sys.path[0],'total_energy_demands.csv'))
    energy_demand_scenario = {}
    electricity_scenario = defaultdict(list)
    heating_scenario = defaultdict(list)
    cooling_scenario = defaultdict(list)
    for i in range(8760):
        #Energy demnad uses uniform distribution from AEO 2021 --> 3-point approximation
        ## Energy Demand
        ## range cooling energy = (0.9*i, i)=(0,(i-0.9i)/0.1i) --> low = 0.112702 = (x-0.9i)/0.1i --> x = tick
        ## range heating energy = 0.69*i, i)
        ## range electricity energy = (0.91*i, i)
        electricity_scenario['low'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.112702+0.91))
        heating_scenario['low'].append(energy_data['Heating (kWh)'][i]*(0.1*0.112702+0.69))
        cooling_scenario['low'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.112702+0.90))
        electricity_scenario['medium'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.50+0.91))
        heating_scenario['medium'].append(energy_data['Heating (kWh)'][i]*(0.1*0.50+0.69))
        cooling_scenario['medium'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.50+0.90))
        electricity_scenario['high'].append(energy_data['Electricity (kWh)'][i]*(0.1*0.887298+0.91))
        heating_scenario['high'].append(energy_data['Heating (kWh)'][i]*(0.1*0.887298+0.69))
        cooling_scenario['high'].append(energy_data['Cooling (kWh)'][i]*(0.1*0.887298+0.90))
    range_data = ['low','medium','high']
    i_solar= range_data[1]
    i_wind= range_data[1]
    i_emission= range_data[1]
    scenario_genrated = {}
    scenario_genrated_normalized = {}
    for i_demand in range_data:
        scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission] = {'Total Electricity (kWh)': [x + y for x, y in zip(cooling_scenario[i_demand],electricity_scenario[i_demand])],
                'Heating (kWh)':heating_scenario[i_demand],
                }
        df_scenario_generated=pd.DataFrame(scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission])
        df_scenario_generated.to_csv(os.path.join(save_path,'D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv'), index=False)
