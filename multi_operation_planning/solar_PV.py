#In this function, capital cost, operating cost, and operating emission of solar PV
#Over the lifespan of district energy system is quantified.
import os
import pandas as pd
import csv
import sys
from pathlib import Path
def solar_pv_calc(A_surf_size,hour_of_day,electricity_demand_max,G_T_now,GT_max, path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    components_path = os.path.join(path_test,'Energy Components')
    solar_component = pd.read_csv(os.path.join(components_path,'solar_PV.csv'))
    lifespan_solar = int(solar_component['Lifespan (year)'][0]) #lifespan of solar PV System
    #lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    ###Solar PV###
    IC_solar = solar_component['Investment cost ($/Wdc)'][0] #Solar PV capital investment cost is 1.75$/Wdc
    OM_solar = solar_component['Fixed solar PV O&M cost ($/kW-year)'][0] #fixed solar PV O&M cost 18$/kW-year
    PD_solar = solar_component['Power density of solar PV system W/m^2'][0] #Module power density of solar PV system W/m^2
    eff_module = solar_component['Module efficiency'][0] #Module efficiency
    eff_inverter = solar_component['Inverter efficiency'][0] #Inverter efficiency
    CAP_solar = PD_solar*A_surf_size/1000
    A_surf_max = electricity_demand_max/(GT_max*eff_module*eff_inverter/1000)
    #salvage_solar = 1-(lifespan_solar-lifespan_project+lifespan_solar*int(lifespan_project/lifespan_solar))/lifespan_solar
    E_solar = A_surf_size*G_T_now*eff_module*eff_inverter/1000 #Solar generation from PV system (kWh) CHANGE G_T
    #invest_solar  = (IC_solar*1000*salvage_solar+OM_solar*UPV_maintenance)*CAP_solar #CAP_solar in kW + investment cost of solar in $
    invest_solar = 1
    return E_solar,invest_solar,A_surf_max
