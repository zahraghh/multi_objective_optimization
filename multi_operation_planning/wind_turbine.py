#In this function, capital cost, operating cost, and operating emission of boiler
#Over the lifespan of district energy system is quantified.
import os
import pandas as pd
import csv
import sys
from pathlib import Path
def wind_turbine_calc(A_swept_size,hour_of_day,electricity_demand_max,V_wind_now,V_max,path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    components_path = os.path.join(path_test,'Energy Components')
    wind_component = pd.read_csv(os.path.join(components_path,'wind_turbine.csv'))
    cut_in_wind_speed = wind_component['Cut-in Speed'][0] #2.5 m/s is the minimum wind speed to run the wind turbines
    cut_out_wind_speed= wind_component['Cut-out Speed'][0]
    lifespan_wind = int(wind_component['Lifespan (year)'][0]) #lifespan of wind turbines
    lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    ###Wind Turbine###
    index_wind = list(wind_component['Swept Area m^2']).index(A_swept_size)
    CAP_wind = wind_component['Rated Power kW'][index_wind]
    IC_wind = wind_component['Investment Cost'][index_wind] #Wind turbine capital cost in Utah 2018 1740$/kW
    rho = 1.2 #air density for wind turbines kg/m^3 CHANGE
    OM_wind = 44 #fixed wind turbines O&M cost 44$/kW-year
    C_p = 0.35 #Power coefficient default value of 0.35 in E+ CHANGE
    if V_wind_now<cut_in_wind_speed:
        V_wind_now = 0
    if V_wind_now>cut_out_wind_speed:
        V_wind_now = 0
    E_wind = 0.5*C_p*rho*A_swept_size*V_wind_now**3/1000 #Wind generation from wind Turbine (kW) CHANGE V_wind

    #salvage_wind = 1-(lifespan_wind-lifespan_project+lifespan_wind*int(lifespan_project/lifespan_wind))/lifespan_wind
    #invest_wind = (IC_wind + OM_wind*UPV_maintenance)*CAP_wind #CAP_wind in kW + investment cost of wind in $
    invest_wind = 1
    return E_wind, invest_wind
