#In this function, capital cost, operating cost, and operating emission of boiler
#Over the lifespan of district energy system is quantified.
import os
import pandas as pd
import csv
import sys
from pathlib import Path
import solar_PV
import wind_turbine
def battery_calc(electricity_demand_bat,hour,E_bat_,_A_solar,_A_swept,_CAP_battery,G_T_now,V_wind_now, path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    components_path = os.path.join(path_test,'Energy Components')
    battery_component = pd.read_csv(os.path.join(components_path,'battery.csv'))
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    #lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    deltat = 1 #hour for batteries
    A_swept = _A_swept #Swept area of rotor m^2
    A_solar = _A_solar #Solar durface m^2 --> gives 160*A_solar W solar & needs= A_solar/0.7 m^2 rooftop
    CAP_battery= _CAP_battery
    index_battery =  list(battery_component['CAP_battery (kWh)']).index(CAP_battery)
    CAP_battery = battery_component['CAP_battery (kWh)'][index_battery]
    eff_bat_ch = battery_component['Battery efficiency charge'][index_battery]
    eff_bat_disch = battery_component['Battery efficiency discharge'][index_battery]
    bat_dod = battery_component['battery depth of discharge'][index_battery] #battery depth of discharge
    lifespan_battery = battery_component['Lifespan (year)'][index_battery]
    E_bat = E_bat_
    renewables_elect =  solar_PV.solar_pv_calc(A_solar, hour,0,G_T_now,1,path_test)[0] + wind_turbine.wind_turbine_calc(A_swept, hour,0,V_wind_now,1,path_test)[0]
    electricity_demand = electricity_demand_bat
    if renewables_elect>=electricity_demand:
        P_ch_dis_old = renewables_elect - electricity_demand
        electricity_demand = 0
        if P_ch_dis_old>CAP_battery/eff_bat_ch -  E_bat: #Charging the battery
            P_ch_dis_old = CAP_battery/eff_bat_ch -  E_bat
        E_bat_new= E_bat + eff_bat_ch*P_ch_dis_old*deltat
    elif renewables_elect<electricity_demand: #Diccharging the battery
        P_ch_dis_old = electricity_demand - renewables_elect
        if E_bat- (1-bat_dod)*CAP_battery<0:
            E_bat_new= E_bat
            P_ch_dis_old = 0
        elif E_bat- (1-bat_dod)*CAP_battery < 1/eff_bat_disch*P_ch_dis_old*deltat:
            P_ch_dis_old = eff_bat_disch*E_bat - (1-bat_dod)*CAP_battery
        electricity_demand = electricity_demand - P_ch_dis_old - renewables_elect
        E_bat_new = E_bat - 1/eff_bat_disch*P_ch_dis_old*deltat
    IC_battery =  battery_component['Investment cost ($/kW)'][index_battery] #Battery capital investment cost is 2338 $/kW
    OM_battery = battery_component['Fixed O&M cost  $/kW-year'][index_battery]#fixed battery O&M cost 6$/kW-year
    #invest_battery = (IC_battery*lifespan_project/lifespan_battery +OM_battery*UPV_maintenance)*CAP_battery
    invest_battery=1
    return E_bat_new,invest_battery,electricity_demand
