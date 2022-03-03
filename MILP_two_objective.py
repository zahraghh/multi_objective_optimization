from pyomo.opt import SolverFactory
import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math
import datetime as dt
from collections import defaultdict
import os
import sys
from pathlib import Path
import json
import multi_operation_planning
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE

### Energy Components ###
def CHP(CAP_CHP_elect_size,F_CHP_size):
    BTUtokWh_convert = 0.000293071 # 1BTU = 0.000293071 kWh
    mmBTutoBTU_convert = 10**6
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    #UPV_NG = float(editable_data['UPV_NG']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
    #UPV_elect = float(editable_data['UPV_elect']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
    NG_prices = float(editable_data['price_NG'])/293.001 #Natural gas price at UoU $/kWh
    lifespan_chp = int(CHP_component['Lifespan (year)'][0])
    ###CHP system###
    if CAP_CHP_elect_size==0:
        return 0,0,0,0,0,0,0
    else:
        index_CHP = list(CHP_component['CAP_CHP_elect_size']).index(CAP_CHP_elect_size)
        IC_CHP = CHP_component['IC_CHP'][index_CHP] #investment cost for CHP system $/kW
        CAP_CHP_Q= CHP_component['CAP_CHP_Q'][index_CHP] #Natural gas input mmBTU/hr, HHV
        eff_CHP_therm = CHP_component['eff_CHP_therm'][index_CHP] #Thermal efficiency of CHP system Q/F
        eff_CHP_elect = CHP_component['eff_CHP_elect'][index_CHP] #Electricity efficiency of CHP system P/F
        OM_CHP =CHP_component['OM_CHP'][index_CHP] #cent/kWh to $/kWh. now is $/kWh
        gamma_CHP =CHP_component['gamma_CHP'][index_CHP]*lbstokg_convert #emissions lb/kWh to kg/kWh
        E_CHP = F_CHP_size*eff_CHP_elect/100 #Electricty generation of CHP system kWh
        Q_CHP = F_CHP_size*eff_CHP_therm/100 #Heat generation of CHP system kWh
        invest_CHP = IC_CHP*CAP_CHP_elect_size #Investment cost of the CHP system $
        OPC_CHP =NG_prices*F_CHP_size + OM_CHP*E_CHP#O&M cost of CHP system $
        OPE_CHP = gamma_CHP*E_CHP # O&M emission of CHP system kg CO2
        return E_CHP,Q_CHP,invest_CHP,OPC_CHP,OPE_CHP,CAP_CHP_elect_size/eff_CHP_elect*100,CAP_CHP_elect_size*eff_CHP_therm/eff_CHP_elect,eff_CHP_therm

def NG_boiler(F_boilers,_CAP_boiler):
    BTUtokWh_convert = 0.000293071 # 1BTU = 0.000293071 kWh
    mmBTutoBTU_convert = 10**6
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    #UPV_NG = float(editable_data['UPV_NG']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
    NG_prices = float(editable_data['price_NG'])/293.001 #Natural gas price at UoU $/kWh
    CAP_boiler = _CAP_boiler
    index_boiler = list(boiler_component['CAP_boiler (kW)']).index(CAP_boiler)
    CAP_boiler = boiler_component['CAP_boiler (kW)'][index_boiler]
    ###Natural gas boiler###
    IC_boiler = float(boiler_component['Investment cost $/MBtu/hour'][index_boiler])/1000/BTUtokWh_convert #Natural gas boiler estimated capital cost of $35/MBtu/hour. now unit is $/kW
    landa_boiler = float(boiler_component['Variabel O&M cost ($/mmBTU)'][index_boiler])/mmBTutoBTU_convert/BTUtokWh_convert #O&M cost of input 0.95 $/mmBTU. now unit is 119 $/kWh
    gamma_boiler = float(boiler_component['Natural gas emission factor (kg-CO2/mmBTU)'][index_boiler])/mmBTutoBTU_convert/BTUtokWh_convert #kg-CO2/kWh is emission factor of natural gas
    eff_boiler = float(boiler_component['Boiler Efficiency'][index_boiler]) #efficiency of natural gas boiler
    Q_boiler = eff_boiler*F_boilers #Net heat generation of NG boilers kWh
    #salvage_boiler = 1-(lifespan_boiler-lifespan_project+lifespan_boiler*int(lifespan_project/lifespan_boiler))/lifespan_boiler
    invest_boiler = IC_boiler*CAP_boiler  #Investment cost of boiler in $
    OPC_boiler = (landa_boiler+NG_prices)*F_boilers #O&M cost of boiler $
    OPE_boiler = gamma_boiler*F_boilers #O&M emissions of boiler in kg CO2
    return Q_boiler,invest_boiler,OPC_boiler,OPE_boiler,eff_boiler
def wind_turbine_calc(A_swept_size,hour_of_day,electricity_demand_max,V_wind_now,V_max):
    cut_in_wind_speed = wind_component['Cut-in Speed'][0] #2.5 m/s is the minimum wind speed to run the wind turbines
    lifespan_wind = int(wind_component['Lifespan (year)'][0]) #lifespan of wind turbines
    #lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    ###Wind Turbine###
    index_wind = list(wind_component['Swept Area m^2']).index(A_swept_size)
    cut_in_wind_speed = wind_component['Cut-in Speed'][index_wind] #2.5 m/s is the minimum wind speed to run the wind turbines
    cut_out_wind_speed= wind_component['Cut-out Speed'][index_wind]
    rated_wind_speed = wind_component['Rated Speed'][index_wind]
    rated_power = wind_component['Rated Power kW'][index_wind]
    CAP_wind = wind_component['Rated Power kW'][index_wind]
    IC_wind = wind_component['Investment Cost'][index_wind] #Wind turbine capital cost in Utah 2018 1740$/kW
    #rho = 1.2 #air density for wind turbines kg/m^3 CHANGE
    OM_wind = 44 #fixed wind turbines O&M cost 44$/kW-year
    #C_p = 0.35 #Power coefficient default value of 0.35 in E+ CHANGE
    cut_out_wind_speed= wind_component['Cut-out Speed'][0]
    if V_wind_now<cut_in_wind_speed or  V_wind_now>cut_out_wind_speed:
        V_wind_now = 0
        E_wind = 0
    elif V_wind_now<cut_out_wind_speed and V_wind_now>rated_wind_speed:
        E_wind = rated_power
    else:
        E_wind = rated_power*((V_wind_now-cut_in_wind_speed)/(rated_wind_speed-cut_in_wind_speed))**3

    #E_wind = 0.5*C_p*rho*A_swept_size*V_wind_now**3/1000 #Wind generation from wind Turbine (kW) CHANGE V_wind
    #salvage_wind = 1-(lifespan_wind-lifespan_project+lifespan_wind*int(lifespan_project/lifespan_wind))/lifespan_wind
    #invest_wind = (IC_wind + OM_wind*UPV_maintenance)*CAP_wind #CAP_wind in kW + investment cost of wind in $
    invest_wind = 1
    #print('wind',CAP_wind,C_p,rho,A_swept_size,IC_wind,OM_wind)
    return E_wind, invest_wind
def solar_pv_calc(A_surf_size,hour_of_day,electricity_demand_max,G_T_now,GT_max):
    lifespan_solar = int(solar_component['Lifespan (year)'][0]) #lifespan of solar PV System
    #lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
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
    if E_solar>CAP_solar:
        E_solar=CAP_solar
    #invest_solar  = (IC_solar*1000*salvage_solar+OM_solar*UPV_maintenance)*CAP_solar #CAP_solar in kW + investment cost of solar in $
    invest_solar=1
    #print('solar',IC_solar,OM_solar,eff_module,eff_inverter,A_surf_size)
    return E_solar,invest_solar,A_surf_max
def battery_calc(electricity_demand_bat,hour,E_bat_,_CAP_battery,G_T_now,V_wind_now):
    #UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    #lifespan_project = float(editable_data['lifespan_project']) #life span of DES
    deltat = 1 #hour for batteries
    CAP_battery = _CAP_battery
    index_battery =  list(battery_component['CAP_battery (kWh)']).index(CAP_battery)
    eff_bat_ch = battery_component['Battery efficiency charge'][index_battery]
    eff_bat_disch = battery_component['Battery efficiency discharge'][index_battery]
    bat_dod = battery_component['battery depth of discharge'][index_battery] #battery depth of discharge
    lifespan_battery = battery_component['Lifespan (year)'][index_battery]
    E_bat = E_bat_
    renewables_elect =  solar_pv_calc(A_solar, hour,0,G_T_now,1)[0] + wind_turbine_calc(A_swept, hour,0,V_wind_now,1)[0]
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
    invest_battery = 1
    #print('battery',_CAP_battery,IC_battery,OM_battery,eff_bat_disch,eff_bat_ch,bat_dod)
    return E_bat_new,invest_battery,electricity_demand
###Decison Variables Stage 2###
num_components = 0
energy_component_type = 1
energy_component_number = {}
solar_PV_generation = []
wind_turbine_generation = []
components_path = os.path.join(path_test,'Energy Components')
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
city = editable_data['city']
use_solar_PV = editable_data['Solar_PV']
use_wind_turbine = editable_data['Wind_turbines']
use_battery = editable_data['Battery']
use_grid = editable_data['Grid']
use_CHP = editable_data['CHP']
use_boilers = editable_data['Boiler']
if use_boilers=='yes':
    num_components +=1
    boiler_component = pd.read_csv(os.path.join(components_path,'boilers.csv'))
    energy_component_number['boilers']=num_components
    energy_component_type +=1
    CAP_boiler= boiler_component['CAP_boiler (kW)'][int(editable_data['boiler_index'])]
else:
    CAP_boiler = 0
if use_CHP=='yes':
    num_components +=1
    CHP_component = pd.read_csv(os.path.join(components_path,'CHP.csv'))
    energy_component_number['CHP']=num_components
    energy_component_type +=1
    CAP_CHP_elect= CHP_component['CAP_CHP_elect_size'][int(editable_data['CHP_index'])]
else:
    CAP_CHP_elect = 0
if use_solar_PV=='yes':
    energy_component_number['solar_PV']=num_components
    solar_component = pd.read_csv(os.path.join(components_path,'solar_PV.csv'))
    num_components +=1
    PV_module = float(editable_data['PV_module']) #area of each commercial PV moduel is 1.7 M^2
    roof_top_area = float(editable_data['roof_top_area']) #60% percentage of the rooftop area of all buildings https://www.nrel.gov/docs/fy16osti/65298.pdf
    A_solar = roof_top_area*int(editable_data['solar_index'])
else:
    A_solar = 0
if use_wind_turbine=='yes':
    num_components +=1
    wind_component = pd.read_csv(os.path.join(components_path,'wind_turbine.csv'))
    energy_component_number['wind_turbine']=num_components
    A_swept= wind_component['Swept Area m^2'][int(editable_data['wind_index'])]
else:
    A_swept = 0
if use_battery=='yes':
    num_components +=1
    battery_component = pd.read_csv(os.path.join(components_path,'battery.csv'))
    energy_component_number['battery']=num_components
    CAP_battery= battery_component['CAP_battery (kWh)'][int(editable_data['battery_index'])]
else:
    CAP_battery = 0
if use_grid== 'yes':
    CAP_grid =1  #means we can use the grid the optimization
else:
    CAP_grid = 0 #means we cannot use the grid the optimization
def Operation(hour, G_T_now,V_wind_now,E_bat_now, electricity_demand_now, heating_demand,electricity_EF):
    solar_PV_generation= round(solar_pv_calc(A_solar, hour,0,G_T_now,1)[0],5)
    wind_turbine_generation = round(wind_turbine_calc(A_swept, hour,0,V_wind_now,1)[0],5)
    battery_results = battery_calc(electricity_demand_now,hour,E_bat_now,CAP_battery,G_T_now,V_wind_now)
    electricity_demand_new = battery_results[2]
    E_bat_new = round(battery_results[0],5)
    energy_component_number = {}
    energy_component_type = 0
    model = pyo.ConcreteModel()
    k=0
    j=0
    i=0
    if  use_CHP=='yes' and CAP_CHP_elect!=0:
        energy_component_number['CHP']=energy_component_type
        model.F_CHP = pyo.Var(bounds=(0,CHP(CAP_CHP_elect,0)[5])) #Decision space for CHP fuel rate
        energy_component_type +=1
        F_F_CHP = 1
        CHP_model = model.F_CHP
        k=1
        #if CHP(CAP_CHP_elect,CHP(CAP_CHP_elect,0)[5]/2)[1]>heating_demand:
        #    F_F_CHP = 0
        #    CHP_model = 0
        #    k=0
    else:
        F_F_CHP = 0
        CHP_model = 0
        k=0
    if  use_boilers=='yes' and CAP_boiler!=0 and heating_demand>0:
        energy_component_number['boilers']=energy_component_type
        energy_component_type +=1
        F_F_boiler = 1
        Boiler_model = heating_demand - CHP(CAP_CHP_elect,CHP_model)[1]
        j=1
    else:
        F_F_boiler =  0
        Boiler_model = 0
        CHP_model = heating_demand/CHP(CAP_CHP_elect,0)[7]
        j=0
    if  use_grid=='yes' and electricity_demand_new>0:
        energy_component_number['grid']=energy_component_type
        energy_component_type +=1
        F_E_grid = 1
        grid_model = electricity_demand_new - CHP(CAP_CHP_elect,CHP_model)[0]
        i=1
    else:
        F_E_grid = 0
        grid_model = 0
        i=0
    if (i*j*k==0):
        if k==0:
            if heating_demand>0:
                Boiler_model = heating_demand
            else:
                Boiler_model = 0
            if electricity_demand_new>0:
                grid_model = electricity_demand_new
            else:
                grid_model = 0
        if i==0:
            CHP_model = 0
            Boiler_model = heating_demand
        CHP_model = round(CHP_model,5)
        Boiler_model = round(Boiler_model,5)
        grid_model = round(grid_model,5)
        cost_objective = round(CHP(CAP_CHP_elect,CHP_model)[3]*F_F_CHP +NG_boiler(Boiler_model,CAP_boiler)[2]*F_F_CHP + grid_model*electricity_prices*F_E_grid,5)
        emissions_objective = round(CHP(CAP_CHP_elect,CHP_model*F_F_CHP)[4] + NG_boiler(Boiler_model*F_F_boiler,CAP_boiler)[3] +grid_model*electricity_EF*F_E_grid,5)
        population_size = int(editable_data['population_size'])
        #print('CHP model',i,j,k,CHP_model,cost_objective)
        return 'cost',[cost_objective]*population_size,'emisisons',[emissions_objective]*population_size,'CHP',[CHP_model]*population_size,'Boilers',[Boiler_model]*population_size,'Grid',[grid_model]*population_size,E_bat_new,solar_PV_generation,wind_turbine_generation


    model.Constraint_elect = pyo.Constraint(expr = grid_model>=0) # Electricity balance of demand and supply sides
    model.Constraint_heat = pyo.Constraint(expr = Boiler_model>=0) # Heating balance of demand and supply sides
    model.f1_cost = pyo.Var()
    model.f2_emissions = pyo.Var()
    model.C_f1_cost = pyo.Constraint(expr= model.f1_cost == 100000*(CHP(CAP_CHP_elect,CHP_model)[3]*F_F_CHP +NG_boiler(Boiler_model,CAP_boiler)[2]*F_F_CHP + grid_model*electricity_prices*F_E_grid))
    model.C_f2_emissions = pyo.Constraint(expr= model.f2_emissions == 100000*(CHP(CAP_CHP_elect,CHP_model*F_F_CHP)[4] + NG_boiler(Boiler_model*F_F_boiler,CAP_boiler)[3] +grid_model*electricity_EF*F_E_grid))
    model.O_f1_cost = pyo.Objective(expr= model.f1_cost)
    model.O_f2_emissions = pyo.Objective(expr= model.f2_emissions)
    model.O_f2_emissions.deactivate()
    opt = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand - CHP(CAP_CHP_elect,value_CHP_model)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_CHP_model = 0
        value_grid_model = electricity_demand_new - CHP(CAP_CHP_elect,value_CHP_model)[0]
    else:
        value_grid_model = 0
    #print(hour,value_CHP_model, CAP_CHP_elect,CAP_boiler,NG_boiler(CAP_boiler/NG_boiler(0,CAP_boiler)[4],CAP_boiler)[0]*F_F_boiler + CHP(CAP_CHP_elect,CHP(CAP_CHP_elect,0)[5])[1]*F_F_CHP, heating_demand,electricity_demand, len(results))
    f2_max = pyo.value(model.f2_emissions)
    f1_min = pyo.value(model.f1_cost)
    ### min f2
    model.O_f2_emissions.activate()
    model.O_f1_cost.deactivate()
    solver = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand - CHP(CAP_CHP_elect,value_CHP_model)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_grid_model = electricity_demand_new - CHP(CAP_CHP_elect,value_CHP_model)[0]
    else:
        value_grid_model = 0
    #print( '( CHP , Boiler, Grid ) = ( ' + str(value_CHP_model) + ' , ' + str(value_Boiler_model) +  ' , ' + str(value_grid_model) +' )')
    #print( 'f1_cost = ' + str(pyo.value(model.f1_cost)) )
    #print( 'f2_emissions = ' + str(pyo.value(model.f2_emissions)) )
    f2_min = pyo.value(model.f2_emissions)
    f1_max = pyo.value(model.f1_cost)

    #print('f2_min',f2_min)

    ### apply normal $\epsilon$-Constraint
    model.O_f1_cost.activate()
    model.O_f2_emissions.deactivate()
    model.e = pyo.Param(initialize=0, mutable=True)
    model.C_epsilon = pyo.Constraint(expr = model.f2_emissions == model.e)
    solver = SolverFactory('glpk')
    results =opt.solve(model,load_solutions=False)
    model.solutions.load_from(results)
    if use_CHP=='yes':
        value_CHP_model=model.F_CHP.value
        if CAP_CHP_elect==0:
            value_CHP_model=0
    else:
        value_CHP_model = 0
    if use_boilers == 'yes':
        value_Boiler_model = heating_demand- CHP(CAP_CHP_elect,value_CHP_model)[1]
        if CAP_boiler==0:
            value_Boiler_model=0
    else:
        value_Boiler_model = 0
    if use_grid =='yes':
        value_grid_model = electricity_demand_new - CHP(CAP_CHP_elect,value_CHP_model)[0]
    else:
        value_grid_model = 0

    #print('emissions range',str(f2_min)+', ' + str(f2_max))
    #print('cost range',str(f1_min)+', ' + str(f1_max))
    n = int(editable_data['population_size'])-2
    step = int((f2_max - f2_min) / n)
    #print('eval',f2_max,f2_min,step)
    if step==0:
        CHP_EC = [round(value_CHP_model,5)]*(n+2)
        Boiler_EC = [round(value_Boiler_model,5)]*(n+2)
        grid_EC = [round(value_grid_model,5)]*(n+2)
        cost_objective_single = round((CHP(CAP_CHP_elect,value_CHP_model)[3]*F_F_CHP +NG_boiler(value_Boiler_model,CAP_boiler)[2]*F_F_CHP + value_grid_model*electricity_prices*F_E_grid),5)
        emissions_objective_single =  round(CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP)[4] + NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler)[3] +value_grid_model*electricity_EF*F_E_grid,5)
        cost_objective = [cost_objective_single]*(n+2)
        emissions_objective = [emissions_objective_single]*(n+2)
    else:
        steps = list(range(int(f2_min),int(f2_max),step)) + [f2_max]

        CHP_EC = []
        Boiler_EC = []
        grid_EC = []
        cost_objective = []
        emissions_objective = []
        for i in steps:
            model.e = i
            solver = SolverFactory('glpk')
            results =opt.solve(model,load_solutions=False)
            model.solutions.load_from(results)
            if use_CHP=='yes':
                value_CHP_model=round(model.F_CHP.value,5)
                if CAP_CHP_elect==0:
                    value_CHP_model=0
            else:
                value_CHP_model = 0
            if use_boilers == 'yes':
                value_Boiler_model = round(heating_demand - CHP(CAP_CHP_elect,value_CHP_model)[1],5)
                if CAP_boiler==0:
                    value_Boiler_model=0
            else:
                value_Boiler_model = 0
            if use_grid =='yes':
                value_grid_model = round(electricity_demand_new - CHP(CAP_CHP_elect,value_CHP_model)[0],5)
            else:
                value_grid_model = 0
            value_CHP_model = round(value_CHP_model,5)
            value_Boiler_model = round(value_Boiler_model,5)
            value_grid_model = round(value_grid_model,5)
            CHP_EC.append(value_CHP_model)
            Boiler_EC.append(value_Boiler_model)
            grid_EC.append(value_grid_model)
            cost_objective.append(round((CHP(CAP_CHP_elect,value_CHP_model)[3]*F_F_CHP +NG_boiler(value_Boiler_model,CAP_boiler)[2]*F_F_CHP + value_grid_model*electricity_prices*F_E_grid),5))
            emissions_objective.append(round(CHP(CAP_CHP_elect,value_CHP_model*F_F_CHP)[4] + NG_boiler(value_Boiler_model*F_F_boiler,CAP_boiler)[3] +value_grid_model*electricity_EF*F_E_grid,5))
        #print('normal $\epsilon$-Constraint')
        #print(Boiler_EC)
        #print(grid_EC)
        #print('cost',cost_objective)
        #print('emisisons',emissions_objective)
    return 'cost',cost_objective,'emisisons',emissions_objective,'CHP',CHP_EC,'Boilers',Boiler_EC,'Grid',grid_EC,E_bat_new,solar_PV_generation,wind_turbine_generation
def results_extraction(hour, results,E_bat):
    components_path = os.path.join(path_test,'Energy Components')
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    use_solar_PV = editable_data['Solar_PV']
    use_wind_turbine = editable_data['Wind_turbines']
    use_battery = editable_data['Battery']
    use_grid = editable_data['Grid']
    use_CHP = editable_data['CHP']
    use_boilers = editable_data['Boiler']
    num_components=0
    if use_boilers=='yes':
        num_components +=1
        boiler_component = pd.read_csv(os.path.join(components_path,'boilers.csv'))
    if use_CHP=='yes':
        num_components +=1
        CHP_component = pd.read_csv(os.path.join(components_path,'CHP.csv'))
    if use_solar_PV=='yes':
        num_components +=1
        PV_module = float(editable_data['PV_module']) #area of each commercial PV moduel is 1.7 M^2
        roof_top_area = float(editable_data['roof_top_area']) #60% percentage of the rooftop area of all buildings https://www.nrel.gov/docs/fy16osti/65298.pdf
    if use_wind_turbine=='yes':
        num_components +=1
        wind_component = pd.read_csv(os.path.join(components_path,'wind_turbine.csv'))
    if use_battery=='yes':
        num_components +=1
        battery_component = pd.read_csv(os.path.join(components_path,'battery.csv'))
    city = editable_data['city']
    file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_MILP_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
    results_path = os.path.join(sys.path[0], file_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    i = 0
    df_operation = {}
    df_cost ={}
    df_object = {}
    df_object_all  = pd.DataFrame(columns = ['Cost ($)','Emission (kg CO2)'])
    df_operation_all = pd.DataFrame(columns = ['CHP Operation (kWh)','Boilers Operation (kWh)','Battery Operation (kWh)','Grid Operation (kWh)','Cost ($)','Emission (kg CO2)'])
    i=0
    solar_results = {}
    wind_results = {}
    CHP_results = {}
    boiler_results = {}
    battery_results = {}
    grid_results = {}
    data_object = {'Cost ($)':results[1],
    'Emission (kg CO2)':results[3]}
    data_operation={'Solar Generation (kWh)':round(results[11],5),
    'Wind Generation (kWh)':round(results[12],5),
    'CHP Operation (kWh)':results[5],
    'Boilers Operation (kWh)':results[7],
    'Battery Operation (kWh)':round(E_bat,5),
    'Grid Operation (kWh)':results[9],
    'Cost ($)':results[1],
    'Emission (kg CO2)':results[3]}
    return pd.DataFrame(data_object),pd.DataFrame(data_operation)
lbstokg_convert = 0.453592 #1 l b = 0.453592 kg
def results_repdays(path_test):
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
    min_electricity = 0
    max_electricity = int(editable_data['max_electricity'])
    min_heating = 0
    max_heating = int(editable_data['max_heating'])
    with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json')) as f:
        scenario_generated = json.load(f)
    key_function = 'yes'
    if key_function=='yes':
        for day in range(num_scenarios): #,num_scenarios
            for represent in range(num_clusters):
                E_bat = {}
                df_object = {}
                df_operation = {}
                gti_list = []
                for hour in range(0,24):
                    data_now = scenario_generated[str(represent)][str(day)][str(hour)]
                    electricity_demand_now =data_now[0] #kWh
                    heating_demand_now = data_now[1] #kWh
                    G_T_now = data_now[2] #Global Tilted Irradiatio n (Wh/m^2) in Slat Lake City from TMY3 file for 8760 hr a year on a titled surface 35 deg
                    gti_list.append(G_T_now)
                    V_wind_now = data_now[3] #Wind Speed m/s in Slat Lake City from AMY file for 8760 hr in 2019
                plt.plot(gti_list)

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
                if heating_demand_now<min_heating:
                    heating_demand_now = min_heating
                if electricity_demand_now<min_electricity:
                    electricity_demand_now=min_electricity
                if  G_T_now<0:
                    G_T_now=0
                if  V_wind_now<0:
                    V_wind_now=0
                #if heating_demand_now>max_heating:
                #    heating_demand_now = max_heating
                #if electricity_demand_now>max_electricity:
                #    electricity_demand_now=max_electricity
                results = Operation(hour, G_T_now,V_wind_now,E_bat[hour], electricity_demand_now, heating_demand_now,electricity_EF)
                E_bat[hour+1] = results[10]
                df_object_hour, df_operation_hour = results_extraction(hour, results,E_bat[hour])
                df_object[hour] = df_object_hour
                df_operation[hour] =df_operation_hour
                if hour !=0:
                    df_object[hour] = df_object[hour].add(df_object[hour-1])
                    df_operation[hour] = df_operation[hour].add(df_operation[hour-1])

            file_name = city+'_EF_'+str(float(editable_data['renewable percentage']) )+'_operation_MILP_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
            results_path = os.path.join(path_test, file_name)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            #print(str(represent)+'_'+str(day),df_object[hour])
            df_object[hour].to_csv(os.path.join(results_path , str(represent)+'_'+str(day)+'_represent_objectives.csv'), index=False)
            df_operation[hour].to_csv(os.path.join(results_path, str(represent)+'_'+str(day)+'_represent_sizing_all.csv'), index=False)
