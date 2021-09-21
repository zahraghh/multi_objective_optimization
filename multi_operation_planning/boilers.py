#In this function, capital cost, operating cost, and operating emission of boiler
#Over the lifespan of district energy system is quantified.
import os
import pandas as pd
import csv
import sys
from pathlib import Path


def NG_boiler(F_boilers,_CAP_boiler,path_test):
    BTUtokWh_convert = 0.000293071 # 1BTU = 0.000293071 kWh
    mmBTutoBTU_convert = 10**6
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    components_path = os.path.join(path_test,'Energy Components')
    boiler_component = pd.read_csv(os.path.join(components_path,'boilers.csv'))
    UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    UPV_NG = float(editable_data['UPV_NG']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
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
    return Q_boiler,invest_boiler,OPC_boiler,OPE_boiler,eff_boiler,_CAP_boiler/eff_boiler
