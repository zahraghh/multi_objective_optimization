#In this function, capita cost, operating cost, and operating emission of CHP
#Over the lifespan of district energy system is quantified.
import os
import pandas as pd
import csv
import sys
from pathlib import Path

def CHP(CAP_CHP_elect_size,F_CHP_size,path_test):
    BTUtokWh_convert = 0.000293071 # 1BTU = 0.000293071 kWh
    mmBTutoBTU_convert = 10**6
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    components_path = os.path.join(path_test,'Energy Components')
    CHP_component = pd.read_csv(os.path.join(components_path,'CHP.csv'))
    UPV_maintenance = float(editable_data['UPV_maintenance']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 7
    UPV_NG = float(editable_data['UPV_NG']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
    UPV_elect = float(editable_data['UPV_elect']) #https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf discount rate =3% page 21 utah
    NG_prices = float(editable_data['price_NG'])/293.001 #Natural gas price at UoU $/kWh
    electricity_prices =  float(editable_data['electricity_price'])/100 #6.8cents/kWh in Utah -->$/kWh CHANGE
    lifespan_chp = int(CHP_component['Lifespan (year)'][0])
    ###CHP system###
    if CAP_CHP_elect_size==0:
        return 0,0,0,0,0,0,0
    else:
        index_CHP = list(CHP_component['CAP_CHP_elect_size']).index(CAP_CHP_elect_size)
        IC_CHP = CHP_component['IC_CHP'][index_CHP] #investment cost for CHP system $/kW
        CAP_CHP_Q= CHP_component['CAP_CHP_Q'][index_CHP] #Natural gas input mmBTU/hr, HHV
        eff_CHP_therm = round(CHP_component['eff_CHP_therm'][index_CHP],2) #Thermal efficiency of CHP system Q/F
        eff_CHP_elect = round(CHP_component['eff_CHP_elect'][index_CHP],2) #Electricity efficiency of CHP system P/F
        OM_CHP =CHP_component['OM_CHP'][index_CHP] #cent/kWh to $/kWh. now is $/kWh
        gamma_CHP =CHP_component['gamma_CHP'][index_CHP] #cent/kWh to $/kWh. now is $/kWh
        E_CHP = F_CHP_size*eff_CHP_elect/100 #Electricty generation of CHP system kWh
        Q_CHP = F_CHP_size*eff_CHP_therm/100 #Heat generation of CHP system kWh
        #salvage_CHP = (lifespan_chp-lifespan_project+lifespan_chp*int(lifespan_project/lifespan_chp))/lifespan_chp
        invest_CHP = IC_CHP*CAP_CHP_elect_size #Investment cost of the CHP system $
        OPC_CHP =NG_prices*F_CHP_size + OM_CHP*E_CHP#O&M cost of CHP system $
        OPE_CHP = gamma_CHP*E_CHP # O&M emission of CHP system kg CO2
        return E_CHP,Q_CHP,invest_CHP,OPC_CHP,OPE_CHP,CAP_CHP_elect_size/eff_CHP_elect*100,CAP_CHP_elect_size*eff_CHP_therm/eff_CHP_elect,eff_CHP_therm
