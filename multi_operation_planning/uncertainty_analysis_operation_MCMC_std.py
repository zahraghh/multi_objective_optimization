import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import math
import random
import datetime as dt
import os
import sys
import pandas as pd
import csv
from pathlib import Path
from nested_dict import nested_dict
from collections import defaultdict
import re
import scipy.stats as st
import numpy as np
from matplotlib.ticker import FuncFormatter
from functools import partial
import warnings
from calendar import monthrange
import json
import pyDOE
import time
from os.path import exists
import multi_operation_planning
from multi_operation_planning import clustring_kmediod_operation, clustring_kmediod_PCA_operation,EGEF_operation
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
lat = float(editable_data['Latitude'])
lon = float(editable_data['Longitude'])
city = editable_data['city']
folder_path = os.path.join(path_test,str(city))
max_electricity = int(editable_data['max_electricity'])
max_heating = int(editable_data['max_heating'])
def best_fit_distribution(data,ax=None):
  """Model data by finding best fit distribution to data"""
  # Get histogram of original data
  y, x = np.histogram(data, bins='auto', density=True)
  x = (x + np.roll(x, -1))[:-1] / 2.0
  # Distributions to check
  #DISTRIBUTIONS = [
    #st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,st.hypsecant,
    #st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.invgamma,st.invgauss,
    #st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,
    #st.levy,st.levy_l,st.levy_stable,  #what's wrong with these distributions?
    #st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
  DISTRIBUTIONS = [st.beta,st.norm, st.uniform, st.expon,st.weibull_min,st.weibull_max,st.gamma,st.chi,st.lognorm,st.cauchy,st.triang,st.f]
  # Best holders
  best_distribution = st.norm  # random variables
  best_params = (0.0, 1.0)
  best_sse = np.inf
  # Estimate distribution parameters from data
  for distribution in DISTRIBUTIONS:
      # fit dist to data
      params = distribution.fit(data)
      warnings.filterwarnings("ignore")
      # Separate parts of parameters
      arg = params[:-2]
      loc = params[-2]
      scale = params[-1]
      # Calculate fitted PDF and error with fit in distribution
      pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
      sse = np.sum(np.power(y - pdf, 2.0))
      # if axis pass in add to plot
      try:
          if ax:
              pd.Series(pdf, x).plot(ax=ax)
          end
      except Exception:
          pass
      # identify if this distribution is better
      if best_sse > sse > 0:
          best_distribution = distribution
          best_params = params
          best_sse = sse
  return (best_distribution.name, best_params)
def LHS_generator(data,weather_input, num_scenarios_LHS,distribution=None):
    ax_new = plt.hist(data, bins = 'auto', range=(min(data)*0.8,max(data)*1.1), density= True)
    std_data = round(np.std(data),3)
    mean_data = round(np.mean(data),3)
    best_fit_name, best_fit_params = best_fit_distribution(data,ax_new)
    best_dist = getattr(st, best_fit_name)
    num_scenarios_LHS = int(editable_data['num_scenarios_LHS'])
    num_scenarios_revised = num_scenarios_LHS + 4
    #weather_data = pd.read_csv(os.path.join(os.path.join(os.path.join(path_test,'Weather files'),'AMYs'),epw_file_name+'.csv'))
    UA_scenario = []
    if std_data == 0:
        UA_scenario.append([mean_data]*num_scenarios_LHS)
        #print([mean_data]*num_scenarios_LHS)
    else:
        lhd = pyDOE.lhs(1, samples=num_scenarios_revised)
        dist_i = getattr(st,best_fit_name)
        if dist_i.shapes is None:
            lhd= dist_i(loc=mean_data, scale=std_data).ppf(lhd)  # this applies to both factors here
        else:
            params = best_fit_params
            arg = params[:-2]
            loc = round(params[-2],3)
            scale = round(params[-1],3)
             #DISTRIBUTIONS = [st.beta,st.norm, st.uniform, st.expon,
             #st.weibull_min,st.weibull_max,st.gamma,st.chi,st.lognorm,st.cauchy,st.triang,st.f]
            if dist_i.name=='weibull_max' or dist_i.name=='weibull_min' or dist_i.name=='triang':
                lhd= dist_i(c=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='beta':
                lhd= dist_i(a=arg[0],b=arg[1],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='gamma':
                lhd= dist_i(a=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='lognorm':
                lhd= dist_i(s=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='chi':
                lhd= dist_i(df=arg[0],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='f':
                lhd= dist_i(dfn=arg[0],dfd=arg[1],loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
            elif dist_i.name=='norm' or dist_i.name=='uniform' or dist_i.name=='expon':
                lhd= dist_i(loc=loc, scale=scale).ppf(lhd)  # this applies to both factors here
        x = []
        for i in range(len(lhd)):
            if weather_input=='gti':
                if lhd[i][0]<0:
                    lhd[i][0]=0
                if lhd[i][0]>1000:
                    lhd[i][0]=1000
            x.append(round(lhd[i][0],3))
        x.sort()
        #print(weather_input,x)
        #print(len(x[2:num_scenarios_LHS+2]),x[2:num_scenarios_LHS+2])
        UA_scenario.append(x[2:num_scenarios_LHS+2])
    return UA_scenario
def random_generator(data, num_scenarios,distribution=None):
    ax_new = plt.hist(data, bins = 'auto', range=(min(data)*0.8,max(data)*1.1), density= True)
    best_fit_name, best_fit_params = best_fit_distribution(data,ax_new)
    best_dist = getattr(st, best_fit_name)
    if (best_fit_name=='norm' or best_fit_name=='uniform' or best_fit_name=='expon' or best_fit_name=='cauchy'):
        dist_rd_value = best_dist.rvs(loc= best_fit_params[0] , scale= best_fit_params[1] , size=num_scenarios)
    elif (best_fit_name=='weibull_min' or best_fit_name=='weibull_max' or best_fit_name=='triang'):
        dist_rd_value = best_dist.rvs(c=best_fit_params[0], loc= best_fit_params[1] , scale= best_fit_params[2] , size=num_scenarios)
    elif best_fit_name=='gamma':
        dist_rd_value = best_dist.rvs(a=best_fit_params[0], loc= best_fit_params[1] , scale= best_fit_params[2] , size=num_scenarios)
    elif best_fit_name=='chi':
        dist_rd_value = best_dist.rvs(df=best_fit_params[0], loc= best_fit_params[1] , scale= best_fit_params[2] , size=num_scenarios)
    elif best_fit_name=='lognorm':
        dist_rd_value = best_dist.rvs(s=best_fit_params[0], loc= best_fit_params[1] , scale= best_fit_params[2] , size=num_scenarios)
    elif best_fit_name=='beta':
        dist_rd_value = best_dist.rvs(a=best_fit_params[0], b=best_fit_params[1], loc= best_fit_params[2] , scale= best_fit_params[3] , size=num_scenarios)
    elif best_fit_name=='f':
        dist_rd_value = best_dist.rvs(dfn=best_fit_params[0], dfd=best_fit_params[1], loc= best_fit_params[2] , scale= best_fit_params[3] , size=num_scenarios)

    return dist_rd_value
def weather_uncertain_input(type_input,number_weatherfile,represent_day,path_test):
    uncertain_dist =  defaultdict(lambda: defaultdict(list))
    uncertain_input = {}
    weather_data = {}
    for year in range(int(editable_data['starting_year']),int(editable_data['ending_year'])+1):
        weather_data[year] = pd.read_csv(os.path.join(folder_path,city+'_'+str(lat)+'_'+str(lon)+'_psm3_60_'+str(year)+'.csv'), header=None)[2:]
        uncertain_input[year] = weather_data[year][number_weatherfile]
        for key in represent_day:
            day = represent_day[key]
            hour = 0
            for index_in_year in range(2+(day+1)*24,2+(day+2)*24):
                uncertain_dist[key][hour].append(float(uncertain_input[year][index_in_year]))
                hour +=1
    return uncertain_dist
def UA_operation(num_scenarios):
    print('Starts generating scenarios (takes around 30 minutes)')
    global solar_UA,wind_UA
    clustring_kmediod_operation.kmedoid_clusters(path_test)
    data_all_labels,represent_day = clustring_kmediod_PCA_operation.kmedoid_clusters(path_test)
    electricity_EF_UA = EGEF_operation.EGEF_state(editable_data['State'])
    try:
        f=open(os.path.join(path_test,'UA_solar.json'))
        f=open(os.path.join(path_test,'UA_wind.json'))
        with open(os.path.join(path_test,'UA_solar.json')) as f:
            solar_UA = json.load(f)
        with open(os.path.join(path_test,'UA_wind.json')) as f:
            wind_UA = json.load(f)
    except IOError:
        solar_UA = weather_uncertain_input('GTI',46,represent_day,path_test)
        with open(os.path.join(path_test,'UA_solar.json'), 'w') as fp:
            json.dump(solar_UA, fp)
        wind_UA = weather_uncertain_input('wind_speed',8,represent_day,path_test)
        with open(os.path.join(path_test,'UA_wind.json'), 'w') as fp:
            json.dump(wind_UA, fp)
    num_scenarios_LHS = int(editable_data['num_scenarios_LHS'])
    num_clusters = int(editable_data['Cluster numbers'])+2
    demand_electricity_UA = defaultdict(lambda: defaultdict(list))
    demand_heating_UA = defaultdict(lambda: defaultdict(list))
    generated_day = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    generated_scenario = nested_dict(3,list)
    for cluster in range(num_clusters):
        for day in range(len(data_all_labels[cluster])):
            for hour in range(24):
                demand_electricity_UA[cluster][hour].append(round(data_all_labels[cluster][day][0][hour],3))
                demand_heating_UA[cluster][hour].append(round(data_all_labels[cluster][day][0][hour+24],3))
    for cluster in range(num_clusters):
        for hour in range(24):
            generated_day[cluster][hour] = [random_generator(demand_electricity_UA[cluster][hour],num_scenarios),
            random_generator(demand_heating_UA[cluster][hour],num_scenarios),
            random_generator(solar_UA[str(cluster)][str(hour)],num_scenarios),
            random_generator(wind_UA[str(cluster)][str(hour)],num_scenarios),
            random_generator(electricity_EF_UA,num_scenarios)]
    for cluster in range(num_clusters):
        for scenario in range(num_scenarios):
            print('state', cluster, scenario)
            rand_0=random.random() #initialize the acceptence probability for electricity
            rand_1=random.random() #initialize the acceptence probability for heating
            rand_2=random.random() #initialize the acceptence probability for solar
            rand_3=random.random() #initialize the acceptence probability for wind
            #timeout_scenario = time.time()
            for j in range(100): #number of iterations in the whole 24 hours.
                #timeout = time.time()
                s0 = []
                s1 = []
                s2 = []
                s3 = []
                s4 = []
                s0_acceptence = []
                s1_acceptence = []
                s2_acceptence = []
                s3_acceptence = []
                for hour in range(24):
                    #print('hour', hour)
                    scenario_0=random.randint(0, num_scenarios-1)
                    scenario_1=random.randint(0, num_scenarios-1)
                    scenario_2=random.randint(0, num_scenarios-1)
                    scenario_3=random.randint(0, num_scenarios-1)
                    scenario_4=random.randint(0, num_scenarios-1)
                    #Electricity Demand#
                    if hour==0:
                        if generated_day[cluster][hour][0][scenario_0]<0:
                            generated_day[cluster][hour][0][scenario_0]=0
                        elif round(generated_day[cluster][hour][0][scenario_0],3)>max_electricity:
                            generated_day[cluster][hour][0][scenario_0] = max_electricity
                        s0.append(round(generated_day[cluster][hour][0][scenario_0],3))
                        accept_0 = rand_0
                    else:
                        while(len(s0)==hour): #while a transition is not accepted
                            list_states_0 = [round(abs(i-s0[hour-1]),3) for i in demand_electricity_UA[cluster][hour-1]]
                            #print('demand elect',min(list_states_0), s0[hour-1], demand_electricity_UA[cluster][hour-1], generated_day[cluster][hour][0])
                            index_pre = list_states_0.index(min(list_states_0))
                            list_states_1 = [round(abs(i-demand_electricity_UA[cluster][hour][index_pre]),3) for i in generated_day[cluster][hour][0]]
                            list_states_1 = [z/sum(list_states_1) for z in list_states_1]
                            scenario_0=random.randint(0, len(list_states_1)-1)
                            if (accept_0<(1-list_states_1[scenario_0])):  # and (time.time()<timeout):
                                #Accepting the transition
                                s0.append(round(generated_day[cluster][hour][0][scenario_0],3))
                                s0_acceptence.append(1-list_states_1[scenario_0])
                                accept_0 = 1-list_states_1[scenario_0]
                            else:
                                #Rejecting the transition
                                rand_0=round(random.uniform(0,1),5)
                                accept_0 = rand_0 #changing the acceptence theta
                    #print('elect hour ',hour,'value ', s0[hour])

                    #Heating Demand#
                    if hour==0:
                        if round(generated_day[cluster][hour][1][scenario_1],3)<0:
                            generated_day[cluster][hour][1][scenario_1]=0
                        elif round(generated_day[cluster][hour][1][scenario_1],3)>max_heating:
                            generated_day[cluster][hour][1][scenario_1] = max_heating
                        s1.append(round(generated_day[cluster][hour][1][scenario_0],3))
                        accept_1 = rand_1
                    else:
                        while(len(s1)==hour): #while a transition is not accepted
                            list_states_0 = [round(abs(i-s1[hour-1]),3) for i in demand_heating_UA[cluster][hour-1]]
                            index_pre = list_states_0.index(min(list_states_0))
                            list_states_1 = [round(abs(i-demand_heating_UA[cluster][hour][index_pre]),3) for i in generated_day[cluster][hour][1]]
                            list_states_1 = [z/sum(list_states_1) for z in list_states_1]
                            scenario_1=random.randint(0, len(list_states_1)-1)
                            #print('mid analysis','year',list_states_1,generated_day[cluster][hour][2])
                            if (accept_1<(1-list_states_1[scenario_1])):  # and (time.time()<timeout):
                                #Accepting the transition
                                s1.append(round(generated_day[cluster][hour][1][scenario_1],3))
                                s1_acceptence.append(1-list_states_1[scenario_1])
                                accept_1 = 1-list_states_1[scenario_1]
                            else:
                                #Rejecting the transition
                                rand_1=round(random.uniform(0,1),5)
                                accept_1 = rand_1 #changing the acceptence theta

                    #Solar Irradiance#
                    if (hour==0) or (round(np.mean(solar_UA[str(cluster)][str(hour)]),0)==0):
                        if round(generated_day[cluster][hour][2][scenario_2],3)<0:
                            generated_day[cluster][hour][2][scenario_2]=0
                        s2.append(round(generated_day[cluster][hour][2][scenario_2],3))
                        accept_2 = rand_2
                        #print('cluster-scenario',cluster, scenario,'gti hour',hour,'value ', s2[hour])
                    else:
                        while(len(s2)==hour): #while a transition is not accepted
                            list_states_0 = [round(abs(i-s2[hour-1]),3) for i in list(solar_UA[str(cluster)][str(hour-1)])]
                            index_pre = list_states_0.index(min(list_states_0))
                            list_states_1 = [round(abs(i-list(solar_UA[str(cluster)][str(hour)])[index_pre])) for i in generated_day[cluster][hour][2]]
                            if sum(list_states_1)==0:
                                s3.append(round(generated_day[cluster][hour][3][scenario_3],3))
                            else:
                                list_states_1 = [z/sum(list_states_1) for z in list_states_1]
                                scenario_2=random.randint(0, len(list_states_1)-1)
                                if (accept_2<(1-list_states_1[scenario_2])):  # and (time.time()<timeout):
                                    #Accepting the transition
                                    s2.append(round(generated_day[cluster][hour][2][scenario_2],3))
                                    s2_acceptence.append(1-list_states_1[scenario_2])
                                    accept_2 = 1-list_states_1[scenario_2]

                                else:
                                    #Rejecting the transition
                                    rand_2=round(random.uniform(0,1),5)
                                    accept_2 = rand_2 #changing the acceptence theta
                        #print('gti hour ',hour,'value ', s2[hour])

                    #Wind Speed#
                    if (hour==0) or (round(np.mean(wind_UA[str(cluster)][str(hour)]),2)==0):
                        if round(generated_day[cluster][hour][3][scenario_3],3)<0:
                            generated_day[cluster][hour][3][scenario_3]=0
                        s3.append(round(generated_day[cluster][hour][3][scenario_3],3))
                        accept_3 = rand_3
                        #print('cluster-scenario',cluster, scenario,'gti hour',hour,'value ', s2[hour])
                    else:
                        while(len(s3)==hour): #while a transition is not accepted
                            list_states_0 = [round(abs(i-s3[hour-1]),3) for i in list(wind_UA[str(cluster)][str(hour-1)])]
                            index_pre = list_states_0.index(min(list_states_0))
                            list_states_1 = [round(abs(i-list(wind_UA[str(cluster)][str(hour)])[index_pre])) for i in generated_day[cluster][hour][3]]
                            if sum(list_states_1)==0:
                                s3.append(round(generated_day[cluster][hour][3][scenario_3],3))
                            else:
                                list_states_1 = [z/sum(list_states_1) for z in list_states_1]
                                scenario_3=random.randint(0, len(list_states_1)-1)
                                if (accept_3<(1-list_states_1[scenario_3])):
                                    #Accepting the transition
                                    s3.append(round(generated_day[cluster][hour][3][scenario_3],3))
                                    s3_acceptence.append(1-list_states_1[scenario_3])
                                    accept_3 = 1-list_states_1[scenario_3]
                                else:
                                    #Rejecting the transition
                                    rand_3=round(random.uniform(0,1),5)
                                    accept_3 = rand_3 #changing the acceptence theta

                    #Emissions#
                    if hour==0:
                        if round(generated_day[cluster][hour][4][scenario_4],3)<0:
                            generated_day[cluster][hour][4][scenario_4]=0
                        s4.append(round(generated_day[cluster][hour][4][scenario_4],3))
                    else:
                        list_states_0 = [abs(i-s4[hour-1]) for i in electricity_EF_UA]
                        index_pre = list_states_0.index(min(list_states_0))
                        list_states_1 = [abs(i-electricity_EF_UA[index_pre]) for i in generated_day[cluster][hour][4]]
                        index_s4 = list_states_1.index(min(list_states_1))
                        s4.append(round(generated_day[cluster][hour][4][scenario_4],3))
                if j==0:
                    s0_final = s0
                    s1_final= s1
                    s2_final = s2
                    s3_final = s3
                    s0_prob = np.prod(s0_acceptence)
                    s1_prob = np.prod(s1_acceptence)
                    s2_prob = np.prod(s2_acceptence)
                    s3_prob = np.prod(s3_acceptence)
                else:
                    if  s0_prob< np.prod(s0_acceptence):
                        s0_final = s0
                        s0_prob = np.prod(s0_acceptence)
                    if  s1_prob< np.prod(s1_acceptence):
                        s1_final = s1
                        s1_prob = np.prod(s1_acceptence)
                    if  s2_prob< np.prod(s2_acceptence):
                        s2_final = s2
                        s2_prob = np.prod(s2_acceptence)
                    if  s3_prob< np.prod(s3_acceptence):
                        s3_final = s3
                        s3_prob = np.prod(s3_acceptence)
                #if j%10==0:
                #    print('probs',j,  s2_prob)
                #    print('gti',s2_final)
            #print('total time', (time.time()-timeout_scenario)/60)
            for hour in range(24):
                generated_scenario[cluster][scenario][hour].append(s0_final[hour])
                generated_scenario[cluster][scenario][hour].append(s1_final[hour])
                generated_scenario[cluster][scenario][hour].append(s2_final[hour])
                generated_scenario[cluster][scenario][hour].append(s3_final[hour])
                generated_scenario[cluster][scenario][hour].append(s4[hour])
                #print('final gti hour ',hour,'value ', s2_final[hour], s2_prob)

                    #print(hour,generated_scenario[cluster][scenario][hour])

    print('Ends generating scenarios')
    return generated_scenario
