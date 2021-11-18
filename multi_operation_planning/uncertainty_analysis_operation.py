import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import csv
import math
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
import multi_operation_planning
from multi_operation_planning import clustring_kmediod_operation, clustring_kmediod_PCA_operation,EGEF_operation

editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
lat = float(editable_data['Latitude'])
lon = float(editable_data['Longitude'])
city = editable_data['city']
folder_path = os.path.join(sys.path[0],str(city))
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
    clustring_kmediod_operation.kmedoid_clusters(path_test)
    data_all_labels,represent_day = clustring_kmediod_PCA_operation.kmedoid_clusters(path_test)
    electricity_EF_UA = EGEF_operation.EGEF_state(editable_data['State'])
    solar_UA = weather_uncertain_input('GTI',46,represent_day,path_test)
    wind_UA = weather_uncertain_input('wind_speed',8,represent_day,path_test)
    num_clusters = int(editable_data['Cluster numbers'])+2
    demand_electricity_UA = defaultdict(lambda: defaultdict(list))
    demand_heating_UA = defaultdict(lambda: defaultdict(list))
    generated_day = defaultdict(lambda: defaultdict(list))
    generated_scenario = nested_dict(3,list)
    for cluster in range(num_clusters):
        for day in range(len(data_all_labels[cluster])):
            for hour in range(24):
                demand_electricity_UA[cluster][hour].append(round(data_all_labels[cluster][day][0][hour],2))
                demand_heating_UA[cluster][hour].append(round(data_all_labels[cluster][day][0][hour+24],2))
    for cluster in range(num_clusters):
        for hour in range(24):
            generated_day[cluster][hour] = [random_generator(demand_electricity_UA[cluster][hour],num_scenarios),
            random_generator(demand_heating_UA[cluster][hour],num_scenarios),
            random_generator(solar_UA[cluster][hour],num_scenarios),
            random_generator(wind_UA[cluster][hour],num_scenarios),
            random_generator(electricity_EF_UA,num_scenarios)]
    for cluster in range(num_clusters):
        for hour in range(24):
            for day in range(num_scenarios):
                if round(generated_day[cluster][hour][0][day],3)<0:
                    generated_day[cluster][hour][0][day]=0
                elif round(generated_day[cluster][hour][0][day],3)>max_electricity:
                    generated_day[cluster][hour][0][day] = max_electricity
                if round(generated_day[cluster][hour][1][day],3)<0:
                    generated_day[cluster][hour][1][day]=0
                elif round(generated_day[cluster][hour][1][day],3)>max_heating:
                    generated_day[cluster][hour][1][day] = max_heating
                if round(generated_day[cluster][hour][2][day],3)<0:
                    generated_day[cluster][hour][2][day]=0
                if round(generated_day[cluster][hour][3][day],3)<0:
                    generated_day[cluster][hour][3][day]=0
                if round(generated_day[cluster][hour][4][day],3)<0:
                    generated_day[cluster][hour][4][day]=0
                generated_scenario[cluster][day][hour].append(round(generated_day[cluster][hour][0][day],3))
                generated_scenario[cluster][day][hour].append(round(generated_day[cluster][hour][1][day],3))
                generated_scenario[cluster][day][hour].append(round(generated_day[cluster][hour][2][day],3))
                generated_scenario[cluster][day][hour].append(round(generated_day[cluster][hour][3][day],3))
                generated_scenario[cluster][day][hour].append(round(generated_day[cluster][hour][4][day],3))

    print('Ends generating scenarios')
    return generated_scenario
