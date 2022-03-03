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
        for scenario in range(num_scenarios_LHS):
            UA_scenario.append(mean_data)
    else:
        lhd = pyDOE.lhs(1, samples=num_scenarios_revised,criterion='center')
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
        for scenario in range(num_scenarios_LHS):
            UA_scenario.append(x[scenario+2])
    return UA_scenario
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
def UA_operation():
    print('Starts generating scenarios (takes around 30 minutes)')
    clustring_kmediod_operation.kmedoid_clusters(path_test)
    data_all_labels,represent_day = clustring_kmediod_PCA_operation.kmedoid_clusters(path_test)
    electricity_EF_UA = EGEF_operation.EGEF_state(editable_data['State'])
