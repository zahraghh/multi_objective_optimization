import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets, sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn_extra
from scipy import stats
from scipy.stats import kurtosis, skew
from collections import defaultdict
import statistics
from itertools import chain
from scipy.interpolate import interp1d
from collections import defaultdict
from nested_dict import nested_dict
from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus

def kmedoid_clusters(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    save_path = os.path.join(path_test, str('Scenario Generation') , city)
    cluster_numbers= int(editable_data['Cluster numbers']) +2
    representative_days_path= os.path.join(path_test,'Scenario Generation',city, 'Operation Representative days')
    representative_day = {}
    representative_scenarios_list = []
    for represent in range(cluster_numbers):
        representative_day[represent] = pd.read_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(represent)+'.csv'))
        representative_scenario = representative_day[represent]['Electricity total (kWh)'].tolist() + representative_day[represent]['Heating (kWh)'].tolist() #+representative_day[represent]['GTI (Wh/m^2)'].tolist() + \
        #representative_day[represent]['Wind Speed (m/s)'].tolist() + representative_day[represent]['Electricity EF (kg/kWh)'].tolist()
        representative_scenarios_list.append(representative_scenario)
    folder_path = os.path.join(path_test,str(city))
    #GTI_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_GTI.csv'))
    #wind_speed_distribution = pd.read_csv(os.path.join(folder_path,'best_fit_wind_speed.csv'))
    range_data = ['low','medium','high']
    scenario_genrated = {}
    scenario_probability = defaultdict(list)
    scenario_number = {}
    num_scenario = 0
    i_solar=range_data[1]
    i_wind=range_data[1]
    i_emission=range_data[1]
    #laod the energy deamnd, solar, wind, and electricity emissions from scenario generation file
    for i_demand in range_data:
        if i_demand=='low':
            p_demand = 0.277778
        elif i_demand=='medium':
            p_demand = 0.444444
        elif i_demand=='high':
            p_demand = 0.277778
        for day in range(365):
            #p_solar[i_solar][day] = sum(solar_probability[i_solar][day*24:(day+1)*24])/(sum(solar_probability[range_data[0]][day*24:(day+1)*24])+sum(solar_probability[range_data[1]][day*24:(day+1)*24])+sum(solar_probability[range_data[2]][day*24:(day+1)*24]))
            #p_wind[i_wind][day] = sum(wind_probability[i_wind][day*24:(day+1)*24])/(sum(wind_probability[range_data[0]][day*24:(day+1)*24])+sum(wind_probability[range_data[1]][day*24:(day+1)*24])+sum(wind_probability[range_data[2]][day*24:(day+1)*24]))
            scenario_probability['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission].append(p_demand)
        scenario_number['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission]=  num_scenario
        num_scenario = num_scenario + 1
        scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission] = pd.read_csv(os.path.join(save_path, 'D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv'), header=None)

    features_scenarios = defaultdict(list)
    features_scenarios_list = []
    features_probability_list = []
    features_scenarios_nested = nested_dict()
    k=0
    days= 365
    for scenario in scenario_genrated.keys():
        scenario_genrated[scenario]=scenario_genrated[scenario]
        for i in range(days):
            if i==0:
                data = scenario_genrated[scenario][1:25]
            else:
                data = scenario_genrated[scenario][25+(i-1)*24:25+(i)*24]
            #Total electricity, heating, solar, wind, EF.
            daily_list =list(chain(data[0].astype('float', copy=False),data[1].astype('float', copy=False)))
            features_scenarios[k*days+i] = daily_list
            features_scenarios_nested[scenario][i] = features_scenarios[k*days+i]
            features_scenarios_list.append(features_scenarios[k*days+i])
            features_probability_list.append(scenario_probability[scenario][i])
        k = k+1
    A = np.asarray(features_scenarios_list)
    B = np.asarray(representative_scenarios_list)
    C = np.asarray(representative_scenarios_list+features_scenarios_list)

    #Convert the dictionary of features to Series
    standardization_data = StandardScaler()
    A_scaled = standardization_data.fit_transform(A)
    C_scaled = standardization_data.fit_transform(C)
    #print('Score of features', scores_pca)
    #print('Explained variance ratio',pca.explained_variance_ratio_)
    # Plot the explained variances
    # Save components to a DataFrame
    inertia_list = []
    search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
    kmeans = KMeans(n_clusters=cluster_numbers, n_init = 1, init = C_scaled[0:cluster_numbers]).fit(C_scaled)
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    z={i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    z_length = []
    for i in range(kmeans.n_clusters):
        z_length.append(len(z[i])/len(labels))
        data_represent_days_modified={'Electricity total (kWh)': representative_day[i]['Electricity total (kWh)'],
        'Heating (kWh)': representative_day[i]['Heating (kWh)'],
        'Percent %': round(len(z[i])/len(labels)*100,4)}
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(i)+ '.csv'), index=False)
    return z_length,representative_day
