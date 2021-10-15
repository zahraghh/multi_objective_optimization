import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets, sklearn.decomposition
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
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
def kmedoid_clusters(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    cluster_numbers= int(editable_data['Cluster numbers']) +2
    representative_days_path= os.path.join(path_test,'Scenario Generation',city, 'Operation Representative days')
    save_path = os.path.join(path_test, str('Scenario Generation') , city)

    representative_day = {}
    representative_scenarios_list = []
    for represent in range(cluster_numbers):
        representative_day[represent] = pd.read_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(represent)+'.csv'))
        representative_scenario = representative_day[represent]['Electricity total (kWh)'].tolist() + representative_day[represent]['Heating (kWh)'].tolist()
        representative_scenarios_list.append(representative_scenario)
    folder_path = os.path.join(path_test,str(city))
    range_data = ['low','medium','high']
    scenario_genrated = {}
    scenario_probability = defaultdict(list)
    scenario_number = {}
    num_scenario = 0
    i_solar= range_data[1]
    i_wind= range_data[1]
    i_emission= range_data[1]
    #laod the energy deamnd, solar, wind, and electricity emissions from scenario generation file
    for i_demand in range_data:
        if i_demand=='low':
            p_demand = 0.277778
        elif i_demand=='medium':
            p_demand = 0.444444
        elif i_demand=='high':
            p_demand = 0.277778
        for day in range(365):
            scenario_probability['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission].append(p_demand)
        scenario_number['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission]=  num_scenario
        num_scenario = num_scenario + 1
        scenario_genrated['D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission] = pd.read_csv(os.path.join(save_path, 'D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv'), header=None)

    features_scenarios = defaultdict(list)
    represent_day = defaultdict(list)

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
            for represent in range(cluster_numbers):
                if round(representative_day[represent]['Electricity total (kWh)'][0],1)==round(daily_list[0],1):
                    represent_day[represent] = i
            features_scenarios[k*days+i] = daily_list
            features_scenarios_nested[scenario][i] = features_scenarios[k*days+i]
            features_scenarios_list.append(features_scenarios[k*days+i])
            features_probability_list.append(scenario_probability[scenario][i])
        k = k+1
    A = np.asarray(features_scenarios_list)
    #Convert the dictionary of features to Series
    standardization_data = StandardScaler()
    A_scaled = standardization_data.fit_transform(A)
    inertia_list = []
    search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
    cluster_numbers= int(editable_data['Cluster numbers']) + 2
    kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(A_scaled)
    #kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=4).fit(scores_pca)
    label = kmedoids.fit_predict(A_scaled)
    #filter rows of original data
    probability_label = defaultdict(list)
    index_label = defaultdict(list)
    index_label_all = []
    filtered_label={}
    for i in range(cluster_numbers):
        filtered_label[i] = A_scaled[label == i]
        index_cluster=np.where(label==i)
        if len(filtered_label[i])!=0:
            index_cluster = index_cluster[0]
            for j in index_cluster:
                probability_label[i].append(features_probability_list[j])
                index_label[i].append(j)
                index_label_all.append(j)
        else:
            probability_label[i].append(0)
    sum_probability = []

    for key in probability_label.keys():
        sum_probability.append(sum(probability_label[key]))
    #print(kmedoids.predict([[0,0,0], [4,4,4]]))
    #print(kmedoids.cluster_centers_,kmedoids.cluster_centers_[0],len(kmedoids.cluster_centers_))
    A_scaled_list={}
    clusters={}
    clusters_list = []
    label_list = []
    data_labels={}
    data_all_labels = defaultdict(list)
    for center in range(len(kmedoids.cluster_centers_)):
        clusters['cluster centers '+str(center)]= kmedoids.cluster_centers_[center]
        clusters_list.append(kmedoids.cluster_centers_[center].tolist())
    for scenario in range(len(A_scaled)):
        data_all_labels[kmedoids.labels_[scenario]].append(standardization_data.inverse_transform(A_scaled[scenario].reshape(1,-1)))
        A_scaled_list[scenario]=A_scaled[scenario].tolist()
        A_scaled_list[scenario].insert(0,kmedoids.labels_[scenario])
        data_labels['labels '+str(scenario)]= A_scaled_list[scenario]
        label_list.append(A_scaled[scenario].tolist())
    df_clusters= pd.DataFrame(clusters)
    df_labels = pd.DataFrame(data_labels)
    df_clusters.to_csv(os.path.join(representative_days_path , 'cluster_centers_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
    df_labels.to_csv(os.path.join(representative_days_path , 'labels_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
    return data_all_labels,represent_day
