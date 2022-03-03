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
import multi_operation_planning
from multi_operation_planning import clustring_kmean_forced
def kmedoid_clusters(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city = editable_data['city']
    save_path = os.path.join(path_test, str('Scenario Generation') , city)
    representative_days_path =  os.path.join(save_path,'Operation Representative days')
    if not os.path.exists(representative_days_path):
        os.makedirs(representative_days_path)
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
            scenario_probability['D:'+i_demand].append(p_demand)
        scenario_number['D:'+i_demand]=  num_scenario
        num_scenario = num_scenario + 1
        scenario_genrated['D:'+i_demand] = pd.read_csv(os.path.join(save_path, 'D_'+i_demand+'_S_'+i_solar+'_W_'+i_wind+'_C_'+i_emission+'.csv'), header=None)
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
    #Convert the dictionary of features to Series
    standardization_data = StandardScaler()
    A_scaled = standardization_data.fit_transform(A)
    inertia_list = []
    search_optimum_cluster = editable_data['Search optimum clusters'] # if I want to search for the optimum number of clusters: 1 is yes, 0 is no
    cluster_range = range(2,20,1)
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.edgecolor'] = 'black'
    if search_optimum_cluster=='yes':
        print('Defining the optimum number of clusters: ')
        plt.rc('font', size=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)
        plt.rc('ytick', labelsize=MEDIUM_SIZE)
        fig, ax = plt.subplots(figsize=(12, 6))
        for cluster_numbers in cluster_range:
            kmedoids = KMedoids(n_clusters=cluster_numbers, init="random",max_iter=1000,random_state=0).fit(A_scaled)
            inertia_list.append(kmedoids.inertia_)
            plt.scatter(cluster_numbers,kmedoids.inertia_)
            print('Cluster number:', cluster_numbers, '  Inertia of the cluster:', int(kmedoids.inertia_))
        ax.set_xlabel('Number of clusters',fontsize=BIGGER_SIZE)
        ax.set_ylabel('Inertia',fontsize=BIGGER_SIZE)
        #ax.set_title('The user should use "Elbow method" to select the number of optimum clusters',fontsize=BIGGER_SIZE)
        ax.plot(list(cluster_range),inertia_list)
        ax.set_xticks(np.arange(2,20,1))
        plt.savefig(os.path.join(sys.path[0], 'Inertia vs Clusters.png'),dpi=300,facecolor='w')
        plt.close()
        print('"Inertia vs Clusters" figure is saved in the directory folder')
        print('You can use the figure to select the optimum number of clusters' )
        print('You should enter the new optimum number of clusters in EditableFile.csv file and re-run this part')

    cluster_numbers= int(editable_data['Cluster numbers'])
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
        #print(data_all_labels)
        A_scaled_list[scenario]=A_scaled[scenario].tolist()
        A_scaled_list[scenario].insert(0,kmedoids.labels_[scenario])
        data_labels['labels '+str(scenario)]= A_scaled_list[scenario]
        label_list.append(A_scaled[scenario].tolist())
    df_clusters= pd.DataFrame(clusters)
    df_labels = pd.DataFrame(data_labels)
    df_clusters.to_csv(os.path.join(representative_days_path , 'cluster_centers_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)
    df_labels.to_csv(os.path.join(representative_days_path , 'labels_C_'+str(len(kmedoids.cluster_centers_))+'_L_'+str(len(kmedoids.labels_))+'.csv'), index=False)

    #Reversing PCA using two methods:
    #Reversing the cluster centers using method 1 (their results are the same)
    Scenario_generated_new = standardization_data.inverse_transform(kmedoids.cluster_centers_)

    #print('15 representative days',clusters_reverse[0][0],Scenario_generated_new[0][0],standardization_data.mean_[0],standardization_data.var_[0])
    representative_day_all = {}
    total_labels = []
    represent_gaps = {}
    scenario_data = {}
    for key in filtered_label.keys():
        total_labels.append(len(filtered_label[key]))
    #print(len(probability_label[0])) 1990
    #print(len(filtered_label[0])) 1990
    for representative_day in range(len(Scenario_generated_new)):
        represent_gaps = {}
        scenario_data = {}
        for i in range(48):
            if Scenario_generated_new[representative_day][i]<0:
                Scenario_generated_new[representative_day][i] = 0
        for k in range(2): # 2 uncertain inputs
            scenario_data[k] = Scenario_generated_new[representative_day][24*k:24*(k+1)].copy()
            min_non_z = np.min(np.nonzero(scenario_data[k]))
            max_non_z = np.max(np.nonzero(scenario_data[k]))
            represent_gaps[k]= [i for i, x in enumerate(scenario_data[k][min_non_z:max_non_z+1]) if x == 0]
            ranges = sum((list(t) for t in zip(represent_gaps[k], represent_gaps[k][1:]) if t[0]+1 != t[1]), [])
            iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
            #print('Present gaps are: ', representative_day,k, 'gaps', ', '.join([str(n) + '-' + str(next(iranges)) for n in iranges]))
            iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
            for n in iranges:
                next_n = next(iranges)
                if (next_n-n) == 0: #for data gaps of 1 hour, get the average value
                    scenario_data[k][n+min_non_z] = (scenario_data[k][min_non_z+n+1]+scenario_data[k][min_non_z+n-1])/2
                elif (next_n-n) > 0  and (next_n-n) <= 6: #for data gaps of 1 hour to 4 hr, use interpolation and extrapolation
                    f_interpol_short= interp1d([n-1,next_n+1], [scenario_data[k][min_non_z+n-1],scenario_data[k][min_non_z+next_n+1]])
                    for m in range(n,next_n+1):
                        scenario_data[k][m+min_non_z] = f_interpol_short(m)
        data_represent_days_modified={'Electricity total (kWh)': scenario_data[0],
        'Heating (kWh)': scenario_data[1],
        'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
        #print(np.mean(Scenario_generated_new[representative_day][0:24]))
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)
    max_heating_scenarios_nested = nested_dict()
    max_electricity_scenarios_nested = nested_dict()
    total_heating_scenarios = []
    total_electricity_scenarios = []
    max_electricity_scenarios_nested_list = defaultdict(list)
    max_heating_scenarios_nested_list = defaultdict(list)
    accuracy_design_day = 0.99
    design_day_heating = []
    design_day_electricity = []
    representative_day_max = {}
    electricity_design_day = {}
    heating_design_day = {}
    i_demand=range_data[2]
    i_solar=range_data[1]
    i_wind=range_data[1]
    i_emission=range_data[1]
    scenario='D:'+i_demand
    for day in range(365):
        for i in range(24):
            k_elect=0
            list_k_electricity = []
            k_heat=0
            list_k_heating = []
            for represent in range(cluster_numbers):
                representative_day_max[represent] = pd.read_csv(os.path.join(representative_days_path ,'Represent_days_modified_'+str(represent)+'.csv'))
                electricity_demand = representative_day_max[represent]['Electricity total (kWh)'] #kWh
                heating_demand = representative_day_max[represent]['Heating (kWh)'] #kWh
                if features_scenarios_nested[scenario][day][0:24][i]>electricity_demand[i]:
                    k_elect=1
                    list_k_electricity.append(k_elect)
                k_elect=0
                if features_scenarios_nested[scenario][day][24:48][i]>heating_demand[i]:
                    k_heat=1
                    list_k_heating.append(k_heat)
                k_heat=0
            if sum(list_k_electricity)==cluster_numbers: #This hour does not meet by any of the representative days
                max_electricity_scenarios_nested_list[i].append(features_scenarios_nested[scenario][day][0:24][i])
                total_electricity_scenarios.append(features_scenarios_nested[scenario][day][0:24][i])
            if sum(list_k_heating)==cluster_numbers: #This hour does not meet by any of the representative days
                max_heating_scenarios_nested_list[i].append(features_scenarios_nested[scenario][day][24:48][i])
                total_heating_scenarios.append(features_scenarios_nested[scenario][day][24:48][i])
    total_electricity_scenarios.sort(reverse=True)
    total_heating_scenarios.sort(reverse=True)

    max_electricity_hour = total_electricity_scenarios[35]
    max_heating_hour = total_heating_scenarios[2]
    #print(max_heating_hour,len(total_heating_scenarios),np.min(total_heating_scenarios),np.max(total_heating_scenarios))
    i_demand=range_data[2]
    i_solar=range_data[1]
    i_wind=range_data[1]
    i_emission=range_data[1]
    scenario='D:'+i_demand+'/S:'+i_solar+'/W:'+i_wind+'/C:'+i_emission
    design_day_heating = []
    design_day_electricity = []
    for i in range(24):
        #print(max_electricity_scenarios_nested_list[i],max_electricity_hour)
        if len(max_electricity_scenarios_nested_list[i])==1:
            if max_electricity_scenarios_nested_list[i][0]<max_electricity_hour:
                design_day_electricity.append(max_electricity_scenarios_nested_list[i][0])
            else:
                design_day_electricity.append(max_electricity_hour)

        else:
                design_day_electricity.append(np.max([j for j in max_electricity_scenarios_nested_list[i] if j<max_electricity_hour]))
        #print(i,len(max_heating_scenarios_nested_list[i]),max_heating_scenarios_nested_list[i])
        heating_dd = [j for j in max_heating_scenarios_nested_list[i] if j<max_heating_hour]
        #print(heating_dd)
        design_day_heating.append(np.max(heating_dd))
    representative_day_max = {}
    electricity_demand_total = defaultdict(list)
    heating_demand_total = defaultdict(list)
    heating_demand_max = {}
    electricity_demand_max = {}
    for represent in range(cluster_numbers):
        representative_day_max[represent] = pd.read_csv(os.path.join(representative_days_path ,'Represent_days_modified_'+str(represent)+'.csv'))
        electricity_demand = representative_day_max[represent]['Electricity total (kWh)'] #kWh
        heating_demand = representative_day_max[represent]['Heating (kWh)'] #kWh
        #hours_representative_day= round(sum_probability[representative_day]/sum(sum_probability),4)*8760
        heating_demand_max[represent]= np.mean(heating_demand)
        electricity_demand_max[represent]= np.mean(electricity_demand)
    high_electricity_index = []
    high_heating_index = []
    high_electricity_value = []
    high_heating_value = []
    key_max_electricity=max(electricity_demand_max, key=electricity_demand_max.get)
    key_max_heating=max(heating_demand_max, key=heating_demand_max.get)
    for key, value in max_electricity_scenarios_nested.items():
        for inner_key, inner_value in max_electricity_scenarios_nested[key].items():
            if inner_value>electricity_demand_max[key_max_electricity]:
                high_electricity_index.append(scenario_number[key]*365+inner_key)
                high_electricity_value.append(inner_value)
    for key, value in max_heating_scenarios_nested.items():
        for inner_key, inner_value in max_heating_scenarios_nested[key].items():
            if inner_value>heating_demand_max[key_max_heating]:
                high_heating_index.append(scenario_number[key]*365+inner_key)
                high_heating_value.append(inner_value)
    sum_probability.append(0.5*len(total_electricity_scenarios)/len(index_label_all)*365)
    sum_probability.append(len(total_heating_scenarios)/len(index_label_all)*365)
    filtered_label[cluster_numbers]=len(total_electricity_scenarios)
    filtered_label[cluster_numbers+1]=len(total_heating_scenarios)
    representative_day = cluster_numbers
    data_represent_days_modified={'Electricity total (kWh)': design_day_electricity,
    'Heating (kWh)': representative_day_max[key_max_electricity]['Heating (kWh)'],
    'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
    df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
    df_represent_days_modified.to_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)

    representative_day = cluster_numbers+1
    data_represent_days_modified={'Electricity total (kWh)': representative_day_max[key_max_heating]['Electricity total (kWh)'],
    'Heating (kWh)': design_day_heating,
    'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
    df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
    df_represent_days_modified.to_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)

    for representative_day in range(len(Scenario_generated_new)):
        represent_gaps = {}
        scenario_data = {}
        for i in range(48): #24*5=120 features in each day
            if Scenario_generated_new[representative_day][i]<0:
                Scenario_generated_new[representative_day][i] = 0
        for k in range(2): # 2 uncertain inputs
            scenario_data[k] = Scenario_generated_new[representative_day][24*k:24*(k+1)].copy()
            min_non_z = np.min(np.nonzero(scenario_data[k]))
            max_non_z = np.max(np.nonzero(scenario_data[k]))
            represent_gaps[k]= [i for i, x in enumerate(scenario_data[k][min_non_z:max_non_z+1]) if x == 0]
            ranges = sum((list(t) for t in zip(represent_gaps[k], represent_gaps[k][1:]) if t[0]+1 != t[1]), [])
            iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
            #print('Present gaps are: ', representative_day,k, 'gaps', ', '.join([str(n) + '-' + str(next(iranges)) for n in iranges]))
            iranges = iter(represent_gaps[k][0:1] + ranges + represent_gaps[k][-1:])
            for n in iranges:
                next_n = next(iranges)
                if (next_n-n) == 0: #for data gaps of 1 hour, get the average value
                    scenario_data[k][n+min_non_z] = (scenario_data[k][min_non_z+n+1]+scenario_data[k][min_non_z+n-1])/2
                elif (next_n-n) > 0  and (next_n-n) <= 6: #for data gaps of 1 hour to 4 hr, use interpolation and extrapolation
                    f_interpol_short= interp1d([n-1,next_n+1], [scenario_data[k][min_non_z+n-1],scenario_data[k][min_non_z+next_n+1]])
                    for m in range(n,next_n+1):
                        scenario_data[k][m+min_non_z] = f_interpol_short(m)
        data_represent_days_modified={'Electricity total (kWh)': scenario_data[0],
        'Heating (kWh)': scenario_data[1],
        'Percent %': round(sum_probability[representative_day]*100/sum(sum_probability),4)}
        #print(np.mean(Scenario_generated_new[representative_day][0:24]))
        df_represent_days_modified=pd.DataFrame(data_represent_days_modified)
        df_represent_days_modified.to_csv(os.path.join(representative_days_path,'Represent_days_modified_'+str(representative_day)+ '.csv'), index=False)
        clustring_kmean_forced.kmedoid_clusters(path_test)
    return data_all_labels
