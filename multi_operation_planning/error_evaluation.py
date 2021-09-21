### Evaluating the inputs and errors of the EditableFile.csv file ###
import os
import sys
import pandas as pd
import csv
import math
def errors(path_test):
    editable_data_path =os.path.join(path_test, 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        if not isinstance(editable_data['city'], str):
            print('Please, enter "city" value in row 15 in EditableFile.csv file (e.g., Salt Lake City)')
            sys.exit()
        if not isinstance(editable_data['Longitude'], str):
            print('Please, enter "Longitude" value in row 16 in EditableFile.csv file (e.g., -111.888142)')
            sys.exit()
        if not isinstance(editable_data['Latitude'], str):
            print('Please, enter "Latitude" value in row 17 in EditableFile.csv file (e.g., 40.758478)')
            sys.exit()
        if not isinstance(editable_data['your_name'], str):
            print('Please, enter "your_name" value (without spacing) in row 18 in EditableFile.csv file (e.g., Zahra+Ghaemi)')
            sys.exit()
        if not isinstance(editable_data['reason_for_use'], str):
            print('Please, enter "reason_for_use" value (without spacing) in row 19 in EditableFile.csv file (e.g., Academic)')
            sys.exit()
        if not isinstance(editable_data['your_affiliation'], str):
            print('Please, enter "your_affiliation" value (without spacing) in row 20 in EditableFile.csv file (e.g., University+of+Utah)')
            sys.exit()
        if not isinstance(editable_data['your_email'], str):
            print('Please, enter "your_email" value in row 21 in EditableFile.csv file')
            sys.exit()
        if not isinstance(editable_data['mailing_list'], str):
            print('Please, enter "mailing_list" value in row 22 in EditableFile.csv file (e.g., yes)')
            sys.exit()
        if not isinstance(editable_data['SAM API key'], str):
            print('Please, enter "SAM API key" value in row 23 in EditableFile.csv file')
            sys.exit()
        if not isinstance(editable_data['Altitude'], str):
            print('Please, enter "Altitude" value in row 24 in EditableFile.csv file (e.g., 1288)')
            sys.exit()
        if not isinstance(editable_data['solar_tilt'], str):
            print('Please, enter "solar_tilt" value in row 25 in EditableFile.csv file (e.g., 35)')
            sys.exit()
        if not isinstance(editable_data['solar_azimuth'], str):
            print('Please, enter "solar_azimuth" value in row 26 in EditableFile.csv file (e.g., 180)')
            sys.exit()
        if not isinstance(editable_data['starting_year'], str):
            print('Please, enter "starting_year" value in row 27 in EditableFile.csv file (e.g., 1998)')
            sys.exit()
        if not isinstance(editable_data['ending_year'], str):
            print('Please, enter "ending_year" value in row 28 in EditableFile.csv file (e.g., 2019)')
            sys.exit()
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        if not isinstance(editable_data['State'], str):
            print('As "State" value in row 32 in EditableFile.csv file is empty, normal distribution with 10% standard deviation will be considered for electricity emissions')
            print('If case study is in the U.S., you can add the abbrevation in row 32')
    #Do we need to reduce the number scenarios of scenarios in ...
    #using the PCA and k-medoid algorithm?
    if editable_data['Perfrom scenario reduction']=='yes':
        if editable_data['Search optimum PCA']=='yes':
            #print('"Search optimum PCA" generates a plot to help you select the optimum number of features.')
            if not isinstance(editable_data['PCA numbers'], str):
                print('Please, enter "PCA numbers" value in row 33 in EditableFile.csv file (e.g., 18).')
                print('It indicates the max number of features in selecting the optimum number of features')
                sys.exit()
        else:
            if not isinstance(editable_data['PCA numbers'], str):
                print('Please, enter "PCA numbers" value in row 33 in EditableFile.csv file (e.g., 18).')
                sys.exit()

        if editable_data['Search optimum clusters']=='yes':
            #print('"Search optimum clusters" generates a plot to help you select the optimum number of clusters.')
            if not isinstance(editable_data['Cluster numbers'], str):
                print('Please, enter "Cluster numbers" value in row 34 in EditableFile.csv file (e.g., 10).')
                print('It indicates the max number of Cluster in selecting the optimum number of clusters')
                sys.exit()
        else:
            if not isinstance(editable_data['Cluster numbers'], str):
                print('Please, enter "Cluster numbers" value in row 34 in EditableFile.csv file (e.g., 10).')
                sys.exit()
    else:
        if (editable_data['PCA numbers']=='yes' or  editable_data['Search optimum clusters']=='yes'):
            print('Please, enter "perfrom scenario reduction" as yes in row 7 in EditableFile.csv file')
    #Do we need to perfrom the two stage stochastic programming using NSGA-II?
    if editable_data['Perform multi-objective optimization']=='yes':
        if not isinstance(editable_data['city EF'], str):
            print('Please, enter "city EF" value in row 38 in EditableFile.csv file (e.g., 1593 lbs/MWh)')
            print('For U.S. case studies, you can find the electricity generation emission factors at state level here: https://www.eia.gov/electricity/state/')
            sys.exit()
        if not isinstance(editable_data['renewable percentage'], str):
            print('Please, enter "renewable percentage" value in row 39 in EditableFile.csv file (e.g., 0.7), which means electricity emission factor = city EF * 0.7)')
            sys.exit()
        if not isinstance(editable_data['price_NG'], str):
            print('Please, enter "price_NG" value in row 40 in EditableFile.csv file (e.g., 5 cent/cubic-ft)')
            sys.exit()
        if not isinstance(editable_data['electricity_price'], str):
            print('Please, enter "electricity_price" value in row 41 in EditableFile.csv file (e.g., 40.758478)')
            sys.exit()
        if not isinstance(editable_data['lifespan_project'], str):
            print('Please, enter "lifespan_project" value in row 42 in EditableFile.csv file (e.g., 20)')
            sys.exit()
        if not isinstance(editable_data['UPV_maintenance'], str):
            print('Please, enter "UPV_maintenance" valuein row 43 in EditableFile.csv file (e.g., 14.887)')
            print('You can find the values for different lifespan of projects and discount rate here on Page 9: https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf')
            sys.exit()
        if not isinstance(editable_data['UPV_NG'], str):
            print('Please, enter "UPV_NG" valuein row 44 in EditableFile.csv file (e.g., 17.56)')
            print('You can find the values for different states, lifespan of projects, and discount rate here on Page 21: https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf')
            sys.exit()
        if not isinstance(editable_data['UPV_elect'], str):
            print('Please, enter "UPV_elect" valuein row 45 in EditableFile.csv file (e.g., 14.76)')
            print('You can find the values for different states, lifespan of projects, and discount rate here on Page 21: https://nvlpubs.nist.gov/nistpubs/ir/2019/NIST.IR.85-3273-34.pdf')
            sys.exit()
        if not isinstance(editable_data['PV_module'], str):
            print('Please, enter "PV_module" value in row 46 in EditableFile.csv file (e.g., 1.7)')
            sys.exit()
        if not isinstance(editable_data['roof_top_area'], str):
            print('Please, enter "roof_top_area" value in row 47 in EditableFile.csv file (e.g., 8000 sqr-m)')
            sys.exit()
        if not isinstance(editable_data['population_size'], str):
            print('Please, enter "population_size" value in row 48 in EditableFile.csv file (e.g., 50)')
            sys.exit()
        if not isinstance(editable_data['num_iterations'], str):
            print('Please, enter "num_iterations" value in row 49 in EditableFile.csv file (e.g., 1000)')
            sys.exit()
        if not isinstance(editable_data['num_processors'], str):
            print('Please, enter "num_processors" value in row 50 in EditableFile.csv file (e.g., 5)')
            sys.exit()
        if not isinstance(editable_data['starting_year'], str):
            print('Please, enter "starting_year" value in row 27 in EditableFile.csv file (e.g., 1998)')
            sys.exit()
