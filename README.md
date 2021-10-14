# Multi-objective Optimization of Operation Planning
This repository provides a framework to perform multi-objective optimization of operation planning of district energy system using two methods, an MILP solver and NSGA-II algorithm. In this framework, we consider uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors using the Monte Carlo simulation. In this framework, the operation planning of energy components are optimized to minimize the operating cost and CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, batteries, and the grid are the energy components considered in this framework. 

## How Can I Install this Repository?
To use this repository, you need to use either Python or Anaconda. You can download and install Python using the following link https://www.python.org/downloads/ or Anaconda using the following link https://docs.anaconda.com/anaconda/install/. 

Two packages should be installed using the conda or PyPI.

1. install scikit-learn-extra either in conda environment:
```
conda install -c conda-forge scikit-learn-extra 
```
or from PyPI:
```
pip install scikit-learn-extra
```
2. install a solver that is free and open-source either in conda environmnet:
```
conda install glpk --channel conda-forge
```
or from PyPI:
```
pip install glpk
```

Download the ZIP file of this repository from this link: https://github.com/zahraghh/multi_objective_optimization


Unzip the "multi_objective_optimization-Journal" folder and locally install the package using the pip command. The /path/to/multi_objective_optimization-Journal is the path to the "multi_objective_optimization-Journal" folder that contains a setup.py file. 
```
pip install -e /path/to/multi_objective_optimization-Journal

```

To use this repository, you can directly compile the "MILP_solver.py" code in the "MILP solver test" folder using GLPK solver or "NSGA-II.py" code in the "NSGA-II algorithm test" folder using NSGA-II algorithm. 

Have a look at the "MILP solver test"/"NSGA-II algorithm test" folder. Four files are needed to compile the "MILP_solver.py"/"NSGA-II.py" code successfully:
1. "Energy Components" folder containing energy components characteristics,
2. "editable_values.csv" file containing variable inputs of the package,
3. "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings, and
4. "MILP_solver.py"/"NSGA-II.py" file to be compiled and run the multi-objective optimization.

## How to Use this Repository?
After the package is installed, we can use multi_objective_optimization\"MILP solver test" or multi_objective_optimization\"NSGA-II algorithm test" folder that contains the necessary help files ("Energy Components" folder, "editable_values.csv', "total_energy_demands.csv") to have our  "MILP_solver.py" or  "NSGA-II.py" file code in it. 

### Part 1:  Weather Data Analysis 
We can first download the weather files and calculate the global titlted irradiance: 
```
import pandas as pd
import os
import sys
import csv
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate the meteorlogical data and their distributions?
    download_windsolar_data.download_meta_data(city_DES)
    #Calculating the  global tilted irradiance on a surface in the City
    GTI.GTI_results(city_DES,path_test)
```
The outcome of this code is a new folder with the name of the city in  the editable_values.csv. If you haven't change the editable_values.csv, the folder name is Salt Lake City, which contains the needed weather parameters to perfrom the optimization. 

### Part 2:  Scenario Generation/Reduction

After the weather data is generated, we can perfrom scenario generation using Monte Carlo simulation and scenario reduction using k-mean algorithm to reduce the number of scenarios:
```
import os
import sys
import pandas as pd
import csv
import json
import multi_operation_planning
from multi_operation_planning import scenario_generation_operation, uncertainty_analysis_operation
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    scenario_generation_operation.scenario_generation_results(path_test)
    generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
    with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
        json.dump(generated_scenario, fp)
```
The outcome of scenarios generation and reduction is the selected representative days that are located in Scenario Generation\City\Representative days folder. The number of representative days is "Cluster numbers" in the  editable_values.csv plus two extreme days (Cluster numbers+2). Another outcome of this step is a JSON file in the main folder "UA_operation_num_scenarios", where num_scenarios is stated in editable_values.csv.

### Part 3: Optimization of District Energy System
After scenarios are generated and reduced, the selected representative days are located in Scenario Generation\City\Representative days folder. Then, we perfrom the optimization on these selected representative days using the **MILP solver**:
```
import os
import sys
import pandas as pd
import csv
import multi_operation_planning
from multi_operation_planning import MILP_two_objective, MILP_results_repdays
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    print('Perfrom multi-objective optimization of operation planning')
    MILP_results_repdays.results_repdays(path_test)
```

or using the **NSGA-II algorithm**:
```
import os
import sys
import pandas as pd
import csv
import multi_operation_planning
from multi_operation_planning import NSGA_two_objectives
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    print('Perfrom multi-objective optimization of operation planning')
    NSGA_two_objectives.NSGA_Operation(path_test)
```

After the optimization is performed (migh take a few minutes to few hours based on the number of iterations and scenarios), a new folder (City_name_operation_MILP/EA_EF_...)  is generated that contains the two csv files for each day of generated scenarios for each representative day. 

### All Parts together
We can also perfrom the three parts together using the **MILP solver**:
```
### Performing Multi-objective Optimization of Operation planning of District Energy systems ###
### Using MILP Solver (GLPK) ###
import os
import sys
import pandas as pd
import csv
import json
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI, scenario_generation_operation, uncertainty_analysis_operation,MILP_two_objective,MILP_results_repdays
editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
path_test =  os.path.join(sys.path[0])
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES,path_test)
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        scenario_generation_operation.scenario_generation_results(path_test)
        generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
            json.dump(generated_scenario, fp)
    #Do we need to perfrom the two stage stochastic programming using MILP solver (GLPK)?
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        MILP_results_repdays.results_repdays(path_test)

```
or using the **NSGA-II algorithm**:
```
### Performing Multi-objective Optimization of Operation planning of District Energy systems ###
### Using NSGA-II Algorithm ###
import pandas as pd
import math
import os
import sys
import csv
import json
import multi_operation_planning
from multi_operation_planning import download_windsolar_data, GTI, scenario_generation_operation, uncertainty_analysis_operation,NSGA_two_objectives
###Decison Variables###
path_test =  os.path.join(sys.path[0])
editable_data_path =os.path.join(path_test, 'editable_values.csv')
editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
num_scenarios = int(editable_data['num_scenarios'])
if __name__ == "__main__":
    city_DES =str(editable_data['city'])
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES,path_test)
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        scenario_generation_operation.scenario_generation_results(path_test)
        generated_scenario=uncertainty_analysis_operation.UA_operation(int(editable_data['num_scenarios']))
        with open(os.path.join(path_test,'UA_operation_'+str(num_scenarios)+'.json'), 'w') as fp:
            json.dump(generated_scenario, fp)
    #Do we need to perfrom the multi-objective optimization of operation planning using NSGA-II?
    if editable_data['Perform multi-objective optimization']=='yes':
        print('Perfrom multi-objective optimization of operation planning')
        NSGA_two_objectives.NSGA_Operation(path_test)
```

## What Can I change?
Three sets of input data are present that a user can change to test a new/modified case study.

### editable_values.csv file
The first and primary input is the "editable_values.csv" file. This CSV file consists of four columns: 

1. The first column is "Names (do not change this column)," which provides the keys used in different parts of the code; therefore, please, leave this column unchanged. 

2. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can modify to make it specific to their case study or leave them as they are. 

3. The third column is "Instruction." This column gives some instructions in filling the "Value" column, and if by changing the "Value," the user must change other rows in the CSV file or not. Please, if you want to change a value, read its instruction. 

4. The fourth column is "Where it's used," which gives the subsection of each value. This column can show the rows that are related to each other. 

The "editable_values.csv" consists of four main sections: 
1. The first section is "Setting Up the Framework." In this section, the user fills the rows from 5 to 9 by answering a series of yes/no questions. If this is the first time a user compiles this program, the answer to all of the questions is 'yes.' A user can change the values to 'no' if they have already downloaded/generated the files for that row. For example, if the weather data is downloaded and Global Tilted Irradiance (GTI) is calculated on flat plates, a user can change the row 5 value to 'no' to skip that part. 

2. The second section is "Weather Data Analysis." In this section, the user fills the rows from 13 to 26. These rows are used to download the data from the National Solar Radiation Database (NSRDB) and using the available solar irradiance in the NSRDB file to calculate the GTI on a flat solar photovoltaic plate. 

3. The third section is "Scenario Generation/Reduction" that consists of row 30 to 35. This section relates to generating uncertain scenarios of energy demands, solar irradiance, wind speed, and electricity emissions using probability distribution functions (PDF) of uncertain energy demands, GTI, wind speed, and electricity EF.  After scenarios representing the uncertainties are generated in the "Scenarios Generation" folder, Principal component analysis (PCA) is used to extract an optimum number of features for each scenario. Then, the k-mean algorithm is used to reduce the number of generated scenarios. If rows 7 (Search optimum PCA) and 8 (Search optimum clusters) have 'yes' values, two figures will be generated in the directory. These two figures can help a user familiar with the explained variance and elbow method to select the number of optimum clusters in the k-mean algorithm and features in PCA. If a user is not familiar with these two concepts, they can select 18 features as a safe number for the optimum number of features. They can select 9 clusters as the optimum number of clusters. For more accuracy, a user can increase the number of clusters, but the computation time increases.

4. The fourth section is "District Energy System Optimization." In this section, the multi-objective optimization of operation planning of a district energy system considering uncertainties is performed to minimize operating cost and emissions. The rows from 39 to 58 are related to the district energy system's characteristics, input parameters to run the multi-objective optimization, and energy components that can be used in the district energy systems. The user is responsible for including a rational set of energy components to provide the electricity and heating needs of buildings. For example, if values of 'CHP' (row 53) and 'Boiler' are written as 'no,' this means no boiler and CHP system will be used in the district energy system. If no boiler and CHP system is used in the district energy system, the heating demand of buildings cannot be satisfied, and an error would occur.

### total_energy_demands.csv file
The "total_energy_demands.csv" file consists of the aggregated hourly electricity (kWh), heating (kWh), and cooling (kWh) needs of a fo for a base year, representing the demand side. This file contains 8760 (number of hours in a year). A user can change electricity, heating, and cooling values to their own case study's energy demands. 

### Energy Components folder
The "Energy Components" folder consists of the CSV files of the five selected energy components in this repository, which are natural gas boilers, CHP, solar PV, wind turbines, and batteries. These CSV files for each energy component consist of a series of capacities, efficiencies, investment cost, operation & maintenance cost, and life span of the energy components. A user can modify these values or add more options to their CSV files to expand the decision space. 
