# Multi-objective Optimization of Operation Planning
This repository provides a framework to perform multi-objective optimization of operation planning of district energy system. In this framework, we consider uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors using Monte Carlo simulation. In this frameworkm, the operation planning of energy components are optimized to minimize the operating cost and CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, batteries, and the grid are the energy components considered in this repository. 

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
2. install a solver that is available for public use either in conda environmnet:
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

Have a look at the "MILP solver test" folder. Four files are needed to compile the "MILP_solver.py" code successfully:
1. "Energy Components" folder containing energy components characteristics,
2. "editable_values.csv" file containing variable inputs of the package,
3. "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings, and
4. "MILP_solver.py" file to be compiled and run the two-stage stochastic programming optimization

The same applied to the "NSGA-II algorithm test" folder, where "NSGA-II.py" file is used instead of "MILP_solver.py" file

## How to Use this Repository?
After the package is installed, we can use multi_objective_optimization\"MILP solver test" or multi_objective_optimization\"NSGA-II algorithm test" folder that contains the necessary help files ("Energy Components" folder, "editable_values.csv', "total_energy_demands.csv") to have our  "MILP_solver.py" or  "NSGA-II.py" file code in it. 

#Part 1: downloading weather data
We can first download the weather files, calculate the global titlted irradiance: 
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

After scenarios are generated and reduced, the selected representative days are located in Scenario Generation\City\Representative days folder. Then, we perfrom the optimization on these selected representative days:
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

or using the NSGA-II algorithm to perfrom the multi-objective optimization:
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

We can also perfrom the three parts together using the MILP solver:
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
or using the NSGA-II algorithm:
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
