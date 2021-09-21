# Multi-objective Optimization of Operation Planning
This repository provides a framework to perform multi-objective optimization of operation planning of district energy system. In this framework, we consider uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors. This framework optimizes the operation planning of energy components to minimize the operating cost and CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, batteries, and the grid are the energy components considered in this repository. 

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


Unzip the "Two_Stage_SP-JOSS" folder and locally install the package using the pip command. The /path/to/Two_Stage_SP-JOSS is the path to the "Two_Stage_SP-JOSS" folder that contains a setup.py file. 
```
pip install -e /path/to/multi_objective_optimization

```

To use this repository, you can directly compile the "main.py" code in the tests\test1 folder.

Have a look at the "tests\test1" folder. Four files are needed to compile the "main.py" code successfully:
1. "Energy Components" folder containing energy components characteristics
2. "editable_values.csv" file containing variable inputs of the package
3. "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings
4. "main.py" file to be compiled and run the two-stage stochastic programming optimization

## How to Use this Repository?
After the package is installed, we can use multi_objective_optimization\tests\Test folder that contains the necessary help files ("Energy Components" folder, "editable_values.csv', "total_energy_demands.csv") to have our main.py code in it. We can first download the weather files, calculate the global titlted irradiance, and quantify distributions of solar irradiance and wind speed by writing a similar code in main.py: 
```
import os
import sys
import pandas as pd
import csv
from multi_objective_optimization import download_windsolar_data, GTI,uncertainty_analysis
if __name__ == "__main__":
    #Reading the data from the Weather Data Analysis section of the editable_values.csv
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city_DES =str(editable_data['city'])
    #Downloading the weather data from NSRDB
    download_windsolar_data.download_meta_data(city_DES)
    #Calculating the  global tilted irradiance on a surface in the City
    GTI.GTI_results(city_DES)
    #Calculating the distribution of global tilted irradiance (might take ~5 mins)
    uncertainty_analysis.probability_distribution('GTI',46) #Name and the column number in the weather data
    #Calculating the distribution of wind speed (might take ~5 mins)
    uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
```
The outcome of this code is a new folder with the name of the city in  the editable_values.csv. If you haven't change the editable_values.csv, the folder name is Salt Lake City, which contains the needed weather parameters. 

After the weather data is generated, we can perfrom scenario generation using Monte Carlo simulation and scenario reduction using k-median algorithm to reduce the number of scenarios:
```
import os
import sys
import pandas as pd
import csv
from Two_Stage_SP import scenario_generation,clustring_kmediod_PCA
if __name__ == "__main__":
    #Reading the data from the  Scenario Generation/Reduction section of the editable_values.csv
    #We need "total_energy_demands.csv" for scenario generation/reduction
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    #Generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions
    state = editable_data['State']
    scenario_generation.scenario_generation_results(state)
    #Reduce the number scenarios of scenarios ...
    #using the PCA and k-medoid algorithm
    clustring_kmediod_PCA.kmedoid_clusters()
```
After scenarios are generated and reduced, the selected representative days are located in Scenario Generation\City\Representative days folder. Then, we perfrom the optimization on these selected representative days:
```
import os
import sys
import pandas as pd
import csv
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
from pyomo.opt import SolverFactory
from Two_Stage_SP import NSGA2_design_parallel_discrete
if __name__ == "__main__":
    #Reading the data from the  District Energy System Optimization section of the editable_values.csv
    # We need total_energy_demands.csv and a folder with charectristic of energy components
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    #Perfrom two-stage stochastic optimization
    problem= NSGA2_design_parallel_discrete.TwoStageOpt()
    #Make the optimization parallel
    with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
        algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
        algorithm.run(int(editable_data['num_iterations']))
    #Generate a csv file as the result
    NSGA2_design_parallel_discrete.results_extraction(problem, algorithm)
```
After the optimization is performed (migh take a few hours based on the number of iterations), a new folder (City_name_Discrete_EF_...)  is generated that contains the two csv files, sizing of energy components and objective values for the Pareto front. 

We can also perfrom the three parts together and geterate the plots using the following code:
```
### Performing Two Stage Stochastic Programming for the Design of District Energy system ###
import os
import sys
import pandas as pd
import csv
from platypus import NSGAII, Problem, Real, Integer, InjectedPopulation,GAOperator,HUX, BitFlip, SBX,PM,PCX,nondominated,ProcessPoolEvaluator
# I use platypus library to solve the muli-objective optimization problem:
# https://platypus.readthedocs.io/en/latest/getting-started.html
from pyomo.opt import SolverFactory
import Two_Stage_SP
from Two_Stage_SP import download_windsolar_data, GTI,uncertainty_analysis,scenario_generation,clustring_kmediod_PCA,NSGA2_design_parallel_discrete
if __name__ == "__main__":
    editable_data_path =os.path.join(sys.path[0], 'editable_values.csv')
    editable_data = pd.read_csv(editable_data_path, header=None, index_col=0, squeeze=True).to_dict()[1]
    city_DES =str(editable_data['city'])
    state = editable_data['State']
    #Do we need to generate the meteorlogical data and their distributions?
    if editable_data['Weather data download and analysis']=='yes':
        download_windsolar_data.download_meta_data(city_DES)
        #Calculating the  global tilted irradiance on a surface in the City
        GTI.GTI_results(city_DES)
        #Calculating the distribution of variable inputs: solar irradiance and wind speed
        print('Calculating the distribution of global tilted irradiance (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('GTI',46) #Name and the column number in the weather data
        print('Calculating the distribution of wind speed (might take ~5 mins)')
        uncertainty_analysis.probability_distribution('wind_speed',8) #Name and the column number in the weather data
    #Do we need to generate scenarios for uncertainties in ...
    #energy demands,solar irradiance, wind speed, and electricity emissions?
    if editable_data['Generate Scenarios']=='yes':
        print('Generate scenarios for uncertain variables')
        scenario_generation.scenario_generation_results(state)
    #Do we need to reduce the number scenarios of scenarios in ...
    #using the PCA and k-medoid algorithm?
    if editable_data['Perfrom scenario reduction']=='yes':
        print('Perfrom scenarios reduction using k-medoid algorithm')
        clustring_kmediod_PCA.kmedoid_clusters()
    #Do we need to perfrom the two stage stochastic programming using NSGA-II?
    if editable_data['Perform two stage optimization']=='yes':
        print('Perfrom two-stage stochastic optimization')
        problem= NSGA2_design_parallel_discrete.TwoStageOpt()
        with ProcessPoolEvaluator(int(editable_data['num_processors'])) as evaluator: #max number of accepted processors is 61 by program/ I have 8 processor on my PC
            algorithm = NSGAII(problem,population_size=int(editable_data['population_size']) ,evaluator=evaluator,variator=GAOperator(HUX(), BitFlip()))
            algorithm.run(int(editable_data['num_iterations']))
        NSGA2_design_parallel_discrete.results_extraction(problem, algorithm)
    #Do we need to generate Pareto-front and parallel coordinates plots for the results?
    if editable_data['Visualizing the final results']=='yes':
        from Two_Stage_SP.plot_results_design import parallel_plots,ParetoFront_EFs
        file_name = city_DES+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
        results_path = os.path.join(sys.path[0], file_name)
        if not os.path.exists(results_path):
            print('The results folder is not available. Please, generate the results first')
            sys.exit()
        Two_Stage_SP.plot_results_design.ParetoFront_EFs()
        Two_Stage_SP.plot_results_design.parallel_plots('cost')
        Two_Stage_SP.plot_results_design.parallel_plots('emissions')
        file_name = editable_data['city']+'_Discrete_EF_'+str(float(editable_data['renewable percentage']) )+'_design_'+str(editable_data['num_iterations'])+'_'+str(editable_data['population_size'])+'_'+str(editable_data['num_processors'])+'_processors'
        print('Plots are generated in the '+ file_name+' folder')


```

## What Can I change?
Three sets of input data are present that a user can change to test a new/modified case study.
