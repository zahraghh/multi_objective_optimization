# Two-Stage Stochastic Programming
This repository provides a framework to perform multi-objective two-stage stochastic programming on a district energy system. In this framework, we consider uncertainties in energy demands, solar irradiance, wind speed, and electricity emission factors. This framework optimizes the sizing of energy components to minimize the total cost and operating CO<sub>2</sub> emissions. Natural gas boilers, combined heating and power (CHP), solar photovoltaic (PV), wind turbines, batteries, and the grid are the energy components considered in this repository. 

## How Can I Use this Repository?
To use this repository, we suggest using a new conda environment. You can download and install anaconda using the following link: https://docs.anaconda.com/anaconda/install/.

After anaconda is installed, search for the anaconda prompt on your system:
- Windows: Click Start, search, or select Anaconda Prompt from the menu.
- macOS: Cmd+Space to open Spotlight Search and type “Navigator” to open the program.
- Linux–CentOS: Open Applications - System Tools - termin
    
Create a new environment for this repository, two_stage_env. We have tested this framework using python=3.7.7.
```
conda create -n two_stage_env python=3.7.7
```

Make sure the environment is created. By running the following code, the list of available environments, including two_stage_env, should be shown.
```
conda env list
```
Activate two_stage_env environment. This command should change the environment from base to two_stage_env.
```
conda activate two_stage_env
```
Now a new environment, two_stage_env, is ready to test the repository on it. 

Two packages should be installed using the conda command in the two_stage_env environment.

1. install scikit-learn-extra in the conda environment:
```
conda install -c conda-forge scikit-learn-extra 
```
or from PiPy:
```
pip install scikit-learn-extra

```
2. install a solver that is available for public use:
```
conda install glpk --channel conda-forge
```
or from PiPy:
```
pip install glpk

```
Download the ZIP file of this repository from this link: https://github.com/zahraghh/Two_Stage_SP/tree/IMECE.

Unzip the "Two_Stage_SP-IMECE" folder and locally install the package using the pip command. The /path/to/Two_Stage_SP-IMECE is the path to the "Two_Stage_SP-IMECE" folder that contains a setup.py file. 
```
pip install -r  /path/to/Two_Stage_SP-IMECE/requirements.txt
```

To use this repository, you should directly compile the "main_two_stage_SP.py" code in the "Framework Test_University of Utah" folder.

Have a look at the "Framework Test_University of Utah" folder. Four files are needed to compile the "main_two_stage_SP.py" code successfully:
1. "Energy Components" folder containing energy components characteristics
2. "editable_values.csv' file containing variable inputs of the package
3. "total_energy_demands.csv" file containing the aggregated hourly electricity, heating, and cooling demands of a group of buildings
4. "main_two_stage_SP.py" file to be compiled and run the two-stage stochastic programming optimization

## What Can I change?
Three sets of input data are present that a user can change to test a new/modified case study.

### editable_values.csv file
The first and primary input is the "editable_values.csv" file. This CSV file consists of four columns: 

1. The first column is "Names (do not change this column)," which provides the keys used in different parts of the code; therefore, please, leave this column unchanged. 

2. The second column is "Values" that a user can change. The values are yes/no questions, text, or numbers, which a user can modify to make it specific to their case study or leave them as they are. 

3. The third column is "Instruction." This column gives some instructions in filling the "Value" column, and if by changing the "Value," the user must change other rows in the CSV file or not. Please, if you want to change a value, read its instruction. 

4. The fourth column is "Where it's used," which gives the subsection of each value. This column can show the rows that are related to each other. 

The "editable_values.csv" consists of four main sections: 
1. The first section is "Setting Up the Framework." In this section, the user fills the rows from 5 to 11 by answering a series of yes/no questions. If this is the first time a user compiles this program, the answer to all of the questions is 'yes.' A user can change the values to 'no' if they have already downloaded/generated the files for that row. For example, if the weather data is downloaded and Global Tilted Irradiance (GTI) is calculated on flat plates, a user can change the row 5 value to 'no' to skip that part. 

2. The second section is "Weather Data Analysis." In this section, the user fills the rows from 15 to 28. These rows are used to download the data from the National Solar Radiation Database (NSRDB) and using the available solar irradiance in the NSRDB file to calculate the GTI on a flat solar photovoltaic plate. In this section, probability distribution functions (PDF) of uncertain meteorological inputs are calculated for the wind speed and GTI.

3. The third section is "Scenario Generation/Reduction" that consists of row 32 to 34. This section relates to generating uncertain scenarios of energy demands, solar irradiance, wind speed, and electricity emissions. After scenarios representing the uncertainties are generated in the "Scenarios Generation" folder, Principal component analysis (PCA) is used to extract an optimum number of features for each scenario. Then, the k-medoid algorithm is used to reduce the number of generated scenarios. If rows 8 (Search optimum PCA) and 9 (Search optimum clusters) have 'yes' values, two figures will be generated in the directory. These two figures can help a user familiar with the explained variance and elbow method to select the number of optimum clusters in the k-medoid algorithm and features in PCA. If a user is not familiar with these two concepts, they can select 18 features as a safe number for the optimum number of features. They can select 10 clusters as the optimum number of clusters. For more accuracy, a user can increase the number of clusters, but the computation time increases.

4. The fourth section is "District Energy System Optimization." In this section, the two-stage optimization of a district energy system considering uncertainties is performed to minimize cost and emissions. The rows from 38 to 47 are related to the district energy system's characteristics, input parameters to run the multi-objective optimization, and energy components that can be used in the district energy systems. The user is responsible for including a rational set of energy components to provide the electricity and heating needs of buildings. For example, if values of 'CHP' (row 53)and 'Boiler' are written as 'no,' this means no boiler and CHP system will be used in the district energy system. If no boiler and CHP system is used in the district energy system, the heating demand of buildings cannot be satisfied, and an error would occur.

### total_energy_demands.csv file
The "total_energy_demands.csv" file consists of the aggregated hourly electricity (kWh), heating (kWh), and cooling (kWh) needs of a group of buildings for a base year, representing the demand side. This file contains 8760 (number of hours in a year). A user can change electricity, heating, and cooling values to their own case study's energy demands. 

### Energy Components folder
The "Energy Components" folder consists of the CSV files of the five selected energy components in this repository, which are natural gas boilers, CHP, solar PV, wind turbines, and batteries. These CSV files for each energy component consist of a series of capacities, efficiencies, investment cost, operation & maintenance cost, and life span of the energy components. A user can modify these values or add more options to their CSV files to expand the decision space. 

## What are the Results?
If all parts of the framework are used, which means a user writes 'yes' for values of rows 5 to 11 in the "editable_values.csv" file, a series of CSV files and figures will be generated.
1. Two figures will be generated in the directory related to the optimum number of features in PCA and the optimum number of clusters in the k-medoid algorithm if rows 7, 8, and 9 are 'yes.' 
Suppose a user is familiar with the connection of explained variance and the number of features. In that case, they can use the "Explained variance vs PCA features" figure in the directory to select the optimum number of features. If a user is familiar with the elbow method, they can use the "Inertia vs Clusters" figure in the directory to select the optimum number of clusters. 
2. A folder named 'City_name_Discrete_EF_...' will be generated that contains five files. 
    1. One "ParetoFront.png" figure that shows the total cost and operating CO<sub>2</sub> emissions trade-off for different scenarios to minimize cost and emissions. 
    2. One CSV file, "objectives.csv," that represents the total cost and operating CO<sub>2</sub> emissions trade-off for the different scenarios to minimize cost and emissions. This CSV file contains the values that are shown in the "ParetoFront.png" figure. 
    3. Two parallel coordinates figures, "Parallel_coordinates_cost.png" and "Parallel_coordinates_emissions.png," which show the variation in the optimum energy configurations to minimize the total cost and operating CO<sub>2</sub> emissions. 
    4. One CSV file that contains the optimum sizing of five selected energy components in this repository, which are natural gas boilers, CHP, solar PV, wind turbines, and batteries, to minimize the total cost and operating CO<sub>2</sub> emissions. 
This CSV file contains all of the values that are used in the two parallel coordinates figures.

