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
