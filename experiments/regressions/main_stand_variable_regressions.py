"""
main_stand_variable_regressions.py is a Python script to run machine learning regression algorithms to estimate stand variables from the standwise data provided by the Finnish Forest Centre (Metsakeskus). 
Copyright (C) 2018 Eelis Halme

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
If not, see http://www.gnu.org/licenses/

******************************************************************************************


Inputs:
  - Target and feature values (path to csv-file where band reflectances are the last columns)
  - Size of the holdout set to be used
  - Algorithm (Support vector or Gaussian process regression)
  - Stand variable that is wished to be estimated (has to be found from the csv-file)


"""
import os
import pandas as pd
import regression_function
currentfolder = os.getcwd()
import regression_treespeciesfunction




# Path to csv-file where targets and features are saved. (Note: the bands' reflectances, i.e. features, should be the last columns)
target_and_features = ".csv"

# Percentage telling how large portion of the target and feature dataset should be taken for the holdout set
holdoutset_size = 20  # in percentages

# Stand variable to be estimated (variable has to be written similarly as the column name in the csv-file)
STAND_VARIABLE = "basalarea"

# Machine learning regression algorithm (SVR or GPR)
algorithm = "SVR"



if STAND_VARIABLE == "treespecies":
    # Dataframe including accuracies for main tree species regressions based on species specific basal area and LAI
    df = regression_treespeciesfunction.maintreespeciesregression(STAND_VARIABLE, algorithm, target_and_features, holdoutset_size)
 
    
else:
    # Lists for estimation scores
    rRMSE, RMSE, RMSE_ratio, bias, baseline_RMSE, elap_time = regression_function.callregressor(STAND_VARIABLE, algorithm, target_and_features, holdoutset_size)
    
    # Constructing Python Pandas DataFrame
    d  = {'Variable':[STAND_VARIABLE], 'rRMSE':rRMSE, 'RMSE':RMSE, 'RMSE_ratio_(%)':RMSE_ratio, 'baseline_RMSE':baseline_RMSE, 'Bias':bias, 'Time':elap_time}
    df = pd.DataFrame(data=d)
    cols = ['Variable', 'rRMSE', 'RMSE', 'RMSE_ratio_(%)', 'baseline_RMSE', 'Bias', 'Time']
    df = df[cols]



# Save the estimation results to a new Excel file (create the file to the current working directory).
excel_name = STAND_VARIABLE + '_' + algorithm + '_regression_results.xlsx'
excel_file_name = os.path.join(currentfolder, excel_name)
df.to_excel(excel_file_name, encoding='utf-8', sheet_name='Sheet1', index=False)