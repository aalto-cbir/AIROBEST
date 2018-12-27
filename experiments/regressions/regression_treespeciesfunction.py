"""
regression_treespeciesfunction.py is a function to perform main tree species estimations with the use of machine learning regression algorithms.
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


This script includes two functions. One is making the actual predictions and 
another computing main tree species from species specific basal areas and leaf area indices.

The main tree species is computed based on basal area and LAI. One stand includes several strata,
and each stratum has its own main tree species, basal area and LAI. With species spefic information
about basal area and LAI, the main tree species can be defined by checking which species contributes
the most to the total basal area or LAI of one single stand.

This means that variables that are run in the regression are for one stand, for example,
pine_BA or broadleaved_LAI or spruce_BA. After the regression estimations, the new estimated main tree species are computed. 
The species values are discrete class values, thus the final accuracy is computed using confusion matrix.


Inputs:
  - variable to be estimated, i.e. "treespecies"
  - algorithm to be used (SVR or GPR; Support vector regression or Gaussian process regression)
  - path to target and feature values
  - size of the holdout set (percentage of the original target and feature dataset)
  - OR the actual holdout stand id list


"""

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calltreespeciesregressor(VARIABLE, algorithm, csvfile_location, holdout_stands):
    
    import time
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    start_time = time.time()
    
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    """
    Read data from csv
    
    """
    
    # Path to the csv-file that contains the target and feature values and read the csv file into Pandas DataFrame
    Dataframe = pd.read_csv(csvfile_location, encoding='utf-8')
    
    # Read the column names of the DataFrame
    column_names = list(Dataframe)
    
    # Index of the first reflectance value column, i.e. column number of "B1"
    ind = column_names.index('B1')
    
    # Column name list that includes all the band columns. (Note! The bands should be the last columns of the csv-file)
    bandlist = column_names[ind:]
    
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    """
    Target and feature values
    
    We extract the mean reflectance values into one list and one stand variable into another list.
    These lists will be used later in the regression tool implementation as target and feature values.
    
    """
    
    # Reflectances:
    # Assign new smaller dataframe for mean reflectances
    df_reflectances = Dataframe[bandlist]
    
    # Reflectance values are the feature values X.
    feature_values = df_reflectances.values.tolist()
    X = np.asarray(feature_values)
    
    # Stand variable:
    # First, choose which variable you want.
    print("\nEstimating:", VARIABLE)
    
    # Extract the variable column from the dataframe
    df_variable = Dataframe[VARIABLE]
    
    # Known target variable values are the target values y.
    target_values = df_variable.values.tolist()
    y = np.asarray(target_values)
    
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    """
    Holdout set
    
        - "Hold out" a portion of the data before beginning the model fitting process
        - Use the holdout set for final estimation accuracy evaluation
    
    """
        
    # We get a holdout set stand id list as function input.
    # Let's search the indices of stand id list that correspond to the stands found in holdout set list.
    
    # From the given target and feature dataset
    all_stands = Dataframe['standid'].values.tolist()
    
    # Find the corresponding indices
    index_lst = []
    for stand in holdout_stands:
        index = [ i for i, x in enumerate(all_stands) if x == stand ]
        index_lst.append(index[0])
    
    # Select now the correct target and feature values from the known indices
    y_holdout = y[np.asarray(index_lst)] # pick rows
    X_holdout = X[np.asarray(index_lst)]
    
    # Stands that do not correspond to the stands of the holdout set, will go to the training phase.
    # Search which stands are not in the holdout stand list
    not_common_stands = list(set(all_stands) - set(holdout_stands))
    indeks_lst = []
    for stand in not_common_stands:
        indeks = [ i for i, x in enumerate(all_stands) if x == stand ]
        indeks_lst.append(indeks[0])
    
    # Select now the correct target and feature values from the known indices
    y = y[np.asarray(indeks_lst)] # pick rows
    X = X[np.asarray(indeks_lst)]
    
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    """
    Machine learning regression algorithm
    
    """    
    
    if algorithm == "GPR":
        print("\nThe used algorithm is Gaussian Process Regression")
        # Gaussian Process regression (GPR) model
        kernel = C(1.00, (1e-5, 1e5)) * RBF(length_scale=1.00, length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1.00, noise_level_bounds=(1e-05, 1e5))
        reg = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=True, copy_X_train=True, random_state=None)
        
    if algorithm == "SVR":
        print("\nThe used algorithm is Support Vector Regression")
        # Support Vector regression (SVR) model
        reg = SVR(kernel='rbf', gamma=(1/(2e7)), tol=1e-10, C=1500.0, epsilon=0.35, shrinking=False, cache_size=1e8, verbose=False, max_iter=1e8)
        
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    """
    Fitting and prediction
    
       - Regression model is trained with the training dataset: X_training = X and y_training = y
       - After training we predict new values using the feature values found from the holdout set
       - Final accuracy evaluation is then performed using the targets of the holdout set
       
    """
    
    # Fit the regression model according to the given training data.
    print("\nTraining the regression model... ")
    X_training = X
    y_training = y
    reg.fit(X_training, y_training)
    
    # Perform regression on samples in the holdout set
    print("\nPrediction with holdout set... ")
    X_validation = X_holdout
    y_validation = y_holdout
    y_pred = reg.predict(X_validation)
    
    # Root mean-square-error (RMSE)
    def rmse(true, predicted):
        return np.sqrt(((true - predicted) ** 2).mean())
    RMSE = rmse(y_validation, y_pred)
    print("\nRMSE: {0:.2f}".format(RMSE))
    
    # Relative RMSE
    rRMSE = RMSE / (y_validation.mean())
    print("Relative RMSE: {0:.2f}".format(rRMSE))
    
    # Baseline RMSE: This is the RMSE for a hypothetical regression model that would always predict the mean of the target vector as the answer.
    baseline_RMSE = rmse(y_validation, y_validation.mean())
    
    # RMSE-ratio: how much the RMSE is from the baseline RMSE
    RMSE_ratio = (RMSE/baseline_RMSE) * 100
    print("RMSE-ratio: {0:.2f}%".format(RMSE_ratio))
    
    # Bias can be calculated as the difference between the mean values of predicted target values (y_pred.mean())
    # and the mean values of true target values (y_true.mean()). [Fazakas et al. 1999]
    # To compare bias between different variables, we divide the bias with mean of true target values.
    bias = (((y_pred.mean())) - (y_validation.mean())) / (y_validation.mean())
    print("Bias: {0:.2f}".format(bias))

    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    elap_time = (time.time() - start_time)
    m, s = divmod(elap_time, 60)
    h, m = divmod(m, 60)
    print("\nTime used in regression:")
    print("%dh %02dmin %02dsec " % (h, m, s))
    print("")
    
    return y_validation, y_pred


def maintreespeciesregression(FOREST_VARIABLE, algorithm, csvfile_location, holdoutset_size):
    
    """
    Estimating main tree species of a tree stand, based on species specific data.
    The main tree species is calculated only for stands where the species dominance is over 75%.
    
    Use a list of tree species data.
     
    All possbile stand variables to estimate:
    'pine_BA'  'pine_LAI'  'spruce_BA'  'spruce_LAI'  'broadleaved_BA'  'broadleaved_LAI' 
             
    """
    
    # In order to make species specific regressions, the regression function needs to be called several times.
    # Hence, the holdout set needs to be given as input so that the same stands would be used for validation and training each time.
    import get_holdout_set
    # Get standids for holdout set. The size of the holdout set is given as function input.
    holdout_standid_lst = get_holdout_set.create_holdout_set(csvfile_location, holdoutset_size)
    print("\nSize of the holdout set is:", len(holdout_standid_lst))
    
    import numpy as np
    import pandas as pd
    
    # Create dataframes for the true values and predictions
    d_true = {'pine_BA':[], 'pine_LAI':[], 'spruce_BA':[], 'spruce_LAI':[], 'broadleaved_BA':[], 'broadleaved_LAI':[]}
    df_true = pd.DataFrame(data=d_true)
    cols_true = ['pine_BA', 'pine_LAI', 'spruce_BA', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LAI']
    df_true = df_true[cols_true]
    
    d_pred = {'pine_BA':[], 'pine_LAI':[], 'spruce_BA':[], 'spruce_LAI':[], 'broadleaved_BA':[], 'broadleaved_LAI':[]}
    df_pred = pd.DataFrame(data=d_pred)
    cols_pred = ['pine_BA', 'pine_LAI', 'spruce_BA', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LAI']
    df_pred = df_pred[cols_pred]
    
    # Define stand variables to be predicted
    variable_list = ['pine_BA', 'pine_LAI', 'spruce_BA', 'spruce_LAI', 'broadleaved_BA', 'broadleaved_LAI']
    
    
    """
    Function returns validation y-vector and prediction y-vector per one forest variable.
    Let's save these into two different dataframes: one for true values, one for predicted values.
    
    """
    # Predict the variables from the list
    for variable in variable_list:
        FOREST_VARIABLE = variable
        
        # Call for the regression function
        y_true, y_pred = calltreespeciesregressor(FOREST_VARIABLE, algorithm, csvfile_location, holdout_standid_lst)
        
        # Add true values and predicted values of this forest variable to the correct columns in the dataframes
        df_true[FOREST_VARIABLE] = y_true 
        df_pred[FOREST_VARIABLE] = y_pred
    
    
    """
    Add up all tree species data. This equals to the actual stand basal area or LAI.
    Using the true values, the correct stand value is obtained. The sum of predictions will probably differ from the correct one.
    
    """
    # --- Basal area ---
    # add up
    true_stand_BA = np.asarray(df_true['pine_BA'].values.tolist()) + np.asarray(df_true['spruce_BA'].values.tolist()) + np.asarray(df_true['broadleaved_BA'].values.tolist())
    pred_stand_BA = np.asarray(df_pred['pine_BA'].values.tolist()) + np.asarray(df_pred['spruce_BA'].values.tolist()) + np.asarray(df_pred['broadleaved_BA'].values.tolist())
    # assign to the dataframe
    d_true_BA  = {'stand_BA': true_stand_BA.tolist()}
    df_true_BA = pd.DataFrame(data=d_true_BA)
    df_true = df_true.assign(stand_BA=df_true_BA.values)
    d_pred_BA  = {'stand_BA': pred_stand_BA.tolist()}
    df_pred_BA = pd.DataFrame(data=d_pred_BA)
    df_pred = df_pred.assign(stand_BA=df_pred_BA.values)
    
    # --- Leaf area index ---
    # add up
    true_stand_LAI = np.asarray(df_true['pine_LAI'].values.tolist()) + np.asarray(df_true['spruce_LAI'].values.tolist()) + np.asarray(df_true['broadleaved_LAI'].values.tolist())
    pred_stand_LAI = np.asarray(df_pred['pine_LAI'].values.tolist()) + np.asarray(df_pred['spruce_LAI'].values.tolist()) + np.asarray(df_pred['broadleaved_LAI'].values.tolist())
    # assign to the dataframe
    d_true_LAI  = {'stand_LAI': true_stand_LAI.tolist()}
    df_true_LAI = pd.DataFrame(data=d_true_LAI)
    df_true = df_true.assign(stand_LAI=df_true_LAI.values)
    d_pred_LAI  = {'stand_LAI': pred_stand_LAI.tolist()}
    df_pred_LAI = pd.DataFrame(data=d_pred_LAI)
    df_pred = df_pred.assign(stand_LAI=df_pred_LAI.values)
    
    # add the corresponding standids to the dataframes
    d_holdout_stands  = {'standid': holdout_standid_lst}
    df_holdout_stands = pd.DataFrame(data=d_holdout_stands)
    df_true = df_true.assign(standid=df_holdout_stands.values)
    df_pred = df_pred.assign(standid=df_holdout_stands.values)
    
    
    """
    Compute percentages
       - That is how much species specific basal area or LAI contributes to the total stand value.
    
    """
    # Percentages of true values.
    true_pine_BA_perc = 100 * np.asarray(df_true['pine_BA'].values.tolist()) / true_stand_BA
    true_spruce_BA_perc = 100 * np.asarray(df_true['spruce_BA'].values.tolist()) / true_stand_BA
    true_broadleaved_BA_perc = 100 * np.asarray(df_true['broadleaved_BA'].values.tolist()) / true_stand_BA
    #
    true_pine_LAI_perc = 100 * np.asarray(df_true['pine_LAI'].values.tolist()) / true_stand_LAI
    true_spruce_LAI_perc = 100 * np.asarray(df_true['spruce_LAI'].values.tolist()) / true_stand_LAI
    true_broadleaved_LAI_perc = 100 * np.asarray(df_true['broadleaved_LAI'].values.tolist()) / true_stand_LAI
    
    # Percentages of predicted values.
    pred_pine_BA_perc = 100 * np.asarray(df_pred['pine_BA'].values.tolist()) / pred_stand_BA
    pred_spruce_BA_perc = 100 * np.asarray(df_pred['spruce_BA'].values.tolist()) / pred_stand_BA
    pred_broadleaved_BA_perc = 100 * np.asarray(df_pred['broadleaved_BA'].values.tolist()) / pred_stand_BA
    #
    pred_pine_LAI_perc = 100 * np.asarray(df_pred['pine_LAI'].values.tolist()) / pred_stand_LAI
    pred_spruce_LAI_perc = 100 * np.asarray(df_pred['spruce_LAI'].values.tolist()) / pred_stand_LAI
    pred_broadleaved_LAI_perc = 100 * np.asarray(df_pred['broadleaved_LAI'].values.tolist()) / pred_stand_LAI
    
    
    # If the maximum percentage is less than 75 %, assign -99999 as class value.
    # True class values
    true_BA_main_tree_classes = []
    for i in range(len(df_true)):
        percentages = [ true_pine_BA_perc[i], true_spruce_BA_perc[i], true_broadleaved_BA_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(1)
        if max_index[0] == 1:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(2)
        if max_index[0] == 2:
            if max(percentages)<75:
                true_BA_main_tree_classes.append(-99999)
            else:
                true_BA_main_tree_classes.append(3)
    
    true_LAI_main_tree_classes = []
    for i in range(len(df_true)):
        percentages = [ true_pine_LAI_perc[i], true_spruce_LAI_perc[i], true_broadleaved_LAI_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(1)
        if max_index[0] == 1:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(2)
        if max_index[0] == 2:
            if max(percentages)<75:
                true_LAI_main_tree_classes.append(-99999)
            else:
                true_LAI_main_tree_classes.append(3)
        
    
    # Predicted values
    pred_BA_main_tree_classes = []
    for i in range(len(df_pred)):
        percentages = [ pred_pine_BA_perc[i], pred_spruce_BA_perc[i], pred_broadleaved_BA_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            pred_BA_main_tree_classes.append(1)
        if max_index[0] == 1:
            pred_BA_main_tree_classes.append(2)
        if max_index[0] == 2:
            pred_BA_main_tree_classes.append(3)
    
    pred_LAI_main_tree_classes = []
    for i in range(len(df_pred)):
        percentages = [ pred_pine_LAI_perc[i], pred_spruce_LAI_perc[i], pred_broadleaved_LAI_perc[i] ]
        max_index = [ i for i,v in enumerate(percentages) if v==(max(percentages)) ]
        if max_index[0] == 0:
            pred_LAI_main_tree_classes.append(1)
        if max_index[0] == 1:
            pred_LAI_main_tree_classes.append(2)
        if max_index[0] == 2:
            pred_LAI_main_tree_classes.append(3) 
    
    # Write these tree species classes to the dataframes
    d_perc_true = {'species_BA': true_BA_main_tree_classes, 'species_LAI': true_LAI_main_tree_classes, 'standid': holdout_standid_lst}
    df_perc_true = pd.DataFrame(data=d_perc_true)
    df_true = pd.merge(left=df_true, right=df_perc_true, left_on='standid', right_on='standid')
    
    d_perc_pred = {'pred_species_BA': pred_BA_main_tree_classes, 'pred_species_LAI': pred_LAI_main_tree_classes, 'standid': holdout_standid_lst}
    df_perc_pred = pd.DataFrame(data=d_perc_pred)
    df_pred = pd.merge(left=df_pred, right=df_perc_pred, left_on='standid', right_on='standid')
    
    
    """
    Accuracy
    
    """
    from sklearn import metrics
    
    # Tree species in stands where over 75 % species dominance
    accuracies = []
    variables = [ 'BA', 'LAI' ]
    for vari in variables:
        if vari == 'BA':
            df = df_true[df_true.species_BA != -99999]  # delete rows/stands where don't have the dominant tree species
            len_BA_dataset = len(df)
            df = pd.merge(left=df, right=df_pred, left_on='standid', right_on='standid') # merge this new "cleaned" to the dataframe of predictions, so that we will have the same stands for true and predicted values. 
            TRUES = df['species_BA'].values.tolist()
            PREDICTIONS = df['pred_species_BA'].values.tolist()
        if vari == 'LAI':
            df = df_true[df_true.species_LAI != -99999]  # delete rows/stands where don't have the dominant tree species
            len_LAI_dataset = len(df)
            df = pd.merge(left=df, right=df_pred, left_on='standid', right_on='standid') # merge this new "cleaned" to the dataframe of predictions, so that we will have the same stands for true and predicted values. 
            TRUES = df['species_LAI'].values.tolist()
            PREDICTIONS = df['pred_species_LAI'].values.tolist()
            
        # Accuracy score. What percentages of the predictions were correct?
        accuracy = metrics.accuracy_score(TRUES, PREDICTIONS)
        print(vari+" Accuracy = {0:.2f}".format(accuracy))
        
        accuracies.append(accuracy)
    
    d_accuracies = {'': ['Accuracy'], 'basalarea': [accuracies[0]], 'LAI': [accuracies[1]]}
    df_accuracies = pd.DataFrame(data=d_accuracies)
    colnames = ['', 'basalarea', 'LAI'] 
    df_accuracies = df_accuracies[colnames]
    
    print("\nBA  dataset length:", len_BA_dataset)
    print("LAI dataset length:", len_LAI_dataset)
    
    return df_accuracies