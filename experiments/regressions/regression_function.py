"""
regression_function.py is a function to perform stand variable estimations with the use of machine learning regression algorithms.
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
  - variable to be estimated
  - algorithm to be used (SVR or GPR; Support vector regression or Gaussian process regression)
  - path to target and feature values (csv-file)
  - size of the holdout set (percentage of the original target and feature dataset)


"""

def callregressor(VARIABLE, algorithm, csvfile_location, holdoutset_size):
    
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
    Creating a holdout set
    
       - "Hold out" a portion of the data before beginning the model fitting process
       - Use the holdout set for final estimation accuracy evaluation
    
    """
    
    # Get standids for holdout set. The size of the holdout set is given as function input.
    import get_holdout_set
    holdout_stands = get_holdout_set.create_holdout_set(csvfile_location, holdoutset_size)
    
    print("\nSize of the holdout set is:", len(holdout_stands))
        
    # Now we have a holdout set list that is created randomly above.
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
    
    return rRMSE, RMSE, RMSE_ratio, bias, baseline_RMSE, elap_time