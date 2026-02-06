import os
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import ms_aux as ms_aux


# DATA LOADER (RAW DATA)

def read_coercivity_csv(filepath):
    """Loader for full_micromagnetic_with_height.csv circulared by CNRS and UWK.


    Parameters
    ----------
    filepath : str
        Path of the full_micromagnetic_with_height.csv file.

    Returns
    -------
    DataFrame : Pandas DataFrame
        pd.DataFrame containing all data in full_micromagnetic_with_height.csv

    """
    fmdh_columns = {
                    'x_pos' : np.float64, # mm
                    'y_pos' : np.float64, # mm
                    'a' : np.float64, # Angstrom
                    'c' : np.float64, # Angstrom
                    'volume' : np.float64, # Angstrom3
                    'Hc_measured' : np.float64, # A / m
                    'main_phase_composition' : np.float64,
                    'formula' : str,
                    'Ms' : np.float64, # A / m 
                    'A' : np.float64, # pJ / m
                    'K1' : np.float64, # MJ / m3
                    'Hc_calculated' : np.float64, # A / m
                    'Mr_calculated' : np.float64, # A / m
                    'BHmax_calculated' : np.float64, # J / m3
                    'measured_height' : np.float64, # nm
                    'Neff' : np.float64
                   }

    return pd.read_csv(filepath,
                       sep=',',
                       header=0,
                       names = fmdh_columns.keys(),
                       dtype = fmdh_columns,
                       index_col=False)



# DATA PREPROCESSING AND "FEATURE ENGINEERING"
def preprocess_and_engineering_df(df):
    """Preprocessing and adding new informative features to the coercivity dataset.


    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe returned by the loader for the full_micromagnetic_with_height.csv file.

    Returns
    -------
    pd.DataFrame
        Only valid points in the input dataframe, together with the additional columns 
        'Nd', 'Ce', 'Fe', 'B', 'has_rare_earth', 'anisotropy_field', 'Hc-exp-minus-cal',
        'valid'.

    """

    #
    def add_element_content_info(df):
        def el_content_2_content_hot(d, element):
            return d[element]
        df['el_content'] = df['formula'].apply(ms_aux.compound_2_element_content)
        df['has_rare_earth'] = df['formula'].apply(ms_aux.has_rare_earth)

        for i in ['Nd', 'Ce', 'Fe', 'B']: # to be generalized
            el_cont_i = partial(el_content_2_content_hot, element=i)
            df[i] = df['el_content'].apply(el_cont_i)
        df.drop(['el_content'], axis = 1, inplace=True)
    
    #
    def add_Hc_delta(df):
        df['Hc-exp-minus-cal'] = df['Hc_measured'] - df['Hc_calculated']

    #
    def add_anisotropy_field(df):
        df['anisotropy_field'] = 2*df['K1'] / df['Ms'] / ms_aux.MU_NOUGHT
    
    #
    def flag_valid_datapoints(df):
        df['valid'] = np.isfinite(df['x_pos'])

        for i in ['y_pos', 'Hc_measured', 'Hc_calculated', 'measured_height']:
            df['valid'] = df['valid'] & np.isfinite(df[i])
    
    add_element_content_info(df)
    add_Hc_delta(df)
    add_anisotropy_field(df)
    flag_valid_datapoints(df)
    
    return df[df['valid']]
    

 # PLOT COERCIVITIES (EXP AND CALC SIDE BY SIDE)   
def plot_Hc_exp_and_calc(df, out_folder=None):
    """Side-by-side plot of Hc_measured and Hc_calculated.


    Parameters
    ----------
    df : Pandas DataFrame
        The coercivity dataframe returned by the preprocess_and_engineering_df function.
    
    out_folder : str, default=None
        Directory where to store the Hc exp vs sim plot.
        If None, the plot won't be saved on disk.
        If the folder does not exist yet, it will be created.
        
    Returns
    -------
    None
    
    """
    
    # Compute shared color scale across both datasets
    vmin = min(np.min(df['Hc_measured']), np.min(df['Hc_calculated']))
    vmax = max(np.max(df['Hc_measured']), np.max(df['Hc_calculated']))

    # Create side-by-side plots with shared color scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    sc1 = axes[0].scatter(
        df['x_pos'],
        df['y_pos'],
        c=df['Hc_measured'],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax )
    
    axes[0].set_title("$H_{c}^{exp}$")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_aspect('equal')


    axes[1].set_title("$H_{c}^{sim}$")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_aspect('equal')

    sc2 = axes[1].scatter(
        df['x_pos'],
        df['y_pos'],
        c=df['Hc_calculated'],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax )

    # Single shared colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), pad=0.1)
    cbar.set_label("A/m", labelpad=15)

    # cbar_ax2 = cbar.ax.twinx()
    # cbar_ax2.set_ylim(vmin*ms_aux.MU_NOUGHT, vmax*ms_aux.MU_NOUGHT)
    # cbar_ax2.set_ylabel("Hc (T)", rotation=270, labelpad=15)

    fig.suptitle('Measured ($H_{c}^{exp}$) and Computed ($H_{c}^{sim}$) Coercivity                                ')
    
    if (out_folder is not None):
        fig_filename = '/'.join([out_folder, 'H_exp_sim.png'])
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(fig_filename, dpi=300)
        plt.show()
        
    else:
        plt.show()
    
    plt.close()

# PLOT DELTA Hc
def plot_delta_Hc(df, out_folder=None):
    """Plot of Delta Hc (exp minus sym).


    Parameters
    ----------
    df : Pandas DataFrame
        The coercivity dataframe returned by the preprocess_and_engineering_df function.
    
    out_folder : str, default=None
        Directory where to store the Hc exp vs sim plot.
        If None, the plot won't be saved on disk.
        If the folder does not exist yet, it will be created.
        
    Returns
    -------
    None
    
    """


    plt.scatter(df['x_pos'],
                df['y_pos'],
                c=df['Hc-exp-minus-cal'],
                cmap="viridis")
    plt.colorbar(label="A/m")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    # plt.title(r"Difference between measured and computed coercivity\n\n$\Delta H_{c} \colon= H_{c}^{exp} - H_{c}^{sim}$")
    plt.title(r"$\Delta H_{c} \colon= H_{c}^{exp} - H_{c}^{sim}$")

    if (out_folder is not None):
        fig_filename = '/'.join([out_folder, 'deltaHc.png'])
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    plt.close()
        
        
# PLOT Ce and Nd content
def plot_ce_nd_content(df, out_folder):
    """Plot of Ce and Nd content.


    Parameters
    ----------
    df : Pandas DataFrame
        The coercivity dataframe returned by the preprocess_and_engineering_df function.
    
    out_folder : str, default=None
        Directory where to store plot.
        If None, the plot won't be saved on disk.
        If the folder does not exist yet, it will be created.
        
    Returns
    -------
    None
    
    """    
    vmin = min(np.min(df['Ce']), np.min(df['Nd']))
    vmax = max(np.max(df['Ce']), np.max(df['Nd']))

    # Create side-by-side plots with shared color scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    sc1 = axes[0].scatter(
        df['x_pos'],
        df['y_pos'],
        c=df['Ce'],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax )

    axes[0].set_title("[Ce]")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_aspect('equal')


    axes[1].set_title("[Nd]")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    axes[1].set_aspect('equal')

    sc2 = axes[1].scatter(
        df['x_pos'],
        df['y_pos'],
        c=df['Nd'].values,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax )

    # Single shared colorbar
    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), pad=0.1)
    cbar.set_label("1", labelpad=15)

    # cbar_ax2 = cbar.ax.twinx()
    # cbar_ax2.set_ylim(vmin*ms_aux.MU_NOUGHT, vmax*ms_aux.MU_NOUGHT)
    # cbar_ax2.set_ylabel("Hc (T)", rotation=270, labelpad=15)

    fig.suptitle('Ce and Nd content                                    ')

    if (out_folder is not None):
        fig_filename = '/'.join([out_folder, 'ce_nd_content.png'])
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    
    plt.close()

# PLOT HEIGHT
def plot_height(df, out_folder):
    """Plot of the wafer heght.


    Parameters
    ----------
    df : Pandas DataFrame
        The coercivity dataframe returned by the preprocess_and_engineering_df function.
    
    out_folder : str, default=None
        Directory where to store the plot.
        If None, the plot won't be saved on disk.
        If the folder does not exist yet, it will be created.
        
    Returns
    -------
    None
    
    """    

    plt.scatter(df['x_pos'],
                df['y_pos'],
                c=df['measured_height'],
                cmap="cividis")
    plt.colorbar(label="nm")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Probe's height \n\n$h$")

    if (out_folder is not None):
        fig_filename = '/'.join([out_folder, 'height.png'])
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

        
# GET CORRELATION MATRIX OF REQUESTED FEATURES AS A pd DataFrame
def get_corr_matrix(df, features):
    """Get the Pearson correlation matrix of specified features as a pd.DataFrame.


    Parameters
    ----------
    df : Pandas DataFrame
        The coercivity dataframe returned by the preprocess_and_engineering_df function.
    
    features : str
        List of valid column names in df, for which the user wants to compute the correlation matrix.

    Returns
    -------
    pd.DataFrame
        DataFrame storing the correlation matrix for the specified features.
        
    """    
    
    return df[features].corr(numeric_only=True)
 
 # FIT (BIVARIATE) LINEAR MODEL    
def run_linear_model(x,
                     y,
                     dep_variable_w_uom,
                     covariates_w_uom,
                     test_size=.2,
                     random_state=1,
                     out_folder=None):
    r"""
    Fit the linear model y = \sum_i w_i x[i].


    Parameters
    ----------
    x : List of np.arrays or List of pd.Series
        List of model covariates.
    
    y : np.array or pd.Series
        Dependent variable.
        
    dep_variable_w_uom : dict
        dict {order key : [dependent variable name label, dependent variable  unit of measure]}.
    
    covariates_w_uom : dict
        dict {order key : [covariate name label, covariate unit of measure]}.
        
    test_size : float64
        Test size used to split the dataset, see sklearn documentation.
    
    random_state : str
        To control the data shuffling process during training and test, see sklearn documentation.
        
    out_folder : Pandas DataFrame
        Directory where to store the comparative plots of training and testing.
        If None, the plot won't be saved on disk.
        If the folder does not exist yet, it will be created.
    


    Returns
    -------
    lin_model : sklearn.linear_model.LinearRegression
        LinearRegression object storing the best fit parameters and other goodness-of-fit quantities, see sklearn documentation.
        
    r2_score_train : float64
        R2 training score.
        
    r2_score_test : float64
        R2 test scores.
        
    rmse_train : float64
        RMSE of the training predictions.
        
    rmse_test : float64
        RMSE of the test predictions.
        
    """    

    def best_fit(XX,intercept_, coefs_):
        if XX.shape[1] != n_covariates:
            raise ValueError("XX features a number of columns greater than the number of covariates!")
        
        result = intercept_ * np.ones(dtype = XX.dtype, shape = (XX.shape[0],) )
        for i in range(XX.shape[1]):
            result+= XX[:,i]*coefs_[i]
        
        return result
    
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def residuals_plot(X_train, 
                       X_test, 
                       y_train,
                       y_test,
                       r2_score_train,
                       r2_score_test,
                       abs_residuals_train,
                       abs_residuals_test,
                       dep_variable_w_uom,
                       covariates_w_uom,
                       scaling=1e-3,
                       out_folder = out_folder):
        
        abs_res_label = r"$|\widehat{" + dep_variable_w_uom[0][0][1:-1] + " } - " + dep_variable_w_uom[0][0][1:-1] +"|$"
        
        
        if(covariates_w_uom[0][1]) == "()":
            uom_xlabel = ''
        else:
            uom_xlabel = covariates_w_uom[0][1]
        
        if(covariates_w_uom[1][1]) == "()":
            uom_ylabel = ''
        else:
            uom_ylabel = covariates_w_uom[1][1]
        
        xlabel = " ".join([covariates_w_uom[0][0] , uom_xlabel ])
        ylabel = " ".join([covariates_w_uom[1][0] , uom_ylabel ])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        im1 = ax1.scatter(X_train[:,0],
                    X_train[:,1],
                    c=y_train_pred,
                    marker='o',
                    s= np.abs(abs_residuals_train * scaling),
                    alpha = .5,
                    cmap = 'plasma',
                    label= " ".join([covariates_w_uom[0][0], "predicted"])
                    )

        im2 = ax2.scatter(X_test[:,0],
                    X_test[:,1],
                    c=y_test_pred,
                    marker='o',
                    s= np.abs(abs_residuals_test * scaling),
                    alpha = .5,
                    cmap = 'plasma',
                    label= " ".join([covariates_w_uom[1][0], "predicted"])
                    )

        fig.suptitle(' '.join(["Predictions and absolute residuals for", dep_variable_w_uom[0][0] ]) )
        # plt.title('  '.join(['$\Delta H_{c} = a_{0} + a_{1} \cdot H_{c}^{c} + a_{2} \cdot h$',  '(R^2 = {0:.2f})'.format(delta_hccalc_height_linreg_score)]))
        cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=1)
        cbar.set_label(dep_variable_w_uom[0][1][1:-1])
        cbar.ax.set_title(r'$\widehat{' + dep_variable_w_uom[0][0][1:-1] + '}$')

        ax1.set_title(r'Training residuals ($R^2 =$ {0:.2f})'.format(r2_score_train))
        ax1.legend(*im1.legend_elements(prop="sizes", num=3),
                   title='  '.join([abs_res_label, "(" + '$\\times$' + "{0:.0e} ".format(1/scaling) + dep_variable_w_uom[0][1][1:-1] + ')' ] ) )                 
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)

        ax2.set_title(r'Test residuals ($R^2 =$ {0:.2f})'.format(r2_score_test))
        ax2.legend(*im2.legend_elements(prop="sizes", num=3),
                   title='  '.join([abs_res_label, "(" + '$\\times$' + "{0:.0e} ".format(1/scaling) + dep_variable_w_uom[0][1][1:-1] + ')' ] ) )                 
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        if (out_folder is not None):
            fig_filename = '/'.join([out_folder,
                                     ''.join(['prediction_residuals_',
                                     datetime.utcnow().strftime('%F-%T.%f')[:-3],
                                     '.png'])
                                    ])
            os.makedirs(out_folder, exist_ok=True)
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()
   
    
    n_covariates = len(x)
    
    X=np.empty(shape=(y.size,n_covariates) , dtype=np.float64)
    for i in range(n_covariates):
        X[:,i] = x[i].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size, random_state=random_state)
    # X_train, X_test, y_train, y_test = X,X,y.values,y.values
    
    lin_model = LinearRegression().fit(X_train, y_train)
    intercept = lin_model.intercept_
    coefs = lin_model.coef_
    rank = lin_model.rank_
    singular_values = lin_model.singular_
    
    y_train_pred = best_fit(X_train, intercept, coefs)
    y_test_pred = best_fit(X_test,  intercept, coefs)
    
    r2_score_train = lin_model.score(X_train, y_train) 
    r2_score_test  = lin_model.score(X_test,  y_test) 
    
    residuals_train = y_train_pred - y_train
    residuals_test  = y_test_pred - y_test

    abs_residuals_train = np.abs(residuals_train)
    abs_residuals_test  = np.abs(residuals_test)

    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)
    
 
    if (n_covariates == 2):
        residuals_plot(X_train, 
                       X_test, 
                       y_train,
                       y_test,
                       r2_score_train,
                       r2_score_test,
                       abs_residuals_train,
                       abs_residuals_test,
                       dep_variable_w_uom,
                       covariates_w_uom,
                       scaling=1e-3,
                       out_folder=out_folder)
    
    print("=============================================")
    print(" ".join(["Summary Regression model for", dep_variable_w_uom[0][0]]) )
    print("Intercept = {}".format(lin_model.intercept_) )
    print("X0 coefficient= {}".format(lin_model.coef_[0]) )
    print("X1 coefficient = {}".format(lin_model.coef_[1]) )
    print("rank = {}".format(lin_model.rank_) )
    print("singular value X0 = {}".format(lin_model.singular_[0]) )
    print("singular value X1 = {}".format(lin_model.singular_[1]) )
    print("r2_score_train = {}".format(r2_score_train))
    print("r2_score_test = {}".format(r2_score_test))
    print("rmse_train = {}".format(rmse_train))
    print("rmse_test = {}".format(rmse_test))
    print("=============================================\n")

    return lin_model, r2_score_train, r2_score_test, rmse_train, rmse_test
