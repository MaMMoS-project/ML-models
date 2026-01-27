import pandas as pd


import ms_aux
import coercivity_ml as cml

# SET DATA AND PLOT FOLDERS, SPECIFY CSV FILEPATH
DATA_FOLDER = '../data'
PLOTS_FOLDER = '../plots'
fmdh_filename = '/'.join([DATA_FOLDER, 'full_micromagnetic_with_height.csv'])
oxide_filename = '/'.join([DATA_FOLDER, 'NdCeFeB_2-7_20251021_ceriumOxide_phase_fractions_mammos_entities.csv'])

# READ CSV
fmdh_df = cml.create_joined_df(fmdh_filename, oxide_filename)



# EXTRACT VALID DATA, ADD NEW FEATURES. All relevant data tb analysed is now stored in fmdh_df_valid.
fmdh_df_valid = cml.preprocess_and_engineering_df(fmdh_df)

# PLOT RELEVANT QUANTITIES
cml.plot_Hc_exp_and_calc(fmdh_df_valid, PLOTS_FOLDER)
cml.plot_delta_Hc(fmdh_df_valid, PLOTS_FOLDER)
cml.plot_ce_nd_content(fmdh_df_valid, PLOTS_FOLDER)
cml.plot_height(fmdh_df_valid, PLOTS_FOLDER)

# COMPUTE CORRELATION MATRIX OF SPECIFIED QUANTITIES AND STORE IT AS A pd.DataFrame
col_corr_matrix = [ # specify desired features
        'x_pos',
        'y_pos',
        'a',
        'c',
        'volume',
        'Hc_measured',
        'main_phase_composition',
        'Ms',
        'A',
        'K1',
        'Hc_calculated',
        # 'Mr_calculated',
        # 'BHmax_calculated',
        'anisotropy_field',
        'measured_height',
        'Neff',
        'Nd',
        'Ce',
        'Hc-exp-minus-cal',
        'alpha'
    ]
corr = cml.get_corr_matrix(fmdh_df, col_corr_matrix) # get the matrix as a pd.DataFrame

corr.style.background_gradient(cmap='coolwarm') # visualization of correlation matrix (only works within JN/IPython)
(corr
 .style
 .background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
 .highlight_null(color='#f1f1f1')
 .format(precision=2))


# FIT BIVARIATE LINEAR MODEL \Delta Hc = u0 + u1 Hc_calculated + u2 height  
x_delta = [ # DATA: covariates of \Delta Hc
           fmdh_df_valid['Hc_calculated'],
           fmdh_df_valid['measured_height']
          ]

dep_var_delta = { 0 : ['$\Delta H_{c}$' , '(A/m)'] } # LABELS: dependent variable name (\Delta Hc) + unit of measure

covariates_delta = { 0 : ["$H_{c}^{sim}$", '(A/m)'], # LABELS: covariates names + unit of measure
                     1 : ['$h$'          , '(nm)'] }

delta_linreg,\
r2_delta_train,\
r2_delta_test,\
rmse_delta_train,\
rmse_delta_test = \
    cml.run_linear_model(x_delta, # covariates
                         fmdh_df_valid['Hc-exp-minus-cal'], # dependent variable
                         dep_var_delta, # dependent variable names + units of measure
                         covariates_delta, # independent variables names + units of measure
                         out_folder = PLOTS_FOLDER) # artifact folder

# FIT BIVARIATE LINEAR MODEL Hc_exp = w0 + w1 height + w2 [Ce]  
x_Hc_exp = [
            fmdh_df_valid['measured_height'],
            fmdh_df_valid['Ce']
           ]


dep_var_hc_exp = { 0 : ['$H_{c}^{exp}$', '(A/m)'] }

covariates_hc_exp = { 0 : ['$h$' , '(nm)'],
                      1 : ['[Ce]', '(1)'] }

Hc_exp_linreg,\
r2_Hc_exp_train,\
r2_Hc_exp_test,\
rmse_Hc_exp_train,\
rmse_Hc_exp_test = \
    cml.run_linear_model(x_Hc_exp,
                         fmdh_df_valid['Hc_measured'],
                         dep_var_hc_exp ,
                         covariates_hc_exp,
                         out_folder = PLOTS_FOLDER)
