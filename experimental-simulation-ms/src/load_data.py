import re
import ms_aux
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

import pdb


compound_column_name = 'compound'
Ms_column_name = 'Ms (ampere/meter)'
tokenized_compound_col_name = 'tokenized-compound'

def load_oqmd(data_path):
    oqmd_stable_filename = '/'.join([data_path, 'oqmd_stable.csv'])

    oqmd_stable_columns = { 'material_id': str,
                             compound_column_name :str, # originally 'composition'
                             'spacegroup' : np.int32,
                             'nelements' : np.int32,
                             'nsites' : np.int32,
                             'energy_per_atom' : np.float64,
                             'formation_energy_per_atom' : np.float64,
                             'band_gap' : np.float64,
                             'volume_per_atom' : np.float64,
                             'magnetization_per_atom' : np.float64,
                             'atomic_volume_per_atom' : np.float64,
                             'volume_deviation' : np.float64,
                             'Ms' : np.float64,
                             }

    oqmd_stable_df = pd.read_csv(oqmd_stable_filename,
                                 sep=',',
                                 header=0,
                                 names = oqmd_stable_columns.keys(),
                                 dtype = oqmd_stable_columns,
                                 index_col=False)

    oqmd_stable_df['Ms (ampere/meter)_s'] = oqmd_stable_df['Ms'] / ms_aux.MU_NOUGHT
    
    oqmd_stable_df = oqmd_stable_df.rename(columns={'compound': 'composition'})
    
    return oqmd_stable_df

def load_literature(data_path):
    literature_filename = '/'.join([data_path, 'literature_values.csv'])
    literature_columns  = {compound_column_name : str, # originally 'compound'
                       'filename' : str,
                       'reference' : str,
                       'DOI' : str,
                       'la' : np.float64,
                       'ce' : np.float64,
                       'pr' : np.float64,
                       'nd' : np.float64,
                       'sm' : np.float64,
                       'gd' : np.float64,
                       'tb' : np.float64,
                       'dy' : np.float64,
                       'mn' : np.float64,
                       'ni' : np.float64,
                       'fe' : np.float64,
                       'co' : np.float64,
                       'b'  : np.float64,
                       'c'  : np.float64,
                       'lattice_a'  : np.float64,
                       'lattice_c'  : np.float64,
                       'lattice_v'  : np.float64,
                       'nfu'  : np.float64, # number of atoms per formula unit
                       'muB'  : np.float64,   
                       'mu0Ms'  : np.float64, # spontaneous magnetization
                       'mu0Ha'  : np.float64, # anisotropy field
                       'Tc'  : np.float64 }

    literature_df = pd.read_csv(literature_filename,
                           sep=';',
                           header=0,
                           names = literature_columns.keys(),
                           dtype = literature_columns,
                           index_col=False)

    literature_df['Ms (ampere/meter)_e'] = literature_df['mu0Ms'] / ms_aux.MU_NOUGHT

    literature_df = literature_df.rename(columns={'compound': 'composition'})
    
    return literature_df

def load_mtc(data_path):
    mtcsum_filename =''.join([data_path, 'm-tcsum.csv'])
    mtcsum_columns = {compound_column_name : str, # originally 'System'
                     'M_s/theo [T]' : np.float64 ,
                     'M_s/exp [T]' : np.float64,
                     'T_Ctheo MF [K]' : np.float64,
                      'T_C/theo [K]' : np.float64,
                      'T_C_exp [K]' : np.float64,
                      'is_RE_1=y' : np.int8, # change into boolean
                      'stable' : str, # change into boolean
                      'source' : str
                     }
    mtcsum_df = pd.read_csv(mtcsum_filename,
                           sep=',',
                           header=0,
                           names = mtcsum_columns.keys(),
                           dtype = mtcsum_columns,
                           index_col=False)

    mtcsum_df = mtcsum_df.rename(columns={'compound': 'composition'})
    # NOTE: possibly add line converting to A/m. Currently not used.
    return mtcsum_df


def load_mtc_nur(data_path):
    mtcsum_nur_new_filename ='/'.join([data_path, 'm-tcsum_nur_new.csv'])

    mtcsum_nur_new_columns = {compound_column_name : str, # originally 'System'
                              'M0_1' : np.float64,
                              's_1' : np.float64,
                              'M0_2' : np.float64,
                              's_2' : np.float64,
                              'M_s/theo [T] 0K' : np.float64 ,
                              'M_s/exp [T] 5K' : np.float64,
                              'M_s/exp [T] 300K' : np.float64,
                              'T_Ctheo MF [K]' : np.float64,
                              'T_C/theo [K]' : np.float64,
                              'T_C_exp [K]' : np.float64,
                              'is_RE_1=y' : np.float64, # change into boolean. missong value for SmFe11Ti
                              'stable' : str, # change into boolean
                              'source' : str
                            }

    mtcsum_nur_new_df = pd.read_csv(mtcsum_nur_new_filename,
                           sep=',',
                           header=0,
                           names = mtcsum_nur_new_columns.keys(),
                           dtype = mtcsum_nur_new_columns,
                           index_col=False)

    mtcsum_nur_new_df = mtcsum_nur_new_df.rename(columns={'compound': 'composition'})
    # NOTE: possibly add line converting to A/m. Currently not used.
    
    return mtcsum_nur_new_df


def load_bhandari_i(data_path):
    bhandari_i_filename = '/'.join([data_path, 'Bhandari_I_exp.csv'])
    bhandari_i_columns = {
                                compound_column_name : str, # originally 'material'
                                'Ms_exp (MA/m)'  : np.float64 ,
                                'Aex_exp (pJ/m)' : np.float64 ,
                                'Ku (MJ/m3)' : np.float64,
                                'Hc_exp (T)'     : np.float64,
                                'Hc_mumax3 (T)'     : np.float64,
                                'cHc (T)'     : np.float64,
                                'Hc_diff (T)'     : np.float64
                           }
    bhandari_i_df = pd.read_csv(bhandari_i_filename,
                           sep='|',
                           header=0,
                           names = bhandari_i_columns.keys(),
                           dtype = bhandari_i_columns,
                           index_col=False)

    bhandari_i_df['Ms (ampere/meter)_e'] = bhandari_i_df['Ms_exp (MA/m)'] * 1000000
    bhandari_i_df = bhandari_i_df.rename(columns={'compound': 'composition'})
    
    return bhandari_i_df

def load_bhandari_xii(data_path):
    bhandari_xii_filename = '/'.join([data_path, 'Bhandari_XII_sim.csv'])
    bhandari_xii_columns = {
                                compound_column_name : str, # originally 'material'
                                'Ms (A/m)'   : np.float64 ,
                                'Aex (pJ/m)' : np.float64 ,
                                'Ku (MJ/m3)' : np.float64
                           }
    bhandari_xii_df = pd.read_csv(bhandari_xii_filename,
                           sep=';',
                           header=0,
                           names = bhandari_xii_columns.keys(),
                           dtype = bhandari_xii_columns,
                           index_col=False)


    bhandari_xii_df['Ms (ampere/meter)_s'] = bhandari_xii_df['Ms (A/m)']
    bhandari_xii_df = bhandari_xii_df.rename(columns={'compound': 'composition'})
    return bhandari_xii_df

def load_bhandari_xiii(data_path):
    bhandari_xiii_filename = '/'.join([data_path, 'Bhandari_XIII_exp.csv'])
    bhandari_xiii_columns = {
                                compound_column_name : str, # originally 'material'
                                'Ms (MA/m)'  : np.float64 ,
                                'Aex (pJ/m)' : np.float64 ,
                                'Ku (MJ/m3)' : np.float64,
                                'Hc (T)'     : np.float64
                           }
    bhandari_xiii_df = pd.read_csv(bhandari_xiii_filename,
                           sep='|',
                           header=0,
                           names = bhandari_xiii_columns.keys(),
                           dtype = bhandari_xiii_columns,
                           index_col=False)

    bhandari_xiii_df['Ms (ampere/meter)_e'] = bhandari_xiii_df['Ms (MA/m)'] * 1000000
    bhandari_xiii_df = bhandari_xiii_df.rename(columns={'compound': 'composition'})
    return bhandari_xiii_df

def load_magnetic_materials_exp(data_path):
    mp_fm_dedup_exp_filename = '/'.join([data_path, 'mp_fm_dedup_exp_data.csv'])
    mp_fm_dedup_exp_columns = {
                                compound_column_name : str, # originally 'formula_pretty'
                                'is_stable' : np.bool_ ,
                                'is_magnetic' : np.bool_ ,
                                'volume' : np.float64, 
                                'density' : np.float64,
                                'density_atomic' : np.float64,
                                'total_magnetization' : np.float64, 
                                'total_magnetization_normalized_vol' : np.float64,
                                'total_magnetization_normalized_formula_units' : np.float64,
                                'source' : str
                                }

    mp_fm_dedup_exp_df = pd.read_csv(mp_fm_dedup_exp_filename,
                           sep=',',
                           header=0,
                           names = mp_fm_dedup_exp_columns.keys(),
                           dtype = mp_fm_dedup_exp_columns,
                           index_col=False,
                           usecols = [7,21,36,58,59,60,65,66,67,71])


    mp_fm_dedup_exp_df['Ms (ampere/meter)_e'] = mp_fm_dedup_exp_df['total_magnetization_normalized_vol'] * ms_aux.MU_B / (ms_aux.AA*ms_aux.AA*ms_aux.AA)
    
    mp_fm_dedup_exp_df = mp_fm_dedup_exp_df.rename(columns={'compound': 'composition'})

    return mp_fm_dedup_exp_df

def load_magnetic_materials_sim(data_path):
    mp_fm_dedup_sim_filename = '/'.join([data_path, 'mp_fm_dedup_sim_data.csv'])
    mp_fm_dedup_sim_columns = {
                                compound_column_name : str, # originally 'formula_pretty'
                                'is_stable' : np.bool_ ,
                                'is_magnetic' : np.bool_ ,
                                'volume' : np.float64, 
                                'density' : np.float64,
                                'density_atomic' : np.float64,
                                'total_magnetization' : np.float64, 
                                'total_magnetization_normalized_vol' : np.float64,
                                'total_magnetization_normalized_formula_units' : np.float64,
                                'source' : str
                                }

    mp_fm_dedup_sim_df = pd.read_csv(mp_fm_dedup_sim_filename,
                           sep=',',
                           header=0,
                           names = mp_fm_dedup_sim_columns.keys(),
                           dtype = mp_fm_dedup_sim_columns,
                           index_col=False,
                           usecols = [7,21,36,58,59,60,65,66,67,71])



    mp_fm_dedup_sim_df['Ms (ampere/meter)_s'] = mp_fm_dedup_sim_df['total_magnetization_normalized_vol'] * ms_aux.MU_B / (ms_aux.AA*ms_aux.AA*ms_aux.AA)    
    
    mp_fm_dedup_sim_df = mp_fm_dedup_sim_df.rename(columns={'compound': 'composition'})
    return mp_fm_dedup_sim_df


LOADERS = {'oqmd': load_oqmd,
           'literature': load_literature,
           'mtc': load_mtc,
           'mtc_nur': load_mtc_nur,
           'bhandari_i': load_bhandari_i,
           'bhandari_xii': load_bhandari_xii,
           'bhandari_xiii': load_bhandari_xiii,
           'magnetic_materials_exp': load_magnetic_materials_exp,
           'magnetic_materials_sim': load_magnetic_materials_sim}

def load_data(*datasets, data_path):
    loaded_data = {}
            
    for name in datasets:
        
        loader = LOADERS[name]
            
        if not loader:
            print(f'No loader found for {name}.')
            continue
        

        print(f'loading {name} ...')
        df = loader(data_path)
        loaded_data[name] = df

    return loaded_data