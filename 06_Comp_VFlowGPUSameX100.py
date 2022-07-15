import os
import sys
import time
import logging
import multiprocessing
from pprint import pprint

import argparse
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as spt
from tqdm import tqdm
from tabulate import tabulate

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# os.environ["VEGASFLOW_LOG_LEVEL"] = "0"

import vegasflow as vflow
from vegasflow import configflow

import tensorflow as tf
import tensorflow_probability as tfp

# tf.autograph.set_verbosity(0)
tf.keras.backend.set_floatx('float64')
print(f'(is_gpu_available: {tf.test.is_gpu_available()})')

# These parameters are set manually in configflow.py :

# BINS_MAX = 50
# ALPHA = 0.5  # changed from 1.5 to = vegas default
# BETA = 0.75  # Vegas+
# TECH_CUT = 1e-8

# Events Limit limits how many events are done in one single run of the event_loop
# MAX_EVENTS_LIMIT = int(2e6)  # changed from int(1e6) to utilize RAM

DEFAULT_ACTIVE_DEVICES = ['GPU']

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--func',  type=str, help="Function to integrate")
parser.add_argument('-r', '--run_num', type=int, help="Run number", default=1)
args = parser.parse_args()

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
formatter = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)

PRECISION = 1e-4

param_run = args.run_num
param_func = args.func
param_seed = param_run
param_filename = f'result/06_Comp_VFlowGPUSameX100/{param_func}_run{param_run}_fp'

param_ndims_lst = [2,4,6,8][::-1]
param_nitn = 15
param_nitn_pre = 5
param_neval = int(1e9) # 1e7


###############################################################

if param_func == 'f1':

    m = 0.5
    sigma = 0.01
    norm_dist = spt.norm(loc=m, scale=sigma)
    dim2target_dict = {
        2: (norm_dist.cdf(1) - norm_dist.cdf(0)) ** 2,
        4: (norm_dist.cdf(1) - norm_dist.cdf(0)) ** 4,
        6: (norm_dist.cdf(1) - norm_dist.cdf(0)) ** 6,
        8: (norm_dist.cdf(1) - norm_dist.cdf(0)) ** 8,
    }

    integ_bounds = [0, 1]

    m_tf = tf.constant(m, dtype=tf.float64)
    alpha_tf = tf.constant(sigma * np.sqrt(2), dtype=tf.float64)
    pi_tf = tf.constant(np.pi, dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,2), dtype=tf.float64)])
    def f1_d2_tf(x):
        pre = 1.0 / (alpha_tf * tf.sqrt(pi_tf)) ** 2
        exponent = -1 * tf.reduce_sum((x - m_tf) ** 2, axis=-1) / alpha_tf ** 2
        return pre * tf.exp(exponent) / dim2target_dict[2]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,4), dtype=tf.float64)])
    def f1_d4_tf(x):
        pre = 1.0 / (alpha_tf * tf.sqrt(pi_tf)) ** 4
        exponent = -1 * tf.reduce_sum((x - m_tf) ** 2, axis=-1) / alpha_tf ** 2
        return pre * tf.exp(exponent) / dim2target_dict[4]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,6), dtype=tf.float64)])
    def f1_d6_tf(x):
        pre = 1.0 / (alpha_tf * tf.sqrt(pi_tf)) ** 6
        exponent = -1 * tf.reduce_sum((x - m_tf) ** 2, axis=-1) / alpha_tf ** 2
        return pre * tf.exp(exponent) / dim2target_dict[6]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,8), dtype=tf.float64)])
    def f1_d8_tf(x):
        pre = 1.0 / (alpha_tf * tf.sqrt(pi_tf)) ** 8
        exponent = -1 * tf.reduce_sum((x - m_tf) ** 2, axis=-1) / alpha_tf ** 2
        return pre * tf.exp(exponent) / dim2target_dict[8]

    dim2func_dict = {
        2: f1_d2_tf,
        4: f1_d4_tf,
        6: f1_d6_tf,
        8: f1_d8_tf,
    }

elif param_func == 'f2':

    norm_m1, norm_m2, norm_m3 = 0.33, 0.47, 0.67
    sigma = 0.01
    norm_dist_1 = spt.norm(loc=norm_m1, scale=sigma)
    norm_dist_2 = spt.norm(loc=norm_m2, scale=sigma)
    norm_dist_3 = spt.norm(loc=norm_m3, scale=sigma)

    norm_triple = norm_dist_1.cdf(1) - norm_dist_1.cdf(0) \
                + norm_dist_2.cdf(1) - norm_dist_2.cdf(0) \
                + norm_dist_3.cdf(1) - norm_dist_3.cdf(0)
    norm_triple /= 3

    dim2target_dict = {
        2: norm_triple ** 2,
        4: norm_triple ** 4,
        6: norm_triple ** 6,
        8: norm_triple ** 8,
    }

    integ_bounds = [0, 1]

    ms_tf = tf.constant([norm_m1, norm_m2, norm_m3], dtype=tf.float64)
    alpha_tf = tf.constant(sigma * np.sqrt(2), dtype=tf.float64)
    pi_tf = tf.constant(np.pi, dtype=tf.float64)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,2), dtype=tf.float64)])
    def f2_d2_tf(x):
        pre = 1. / (alpha_tf * tf.sqrt(pi_tf)) ** 2
        exponent1 = -1. * tf.reduce_sum(((x - ms_tf[0]) ** 2), axis=-1) / alpha_tf ** 2
        exponent2 = -1. * tf.reduce_sum(((x - ms_tf[1]) ** 2), axis=-1) / alpha_tf ** 2
        exponent3 = -1. * tf.reduce_sum(((x - ms_tf[2]) ** 2), axis=-1) / alpha_tf ** 2
        res = pre / 3 * (tf.exp(exponent1) + tf.exp(exponent2) + tf.exp(exponent3))
        return res / dim2target_dict[2]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,4), dtype=tf.float64)])
    def f2_d4_tf(x):
        pre = 1. / (alpha_tf * tf.sqrt(pi_tf)) ** 4
        exponent1 = -1. * tf.reduce_sum(((x - ms_tf[0]) ** 2), axis=-1) / alpha_tf ** 2
        exponent2 = -1. * tf.reduce_sum(((x - ms_tf[1]) ** 2), axis=-1) / alpha_tf ** 2
        exponent3 = -1. * tf.reduce_sum(((x - ms_tf[2]) ** 2), axis=-1) / alpha_tf ** 2
        res = pre / 3 * (tf.exp(exponent1) + tf.exp(exponent2) + tf.exp(exponent3))
        return res / dim2target_dict[4]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,6), dtype=tf.float64)])
    def f2_d6_tf(x):
        pre = 1. / (alpha_tf * tf.sqrt(pi_tf)) ** 6
        exponent1 = -1. * tf.reduce_sum(((x - ms_tf[0]) ** 2), axis=-1) / alpha_tf ** 2
        exponent2 = -1. * tf.reduce_sum(((x - ms_tf[1]) ** 2), axis=-1) / alpha_tf ** 2
        exponent3 = -1. * tf.reduce_sum(((x - ms_tf[2]) ** 2), axis=-1) / alpha_tf ** 2
        res = pre / 3 * (tf.exp(exponent1) + tf.exp(exponent2) + tf.exp(exponent3))
        return res / dim2target_dict[6]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,8), dtype=tf.float64)])
    def f2_d8_tf(x):
        pre = 1. / (alpha_tf * tf.sqrt(pi_tf)) ** 8
        exponent1 = -1. * tf.reduce_sum(((x - ms_tf[0]) ** 2), axis=-1) / alpha_tf ** 2
        exponent2 = -1. * tf.reduce_sum(((x - ms_tf[1]) ** 2), axis=-1) / alpha_tf ** 2
        exponent3 = -1. * tf.reduce_sum(((x - ms_tf[2]) ** 2), axis=-1) / alpha_tf ** 2
        res = pre / 3 * (tf.exp(exponent1) + tf.exp(exponent2) + tf.exp(exponent3))
        return res / dim2target_dict[8]

    dim2func_dict = {
        2: f2_d2_tf,
        4: f2_d4_tf,
        6: f2_d6_tf,
        8: f2_d8_tf,
    }

elif param_func == 'f3':

    dim2target_dict = {
        # from 0 to 10
        2: 8 * np.cos(5) * np.sin(5) ** 3,    
        4: 32 * (np.cos(5) + np.cos(15)) * np.sin(5) ** 5,
        6: 128 * (np.cos(5) + np.cos(15) + np.cos(25)) * np.sin(5) ** 7,
        8: 128 * (np.cos(35) - np.cos(45)) * np.sin(5) ** 7,
    }

    integ_bounds = [0, 10]
    integ_size = integ_bounds[1] - integ_bounds[0]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,2), dtype=tf.float64)])
    def f3_d2_tf(x):
       return tf.sin(tf.reduce_sum(x * integ_size, axis=-1)) * integ_size ** 2 / dim2target_dict[2]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,4), dtype=tf.float64)])
    def f3_d4_tf(x):
       return tf.sin(tf.reduce_sum(x * integ_size, axis=-1)) * integ_size ** 4 / dim2target_dict[4]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,6), dtype=tf.float64)])
    def f3_d6_tf(x):
       return tf.sin(tf.reduce_sum(x * integ_size, axis=-1)) * integ_size ** 6 / dim2target_dict[6]

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,8), dtype=tf.float64)])
    def f3_d8_tf(x):
       return tf.sin(tf.reduce_sum(x * integ_size, axis=-1)) * integ_size ** 8 / dim2target_dict[8]

    dim2func_dict = {
        2: f3_d2_tf,
        4: f3_d4_tf,
        6: f3_d6_tf,
        8: f3_d8_tf,
    }

else:
    assert False, f'wrong func param value: {args.func}'

###############################################################

def compute_variance_weighted_result(means, sdevs):
    """ Computes weighted mean and stddev of given means and
        stddevs arrays, using Inverse-variance weighting
    """
    assert means.size == sdevs.size
    assert means.shape == sdevs.shape
    var = 1./np.sum(1./sdevs**2, axis=-1)
    mean = np.sum(means/(sdevs**2), axis=-1)
    mean *= var
    return mean, np.sqrt(var)

def compute_rel_unc(mean_a, unc_a, mean_b, unc_b):
    """Relative uncertainty"""
    ret = np.abs(mean_a - mean_b)
    sqr = np.sqrt(unc_a**2 + unc_b**2)
    ret = ret / sqr
    return ret

def compute_chi2_Q(w_mean, means, sdevs):
    """"""
    assert means.size == sdevs.size
    assert means.shape == sdevs.shape 
    chi2 = np.sum(((means - w_mean) / sdevs) ** 2)
    dof = means.size - 1
    Q = 1 - spt.chi2.cdf(chi2, dof)
    return chi2, Q

###############################################################


if __name__ == '__main__':

    print(f'param_run: {param_run}')
    print(f'param_func: {param_func}')

    print(f'param_ndims_lst: {param_ndims_lst}')
    print(f'param_nitn: {param_nitn}')
    print(f'param_nitn_pre: {param_nitn_pre}')
    print(f'param_neval: {param_neval:.0E}')

    # print(f'param_neval_frac: {param_neval_frac}')
    # print(f'param_nhcube_batch: {param_nhcube_batch}')
    
    # print('#hcubes on various dims:')
    # temp_hcube_dict = {}
    # for d in param_ndims_lst:
    #     temp_per_dim = int(np.power(param_neval*(1-param_neval_frac)/2, 1/d))
    #     temp_total = np.power(temp_per_dim, d)
    #     temp_hcube_dict[d] = f'{temp_per_dim} [{temp_total} total]'
    # pprint(temp_hcube_dict)
    # print('max possible neval in batch: '
    #       f'{int(param_neval_frac*param_neval+(1-param_neval_frac)*2)}')
    print('---')

    np.random.seed(param_seed)
    tf.random.set_seed(param_seed)

    result_means = []
    result_sdevs = []
    result_rels = []
    result_times = []
    result_times_pre = []
    # result_ncalls = []
    # result_avg_nevals = []
    result_chi2_values = []
    result_Q_values = []
    # result_nitn_used = []
    result_run_nums = []


    for ndims in param_ndims_lst:
        logger.info(f'ndims={ndims}')

        temp_means = []
        temp_sdevs = []
        # temp_sum_evals = []
        
        integrand = dim2func_dict[ndims]
        vegas_integ = vflow.VegasFlow(
            ndims,
            param_neval,
            verbose=True,
            list_devices=DEFAULT_ACTIVE_DEVICES,
            )
        vegas_integ.compile(integrand, compilable=True)
        print(vegas_integ.devices)

        time_a = time.time()
        for i in tqdm(range(param_nitn_pre)):
            vegas_integ.run_integration(1)
        pre_time = round(time.time() - time_a, 3)

        #w_sdev = PRECISION + 1
        # nitn_used = 0
        nitn_true = param_nitn-param_nitn_pre
        true_time = 0
        for i in tqdm(range(nitn_true)):
            # if w_sdev < PRECISION * dim2target_dict[ndims]:
            #     # early stopping
            #     break
            temp_time = time.time()
            temp_mean, temp_sdev = vegas_integ.run_integration(1)
            true_time += time.time() - temp_time
            temp_means.append(temp_mean)
            temp_sdevs.append(temp_sdev)
            # temp_sum_evals.append(current_result.sum_neval)
            # _, w_sdev = compute_variance_weighted_result(np.array(temp_means),
            #                                              np.array(temp_sdevs))
            # nitn_used += 1
            
        total_time = round(true_time + pre_time, 3)

        temp_means = np.array(temp_means)
        temp_sdevs = np.array(temp_sdevs)
        temp_foldername = os.path.join(*param_filename.split('/')[:-1])
        os.makedirs(temp_foldername, exist_ok=True)
        np.save(param_filename+f'_d{ndims}_means_sdevs', np.stack([temp_means, temp_sdevs]))
        # temp_sum_evals = np.array(temp_sum_evals)

        w_mean, w_sdev = compute_variance_weighted_result(temp_means, temp_sdevs)
        # avg_neval = np.mean(temp_sum_evals)
        rel_unc = compute_rel_unc(w_mean, w_sdev, 1, 0)
        chi2, Q = compute_chi2_Q(w_mean, temp_means, temp_sdevs)
        # ncalls = integrand.ncalls

        result_means.append(w_mean)
        result_sdevs.append(w_sdev)
        result_rels.append(rel_unc)
        result_times.append(total_time)
        result_times_pre.append(pre_time)
        # result_ncalls.append(ncalls)
        # result_avg_nevals.append(avg_neval)
        result_chi2_values.append(chi2)
        result_Q_values.append(Q)
        # result_nitn_used.append(nitn_used)
        result_run_nums.append(param_run)

    temp_df = pd.DataFrame({
        'func':         [param_func] * len(param_ndims_lst),
        'ndim':         param_ndims_lst,
        'nitn':         [param_nitn] * len(param_ndims_lst),
        # 'nitn_u':       result_nitn_used,
        'neval':        [param_neval] * len(param_ndims_lst),
        'res_mean':     result_means,
        'res_err':      result_sdevs,
        'res_rel':      result_rels,
        'total_time':   result_times,
        'pre_time':     result_times_pre,
        # 'ncall':        result_ncalls,
        # 'avg_neval':    result_avg_nevals,
        'chi2':         result_chi2_values,
        'Q':            result_Q_values,
        'run_num':      result_run_nums,
    })

    # save
    temp_foldername = os.path.join(*param_filename.split('/')[:-1])
    os.makedirs(temp_foldername, exist_ok=True)
    temp_df.to_csv(param_filename, index=False)

    # display
    temp_df = temp_df[['func', 'ndim', 'nitn', 'neval',  # 'nitn_u', 
                       'res_mean', 'res_err', 'res_rel', 'total_time', # 'ncall',
                       'chi2', 'Q', 'run_num']]
    print(tabulate(temp_df, headers=temp_df.columns))
