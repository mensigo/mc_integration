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

import vegas

parser = argparse.ArgumentParser()
parser.add_argument("-f", '--func',  type=str, help="Function to integrate")
parser.add_argument("-r", '--run_num', type=int, help="Run number", default=1)
args = parser.parse_args()

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
formatter = logging.Formatter('%(asctime)s  %(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)

NPROC = 6
PRECISION = 1e-4

param_run = args.run_num
param_func = args.func
param_seed = param_run
param_filename = f'result/06_Comp_VegasNX10/{param_func}_run{param_run}_fp'

param_ndims_lst = [2,4,6,8][::-1]
param_nitn = 15 * 10
param_nitn_pre = 5
param_neval = int(1e7) # 1e8
param_neval_frac = 0.5
param_nhcube_batch = int(1e4)  # 1e6
param_maxinc_axis = 50
param_max_mem = 2e9

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

    alpha = sigma * np.sqrt(2)
    pi = np.pi

    # @vegas.batchintegrand
    def f1_d2_np(x):
        pre = 1.0 / (alpha * np.sqrt(pi)) ** 2
        exponent = -1 * np.sum((x - .5) ** 2, axis=-1) / alpha ** 2
        return pre * np.exp(exponent) / dim2target_dict[2]

    # @vegas.batchintegrand
    def f1_d4_np(x):
        pre = 1.0 / (alpha * np.sqrt(pi)) ** 4
        exponent = -1 * np.sum((x - .5) ** 2, axis=-1) / alpha ** 2
        return pre * np.exp(exponent) / dim2target_dict[4]

    # @vegas.batchintegrand
    def f1_d6_np(x):
        pre = 1.0 / (alpha * np.sqrt(pi)) ** 6
        exponent = -1 * np.sum((x - .5) ** 2, axis=-1) / alpha ** 2
        return pre * np.exp(exponent) / dim2target_dict[6]

    # @vegas.batchintegrand
    def f1_d8_np(x):
        pre = 1.0 / (alpha * np.sqrt(pi)) ** 8
        exponent = -1 * np.sum((x - .5) ** 2, axis=-1) / alpha ** 2
        return pre * np.exp(exponent) / dim2target_dict[8]

    dim2func_dict = {
        2: f1_d2_np,
        4: f1_d4_np,
        6: f1_d6_np,
        8: f1_d8_np,
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

    alpha = sigma * np.sqrt(2)
    pi = np.pi

    # @vegas.batchintegrand
    def f2_d2_np(x):
        pre = 1. / (alpha * np.sqrt(pi)) ** 2
        exponent1 = -1. * np.sum(((x - norm_m1) ** 2), axis=-1) / alpha ** 2
        exponent2 = -1. * np.sum(((x - norm_m2) ** 2), axis=-1) / alpha ** 2
        exponent3 = -1. * np.sum(((x - norm_m3) ** 2), axis=-1) / alpha ** 2
        return pre / 3 * (np.exp(exponent1) + np.exp(exponent2) + np.exp(exponent3)) / dim2target_dict[2]

    # @vegas.batchintegrand
    def f2_d4_np(x):
        pre = 1. / (alpha * np.sqrt(pi)) ** 4
        exponent1 = -1. * np.sum(((x - norm_m1) ** 2), axis=-1) / alpha ** 2
        exponent2 = -1. * np.sum(((x - norm_m2) ** 2), axis=-1) / alpha ** 2
        exponent3 = -1. * np.sum(((x - norm_m3) ** 2), axis=-1) / alpha ** 2
        return pre / 3 * (np.exp(exponent1) + np.exp(exponent2) + np.exp(exponent3)) / dim2target_dict[4]

    # @vegas.batchintegrand
    def f2_d6_np(x):
        pre = 1. / (alpha * np.sqrt(pi)) ** 6
        exponent1 = -1. * np.sum(((x - norm_m1) ** 2), axis=-1) / alpha ** 2
        exponent2 = -1. * np.sum(((x - norm_m2) ** 2), axis=-1) / alpha ** 2
        exponent3 = -1. * np.sum(((x - norm_m3) ** 2), axis=-1) / alpha ** 2
        return pre / 3 * (np.exp(exponent1) + np.exp(exponent2) + np.exp(exponent3)) / dim2target_dict[6]

    # @vegas.batchintegrand
    def f2_d8_np(x):
        pre = 1. / (alpha * np.sqrt(pi)) ** 8
        exponent1 = -1. * np.sum(((x - norm_m1) ** 2), axis=-1) / alpha ** 2
        exponent2 = -1. * np.sum(((x - norm_m2) ** 2), axis=-1) / alpha ** 2
        exponent3 = -1. * np.sum(((x - norm_m3) ** 2), axis=-1) / alpha ** 2
        return pre / 3 * (np.exp(exponent1) + np.exp(exponent2) + np.exp(exponent3)) / dim2target_dict[8]

    dim2func_dict = {
        2: f2_d2_np,
        4: f2_d4_np,
        6: f2_d6_np,
        8: f2_d8_np,
    }

elif param_func == 'f3':

    dim2target_dict = {
        # from 0 to 10
        2: 8 * np.cos(5) * np.sin(5) ** 3,    
        4: 32 * (np.cos(5) + np.cos(15)) * np.sin(5) ** 5,
        6: 128 * (np.cos(5) + np.cos(15) + np.cos(25)) * np.sin(5) ** 7,
        8: 128 * (np.cos(35) - np.cos(45)) * np.sin(5) ** 7,
    }

    integ_bounds = [0, 1]  # new normalized bounds, old are [0, 10]
    integ_size = 10

    #@vegas.batchintegrand
    def f3_d2_np(x):
       return np.sin(np.sum(x * integ_size, axis=-1)) * integ_size ** 2  / dim2target_dict[2]

    #@vegas.batchintegrand
    def f3_d4_np(x):
       return np.sin(np.sum(x * integ_size, axis=-1)) * integ_size ** 4  / dim2target_dict[4]

    #@vegas.batchintegrand
    def f3_d6_np(x):
       return np.sin(np.sum(x * integ_size, axis=-1)) * integ_size ** 6  / dim2target_dict[6]

    #@vegas.batchintegrand
    def f3_d8_np(x):
       return np.sin(np.sum(x * integ_size, axis=-1)) * integ_size ** 8  / dim2target_dict[8]

    dim2func_dict = {
        2: f3_d2_np,
        4: f3_d4_np,
        6: f3_d6_np,
        8: f3_d8_np,
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

class parallel_integrand(vegas.BatchIntegrand):
    """ Convert (batch) integrand into multiprocessor integrand.
        Integrand should return a numpy array.
    """
    def __init__(self, fcn, pool, nproc):
        " Save integrand; create pool of nproc processes. "
        self.fcn = fcn
        self.nproc = nproc
        self.pool = pool
        self.ncalls = 0
    # def __del__(self):
    #     " Standard cleanup. "
    #     self.pool.close()
    #     self.pool.join()
    def __call__(self, x):
        " Divide x into self.nproc chunks, feeding one to each process. "
        self.ncalls += 1
        nx = x.shape[0] // self.nproc + 1
        results = self.pool.map(
            self.fcn,
            [x[i*nx : (i+1)*nx] for i in range(self.nproc)],
            None,
            )
        return np.concatenate(results)

###############################################################


if __name__ == '__main__':

    print(f'param_run: {param_run}')
    print(f'param_func: {param_func}')

    print(f'param_ndims_lst: {param_ndims_lst}')
    print(f'param_nitn: {param_nitn}')
    print(f'param_nitn_pre: {param_nitn_pre}')
    print(f'param_neval: {param_neval:.0E}')

    print(f'param_neval_frac: {param_neval_frac}')
    print(f'param_nhcube_batch: {param_nhcube_batch}')
    
    print('#hcubes on various dims:')
    temp_hcube_dict = {}
    for d in param_ndims_lst:
        temp_per_dim = int(np.power(param_neval*(1-param_neval_frac)/2, 1/d))
        temp_total = np.power(temp_per_dim, d)
        temp_hcube_dict[d] = f'{temp_per_dim} [{temp_total} total]'
    pprint(temp_hcube_dict)
    print('max possible neval in batch: '
          f'{int(param_neval_frac*param_neval+(1-param_neval_frac)*2)}')
    print('---')

    np.random.seed(param_seed)

    result_means = []
    result_sdevs = []
    result_rels = []
    result_times = []
    result_times_pre = []
    result_ncalls = []
    result_avg_nevals = []
    result_chi2_values = []
    result_Q_values = []
    # result_nitn_used = []
    result_run_nums = []

    with multiprocessing.Pool(processes=NPROC) as my_pool:

        for ndims in param_ndims_lst:
            logger.info(f'ndims={ndims}')

            temp_means = []
            temp_sdevs = []
            temp_sum_evals = []
            
            # integrand = dim2func_dict[ndims]
            integrand = parallel_integrand(dim2func_dict[ndims], my_pool, NPROC)

            integ = vegas.Integrator([integ_bounds] * ndims)

            time_a = time.time()
            for i in tqdm(range(param_nitn_pre)):
                integ(integrand,
                      nitn=1,
                      neval=param_neval,
                      neval_frac=param_neval_frac,
                      nhcube_batch=param_nhcube_batch,
                      max_mem=param_max_mem,
                      maxinc_axis=param_maxinc_axis,
                      )
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
                current_result = integ(integrand,
                                       nitn=1,
                                       neval=param_neval,
                                       neval_frac=param_neval_frac,
                                       nhcube_batch=param_nhcube_batch,
                                       max_mem=param_max_mem,
                                       maxinc_axis=param_maxinc_axis,
                                       )
                true_time += time.time() - temp_time
                temp_means.append(current_result.mean)
                temp_sdevs.append(current_result.sdev)
                temp_sum_evals.append(current_result.sum_neval)
                # _, w_sdev = compute_variance_weighted_result(np.array(temp_means),
                #                                              np.array(temp_sdevs))
                # nitn_used += 1
                
            total_time = round(true_time + pre_time, 3)

            temp_means = np.array(temp_means)
            temp_sdevs = np.array(temp_sdevs)
            temp_sum_evals = np.array(temp_sum_evals)
            temp_foldername = os.path.join(*param_filename.split('/')[:-1])
            os.makedirs(temp_foldername, exist_ok=True)
            np.save(param_filename+f'_d{ndims}_means_sdevs', np.stack([temp_means, temp_sdevs, temp_sum_evals]))

            w_mean, w_sdev = compute_variance_weighted_result(temp_means, temp_sdevs)
            avg_neval = np.mean(temp_sum_evals)
            rel_unc = compute_rel_unc(w_mean, w_sdev, 1, 0)
            chi2, Q = compute_chi2_Q(w_mean, temp_means, temp_sdevs)
            ncalls = integrand.ncalls

            result_means.append(w_mean)
            result_sdevs.append(w_sdev)
            result_rels.append(rel_unc)
            result_times.append(total_time)
            result_times_pre.append(pre_time)
            result_ncalls.append(ncalls)
            result_avg_nevals.append(avg_neval)
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
        'ncall':        result_ncalls,
        'avg_neval':    result_avg_nevals,
        'chi2':         result_chi2_values,
        'Q':            result_Q_values,
        'run_num':      result_run_nums,
    })

    # save
    temp_foldername = os.path.join(*param_filename.split('/')[:-1])
    os.makedirs(temp_foldername, exist_ok=True)
    temp_df.to_csv(param_filename + '.csv', index=False)

    # display
    temp_df = temp_df[['func', 'ndim', 'nitn', 'neval',  # 'nitn_u', 
                       'res_mean', 'res_err', 'res_rel',
                       'total_time', 'ncall', 'chi2', 'Q', 'run_num']]
    print(tabulate(temp_df, headers=temp_df.columns))
