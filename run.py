# -*- coding: utf-8 -*-
import gfedplat as fp
import os


if __name__ == '__main__':
    import copy
    params = fp.read_params()
    num_runs = params.get('num_runs', 5)
    seeds = list(range(1, num_runs + 1))

    for seed in seeds:
        print(f"Run with seed {seed}")
        params_copy = copy.deepcopy(params)
        params_copy['seed'] = seed
        data_loader, algorithm = fp.initialize(params_copy)
        algorithm.save_folder = data_loader.nickname + '/C' + str(params_copy['C']) + '/' + params_copy['module'] + '/' + params_copy['algorithm'] + '/'
        if not os.path.exists(algorithm.save_folder):
            os.makedirs(algorithm.save_folder)
        algorithm.save_name = 'seed' + str(params_copy['seed']) + ' N' + str(data_loader.pool_size) + ' C' + str(params_copy['C']) + ' ' + algorithm.save_name
        algorithm.run()
