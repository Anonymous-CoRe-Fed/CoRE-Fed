import gfedplat as fp
import numpy as np
import random
import torch


def separate_data(data, train_prob, num_clients, target_class_num, item_classes_num, batch_size, alpha, niid=False, balance=False, partition=None):

    least_samples = batch_size / (1-train_prob)
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'dir'
        item_classes_num = target_class_num

    if partition == "dir":

        min_size = 0
        K = target_class_num
        N = len(dataset_label)

        while min_size < target_class_num:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))

                proportions = np.array(
                    [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(
            y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def seperate_batch_data(input_data, target_data, batch_size):
    batch_data_indices_list = fp.DataLoader.separate_list(
        list(range(len(target_data))), batch_size)
    local_data = []
    for batch_data_indices in batch_data_indices_list:

        batch_input_data = input_data[batch_data_indices]

        batch_target_data = target_data[batch_data_indices]
        local_data.append((batch_input_data, batch_target_data))
    return local_data


def create_data_pool(X, y, pool_size, shuffle, train_prob, batch_size, target_class_num):

    data_pool = [{} for _ in range(pool_size)]
    raw_test_data_pool = [{} for _ in range(pool_size)]
    for pool_idx in range(pool_size):
        input_data = torch.Tensor(X[pool_idx]).float()
        target_data = torch.Tensor(y[pool_idx]).long()
        if shuffle:
            indices = list(range(len(target_data)))
            random.shuffle(indices)
            input_data = input_data[indices]
            target_data = target_data[indices]

        training_input_data = input_data[:int(train_prob*len(target_data))]
        training_target_data = target_data[:int(train_prob*len(target_data))]
        test_input_data = input_data[int(train_prob*len(target_data)):]
        test_target_data = target_data[int(train_prob*len(target_data)):]

        data_pool[pool_idx]['local_training_data'] = seperate_batch_data(
            training_input_data, training_target_data, batch_size)
        data_pool[pool_idx]['local_training_number'] = len(
            training_target_data)
        data_pool[pool_idx]['data_name'] = str(pool_idx)

        data_pool[pool_idx]['local_test_data'] = seperate_batch_data(
            test_input_data, test_target_data, batch_size)
        data_pool[pool_idx]['local_test_number'] = len(test_target_data)
        raw_test_data_pool[pool_idx]['input_data'] = test_input_data
        raw_test_data_pool[pool_idx]['target_data'] = test_target_data

    return data_pool
