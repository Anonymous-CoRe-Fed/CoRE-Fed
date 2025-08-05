# -*- coding: utf-8 -*-
import gfedplat as fp
import copy
import numpy as np
import torch
import time

class CoReFed(fp.Algorithm):
    def __init__(self,
                 name='CoReFed',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 client_test=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 params=None,
                 *args,
                 **kwargs):

        if save_name is None:
            save_name = name + ' ' + module.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay'])
        # Call the parent class's __init__ method
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, client_test=client_test, max_comm_round=max_comm_round, max_training_num=max_training_num, epochs=epochs, save_name=save_name, outFunc=outFunc, write_log=write_log, dishonest=dishonest, params=params, *args, **kwargs)
        # Define the client history recorder
        self.client_online_round_history = [None] * self.client_num 
        self.client_gradient_history = [None] * self.client_num 

        self.used_history_flag = False

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))

    def run(self):
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()

            m_locals, _ = self.train_a_round()
            com_time_end = time.time()
            cal_time_start = time.time()
            old_model = self.module.span_model_params_to_vec()

            self.weight_aggregate_fairness(m_locals)

            self.current_training_num += self.epochs * batch_num

            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start

    def contrastive_loss(self, client_embeddings, global_embeddings, temperature=0.07):

        eps = 1e-8

        # Check temperature to avoid division by zero or too small values
        if temperature < 1e-5:
            print(f"Warning: Temperature {temperature} is too small, setting to 1e-5 to avoid instability.")
            temperature = 1e-5

        # Normalize embeddings with epsilon to avoid division by zero
        client_embeddings_norm = client_embeddings / (torch.norm(client_embeddings, dim=1, keepdim=True) + eps)
        if global_embeddings.dim() == 1:
            global_embeddings_norm = global_embeddings / (torch.norm(global_embeddings) + eps)
            global_embeddings_norm = global_embeddings_norm.unsqueeze(0)  # Make it 2D for concatenation
        else:
            global_embeddings_norm = global_embeddings / (torch.norm(global_embeddings, dim=1, keepdim=True) + eps)

        # Flatten embeddings if needed
        if client_embeddings_norm.dim() > 2:
            client_embeddings_norm = client_embeddings_norm.view(client_embeddings_norm.size(0), -1)
        if global_embeddings_norm.dim() > 2:
            global_embeddings_norm = global_embeddings_norm.view(global_embeddings_norm.size(0), -1)

        # Concatenate client embeddings and global embeddings to form batch
        embeddings = torch.cat([client_embeddings_norm, global_embeddings_norm], dim=0)  # (num_clients + batch_size, embedding_dim)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (num_clients + batch_size, num_clients + batch_size)

        num_clients = client_embeddings_norm.size(0)
        batch_size = global_embeddings_norm.size(0)

        # Create mask to remove similarity of samples to themselves and their positive pairs
        mask = torch.eye(num_clients + batch_size, dtype=torch.bool).to(client_embeddings.device)
        positive_mask = torch.zeros_like(mask)
        # Assuming positive pairs are aligned by index (i-th client with i-th global embedding)
        min_pairs = min(num_clients, batch_size)
        for i in range(min_pairs):
            positive_mask[i, num_clients + i] = True
            positive_mask[num_clients + i, i] = True
        combined_mask = mask | positive_mask

        # Compute logits
        logits = similarity_matrix / temperature

        # Mask logits for negatives with large negative number
        large_neg = -1e9
        logits = logits.masked_fill(~combined_mask, large_neg)

        # Slice logits to only include min_pairs pairs for loss computation
        indices = list(range(min_pairs)) + list(range(num_clients, num_clients + min_pairs))
        logits = logits[indices, :][:, indices]

        # Check if any row in logits is all large_neg, which would cause NaN loss
        row_all_neg = (logits == large_neg).all(dim=1)
        if row_all_neg.any():
            print(f"Warning: Some rows in logits are all masked with large negative values, which may cause NaN loss. Rows: {row_all_neg.nonzero(as_tuple=True)[0]}")

        # Create targets for cross entropy loss
        targets = torch.arange(2 * min_pairs).to(client_embeddings.device)

        # Use cross entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)

        return loss

    def weight_aggregate_fairness(self, m_locals):

        #Weighted aggregation based on participation frequency and cosine similarity.

        eps = 1e-8
        k = self.params.get('fairness_scaling_k', 2)
        fairness_exponent_tau = self.params.get('fairness_exponent_tau', 0.5)  # exponent scaling factor for fairness weighting

        # Compute dynamic sliding window tau based on online client history (same logic as in train_a_round)
        total_client_num = 0
        for item in self.client_online_round_history:
            if item is not None:
                total_client_num += 1
        if total_client_num > self.online_client_num:
            sliding_tau = int(total_client_num / self.online_client_num)
        else:
            sliding_tau = 1  # fallback to 1 if condition not met

        # Compute participation frequency fi for each client over sliding window sliding_tau
        fi_list = []
        for client_id in range(self.client_num):
            last_online_round = self.client_online_round_history[client_id]
            if last_online_round is None:
                fi = 0.0
            else:
                # Count how many rounds client was online in last sliding_tau rounds
                count = 0
                for r in range(self.current_comm_round - sliding_tau + 1, self.current_comm_round + 1):
                    if self.client_online_round_history[client_id] is not None and self.client_online_round_history[client_id] >= r:
                        count += 1
                fi = count / sliding_tau
            fi_list.append(fi)

        # Filter fi_list to only include online clients
        online_client_ids = [client.id for client in self.online_client_list]
        fi_list_online = [fi_list[client_id] for client_id in online_client_ids]

        fi_tensor = torch.tensor(fi_list_online, dtype=torch.float32, device=self.device)

        # Compute cosine similarity alpha_i between client and global embedding
        # Use self.alignment_matrix if exists, else compute from embeddings if available
        if hasattr(self, 'alignment_matrix') and self.alignment_matrix is not None:
            alpha = self.alignment_matrix.squeeze()
            if alpha.dim() == 0:
                alpha = alpha.unsqueeze(0)
        else:
            # Fallback: set alpha to ones
            alpha = torch.ones(len(online_client_ids), device=self.device)

        # Compute weights wi = (1/(fi + eps))^fairness_exponent_tau * sigmoid(k * alpha_i)
        weights = ((1.0 / (fi_tensor + eps)) ** fairness_exponent_tau) * self.sigmoid(k * alpha)

        # Normalize weights
        weights = weights / weights.sum()

        # Extract parameter dicts from model objects or use state_dicts directly
        state_dicts = []
        for model in m_locals:
            if hasattr(model, 'named_parameters') and callable(getattr(model, 'named_parameters')):
                param_dict = {}
                for name, param in model.named_parameters():
                    param_dict[name] = param.data.clone()
                state_dicts.append(param_dict)
            elif hasattr(model, 'state_dict') and callable(getattr(model, 'state_dict')):
                # Use state_dict method to get parameters
                state_dicts.append(model.state_dict())
            elif hasattr(model, 'model'):
                # Check if model has .model attribute with named_parameters or state_dict
                inner_model = model.model
                if hasattr(inner_model, 'named_parameters') and callable(getattr(inner_model, 'named_parameters')):
                    param_dict = {}
                    for name, param in inner_model.named_parameters():
                        param_dict[name] = param.data.clone()
                    state_dicts.append(param_dict)
                elif hasattr(inner_model, 'state_dict') and callable(getattr(inner_model, 'state_dict')):
                    state_dicts.append(inner_model.state_dict())
                else:
                    raise AttributeError(f"Inner model object {inner_model} has no named_parameters method or state_dict method")
            elif isinstance(model, dict):
                # Assume model is a state_dict
                state_dicts.append(model)
            else:
                raise AttributeError(f"Model object {model} has no named_parameters method, no state_dict method, and is not a state_dict")

        # Weighted aggregation of parameter dicts
        aggregated = copy.deepcopy(state_dicts[0])

        # Filter keys to only tensor values
        tensor_keys = [key for key, value in aggregated.items() if torch.is_tensor(value)]

        for key in tensor_keys:
            aggregated[key] = torch.zeros_like(aggregated[key])

        for i, local_params in enumerate(state_dicts):
            for key in tensor_keys:
                aggregated[key] += weights[i] * local_params[key]

        # Update model parameters with aggregated result
        if hasattr(self.module, 'model') and hasattr(self.module.model, 'load_state_dict'):
            self.module.model.load_state_dict(aggregated)
        else:
            self.module.load_state_dict(aggregated)

    def train_a_round(self):

        # Reset accuracy sums and counts for the current round to avoid accumulation across rounds
        round_num = self.current_comm_round

        com_time_start = time.time()

        # Call parent train() to get model objects and losses
        m_locals, l_locals = super().train()

        com_time_end = time.time()

        # Evaluate gradients and losses for custom logic
        g_locals, l_locals_eval = self.evaluate()

        # Dealing with historical fairness
        client_id_list = self.get_clinet_attr('id')
        add_grads = []
        self.used_history_flag = False
        
        total_client_num = 0
        for item in self.client_online_round_history:
            if item is not None:
                total_client_num += 1
        if total_client_num > self.online_client_num:
            tau = int(total_client_num / self.online_client_num)
            for client_id, item in enumerate(self.client_online_round_history):
                if item is not None:
                    if self.current_comm_round - item <= tau:  
                        if client_id not in client_id_list: 
                            add_grads.append(self.client_gradient_history[client_id])
        if len(add_grads) == 0:
            add_grads = None
        else:
            add_grads = torch.vstack(add_grads)
            self.used_history_flag = True

        prefer_vec = torch.Tensor([1.0] * self.online_client_num).float().to(self.device)
        prefer_vec = prefer_vec / torch.norm(prefer_vec)

        # Apply knowledge distillation and alignment matrix calculation
        client_embeddings = []
        for idx, client in enumerate(self.online_client_list):
            client_data = None
            if client.local_training_data is not None and len(client.local_training_data) > 0:
                client_data = next(iter(client.local_training_data))[0]  # get input tensor from first batch
            elif client.local_test_data is not None and len(client.local_test_data) > 0:
                client_data = next(iter(client.local_test_data))[0]
            else:
                pass
            if client_data is not None:
                client_data = client_data.to(self.device)
                embedding = self.module.model.forward(client_data, return_embedding=True)
                if embedding.dim() > 1:
                    embedding = embedding.mean(dim=0)
                client_embeddings.append(embedding)
            else:
                print(f"Client {client.id} has no training or test data to provide input for embedding extraction.")

        global_embedding = None
        for client in self.online_client_list:
            client_data = None
            if client.local_training_data is not None and len(client.local_training_data) > 0:
                client_data = next(iter(client.local_training_data))[0]
            elif client.local_test_data is not None and len(client.local_test_data) > 0:
                client_data = next(iter(client.local_test_data))[0]
            if client_data is not None:
                client_data = client_data.to(self.device)
                global_embedding = self.module.model.forward(client_data, return_embedding=True)
                if global_embedding.dim() > 1:
                    global_embedding = global_embedding.mean(dim=0)
                break

        if global_embedding is None:
            print("No data available to compute global embedding.")
            self.alignment_matrix = None
        else:
            if len(client_embeddings) > 0 and global_embedding is not None:
                client_embeddings_tensor = torch.stack(client_embeddings)

                loss = self.contrastive_loss(client_embeddings_tensor, global_embedding)

                eps = 1e-8
                client_embeddings_norm = client_embeddings_tensor / (torch.norm(client_embeddings_tensor, dim=1, keepdim=True) + eps)
                if global_embedding.dim() == 1:
                    global_embedding_norm = global_embedding / (torch.norm(global_embedding) + eps)
                    global_embedding_norm = global_embedding_norm.unsqueeze(0)
                else:
                    global_embedding_norm = global_embedding / (torch.norm(global_embedding, dim=1, keepdim=True) + eps)

                self.alignment_matrix = torch.matmul(client_embeddings_norm, global_embedding_norm.T)

                alpha = 0.5
                updated_client_embeddings = []
                for i in range(client_embeddings_tensor.size(0)):
                    weighted_global = torch.sum(self.alignment_matrix[i] * global_embedding_norm, dim=0)
                    new_emb = client_embeddings_tensor[i] + alpha * (weighted_global - client_embeddings_tensor[i])
                    updated_client_embeddings.append(new_emb)

                updated_client_embeddings_tensor = torch.stack(updated_client_embeddings)

                client_embeddings_tensor = updated_client_embeddings_tensor

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Perform weighted fairness aggregation instead of FedAvg aggregation
        self.weight_aggregate_fairness(m_locals)

        # Perform client testing (accuracy calculation) after knowledge distillation and aggregation
        for client in self.online_client_list:
            client.test(self.module)

        self.current_training_num += 1

        last_client_id_list = self.get_clinet_attr('id')
        last_g_locals = copy.deepcopy(g_locals)
        for idx, client_id in enumerate(last_client_id_list):
            self.client_online_round_history[client_id] = self.current_comm_round
            temp = self.client_gradient_history[client_id]
            self.client_gradient_history[client_id] = None
            del temp
            self.client_gradient_history[client_id] = last_g_locals[idx]

        self.communication_time += com_time_end - com_time_start

        return m_locals, l_locals
