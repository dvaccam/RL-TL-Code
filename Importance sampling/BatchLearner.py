import numpy as np
import sys
import time


class BatchLearner:
    def __init__(self, gamma, policy_factory, q_estimator, v_estimator, gradient_estimator, seed):
        self.gamma = gamma
        self.policy_factory = policy_factory
        self.q_estimator = q_estimator
        self.v_estimator = v_estimator
        self.gradient_estimator = gradient_estimator
        self.initial_seed = seed
        self.seed = seed



    def learn(self, target_task, n_target_samples, n_runs=1, source_tasks=None, source_policies=None, out_stream=sys.stdout):
        self.seed = self.initial_seed
        self.out_stream = out_stream
        if self.q_estimator is not None and self.v_estimator is not None:
            if source_tasks is not None:
                print("Transfer app", file=self.out_stream)
            else:
                print("No transfer app", file=self.out_stream)
        else:
            if source_tasks is not None:
                print("Transfer", file=self.out_stream)
            else:
                print("No transfer", file=self.out_stream)

        results = np.zeros((n_runs, len(n_target_samples), 3), dtype=np.float64)
        for run_idx in range(n_runs):
            print("Run:", run_idx + 1, file=self.out_stream)
            np.random.seed(self.seed)
            self.seed = int(np.random.uniform(high=2 ** 32))

            source_samples = None
            phi_source_q = None
            phi_ns_source_q = None
            phi_source_v = None
            phi_ns_source_v = None

            if source_tasks is not None and source_policies is not None:
                source_samples = []
                phi_source_q = []
                phi_ns_source_q = []
                phi_source_v = []
                phi_ns_source_v = []
                for i in range(len(source_tasks)):
                    samples = self.collect_samples(source_tasks[i], n_target_samples[-1], source_policies[i])
                    source_samples.append(samples)
                    phi_source_q.append(self.q_estimator.map_to_feature_space(samples['fs'], samples['a']))
                    phi_ns_source_q.append(self.q_estimator.map_to_feature_space(samples['ns'], samples['na']))
                    phi_source_v.append(self.v_estimator.map_to_feature_space(samples['fs']))
                    phi_ns_source_v.append(self.v_estimator.map_to_feature_space(samples['ns']))

            for size_idx, target_size in enumerate(n_target_samples):
                print("No. samples:", target_size, file=self.out_stream)
                if target_size == n_target_samples[-1]:
                    source_samples = None
                    phi_source_q = None
                    phi_ns_source_q = None
                    phi_source_v = None
                    phi_ns_source_v = None
                alpha_1_target_opt, alpha_2_target_opt = \
                    self.optimize_policy_parameters(target_size, target_task,
                                                    source_samples, source_tasks, source_policies,
                                                    phi_source_q, phi_ns_source_q, phi_source_v, phi_ns_source_v)
                optimal_pi = self.policy_factory.create_policy(alpha_1_target_opt, alpha_2_target_opt)
                target_task.env.set_policy(optimal_pi, self.gamma)
                J1_opt = target_task.env.J
                results[run_idx, size_idx] = J1_opt
                print("Ended at:", [alpha_1_target_opt, alpha_2_target_opt], "J:", J1_opt, file=self.out_stream)

        return results



    def optimize_policy_parameters(self, target_size, target_task, source_samples=None, source_tasks=None,
                                   source_policies=None,
                                   phi_source_q=None, phi_ns_source_q=None, phi_source_v=None, phi_ns_source_v=None):
        step_size = 0.01
        max_iters = 2000
        iter = 1
        np.random.seed(self.seed)
        grad = np.zeros(2, dtype=np.float64)
        grad_norm = 1.
        alpha1 = np.random.uniform()
        alpha2 = np.random.uniform()
        print("Starting point:", [alpha1, alpha2], file=self.out_stream)

        while grad_norm > 1e-3 and iter <= max_iters:
            alpha1 += step_size * grad[0]
            alpha2 += step_size * grad[1]
            alpha1 = max(min(alpha1, 1.0), 0.0)
            alpha2 = max(min(alpha2, 1.0), 0.0)
            pol = self.policy_factory.create_policy(alpha1, alpha2)
            target_samples = self.collect_samples(target_task, target_size, pol)
            if source_samples is not None:
                source_comps = np.zeros((len(source_tasks),), dtype=np.float64)
                source_relevance = np.zeros((len(source_tasks), source_samples[0]['fs'].shape[0]), dtype=np.float64)
                for i in range(len(source_tasks)):
                    source_comps[i] = self.calculate_compliance(target_samples, source_tasks[i], source_policies[i])
                    if source_comps[i] != 0.:
                        source_relevance[i] = self.calculate_relevance(source_samples[i], target_task, pol)
                        if source_relevance[i].sum() == 0.:
                            source_comps[i] = 0.
                if source_comps.sum() != 0.:
                    source_sizes = ((source_samples[0]['fs'].shape[0] - target_size)*(source_comps/source_comps.sum())).astype(np.int64)
                    selected_samples = {'fs': [], 'a': [], 'ns': [], 'na': [], 'r': [], 'fsi': [], 'ai': [], 'nsi': [],
                                        'nai': []}
                    phi_source_q_sel = []
                    phi_ns_source_q_sel = []
                    phi_source_v_sel = []
                    phi_ns_source_v_sel = []
                    for i in range(len(source_tasks)):
                        if source_comps[i] != 0.:
                            samples_idx = np.random.choice(source_samples[i]['fs'].shape[0], p=source_relevance[i]/source_relevance[i].sum(), size=source_sizes[i])
                            for k in selected_samples.keys():
                                selected_samples[k].append(source_samples[i][k][samples_idx]) 
                            phi_source_q_sel.append(phi_source_q[i][samples_idx])
                            phi_ns_source_q_sel.append(phi_ns_source_q[i][samples_idx])
                            phi_source_v_sel.append(phi_source_v[i][samples_idx])
                            phi_ns_source_v_sel.append(phi_ns_source_v[i][samples_idx])
                    if self.v_estimator is not None and self.q_estimator is not None:
                        self.q_estimator.source_phi_sa = np.vstack(phi_source_q_sel)
                        self.q_estimator.source_phi_nsa = np.vstack(phi_ns_source_q_sel)
                        self.q_estimator.source_rewards = np.hstack(selected_samples['r'])
                        self.v_estimator.source_phi_s = np.vstack(phi_source_v_sel)
                        self.v_estimator.source_phi_ns = np.vstack(phi_ns_source_v_sel)
                        self.v_estimator.source_rewards = np.hstack(selected_samples['r'])

                    transfer_samples = {'fs': np.vstack([target_samples['fs']] + selected_samples['fs']),
                                        'a': np.vstack([target_samples['a']] + selected_samples['a']),
                                        'ns': np.vstack([target_samples['ns']] + selected_samples['ns']),
                                        'na': np.vstack([target_samples['na']] + selected_samples['na']),
                                        'r': np.concatenate([target_samples['r']] + selected_samples['r']),
                                        'fsi': np.concatenate([target_samples['fsi']] + selected_samples['fsi']),
                                        'ai': np.concatenate([target_samples['ai']] + selected_samples['ai']),
                                        'nsi': np.concatenate([target_samples['nsi']] + selected_samples['nsi']),
                                        'nai': np.concatenate([target_samples['nai']] + selected_samples['nai'])}
                    if self.v_estimator is not None and self.q_estimator is not None:
                        Qs = self.q_estimator.fit(target_samples, predict=True,
                                                  source_weights=np.ones(source_sizes.sum(), dtype=np.float64))
                        Vs = self.v_estimator.fit(target_samples, predict=True,
                                                  source_weights=np.ones(source_sizes.sum(), dtype=np.float64))
                    else:
                        Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                        Vs = target_task.env.V[transfer_samples['fsi']]
                    grad = self.gradient_estimator.estimate_gradient(target_samples,
                                                                     pol.log_gradient_matrix[transfer_samples['fsi'],transfer_samples['ai']],
                                                                     Q=Qs, V=Vs)
                else:
                    if self.v_estimator is not None and self.v_estimator is not None:
                        Qs = self.q_estimator.fit(target_samples, predict=True)
                        Vs = self.v_estimator.fit(target_samples, predict=True)
                    else:
                        Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                        Vs = target_task.env.V[target_samples['fsi']]
                    grad = self.gradient_estimator.estimate_gradient(target_samples,
                                                                     pol.log_gradient_matrix[target_samples['fsi'],target_samples['ai']],
                                                                     Q=Qs, V=Vs)
            else:
                if self.v_estimator is not None and self.v_estimator is not None:
                    Qs = self.q_estimator.fit(target_samples, predict=True)
                    Vs = self.v_estimator.fit(target_samples, predict=True)
                else:
                    Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                    Vs = target_task.env.V[target_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(target_samples,
                                                                 pol.log_gradient_matrix[target_samples['fsi'], target_samples['ai']],
                                                                 Q=Qs, V=Vs)
            grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                              (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
            grad_norm = np.linalg.norm(grad)
            grad /= np.linalg.norm(grad, ord=np.inf) if grad_norm != 0. else 1.
            iter += 1
            step_size -= (0.01 - 0.001) / max_iters
        if iter > max_iters:
            print("Did not converge;", grad_norm, iter, file=self.out_stream)
        return alpha1, alpha2



    def collect_samples(self, task, n_samples, policy):
        np.random.seed(self.seed)
        task.env._seed(self.seed)
        task.env.set_policy(policy, self.gamma)
        return task.env.sample_step(n_samples)



    def calculate_compliance(self, dataset, source_task, source_policy):
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        next_state_idx = dataset['nsi']
        source_task.env.set_policy(source_policy, self.gamma)
        probs = source_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
        sorted_idx = np.lexsort((action_idx, state_idx))
        probs = probs[sorted_idx]
        state_sorted = state_idx[sorted_idx]
        action_sorted = action_idx[sorted_idx]
        state_groups = np.ones(probs.shape[0], dtype=bool)
        action_groups = np.ones(probs.shape[0], dtype=bool)
        state_groups[1:] = state_sorted[1:] != state_sorted[:-1]
        action_groups[1:] = action_sorted[1:] != action_sorted[:-1]
        groups = np.logical_or(state_groups, action_groups)
        probs_sa = np.multiply.reduceat(probs, groups)
        return probs_sa.mean()



    def calculate_relevance(self, dataset, target_task, target_policy):
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        next_state_idx = dataset['nsi']
        target_task.env.set_policy(target_policy, self.gamma)
        probs = target_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
        return probs
