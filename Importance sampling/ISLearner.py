import numpy as np
import sys
import time



class ISLearner:
    def __init__(self, gamma, policy_factory, q_estimator, v_estimator, gradient_estimator, weights_estimator, seed):
        self.gamma = gamma
        self.policy_factory = policy_factory
        self.q_estimator = q_estimator
        self.v_estimator = v_estimator
        self.gradient_estimator = gradient_estimator
        self.weights_estimator = weights_estimator
        self.initial_seed = seed
        self.seed = seed



    def learn(self, target_task, n_target_samples, n_runs=1, source_tasks=None, source_policies=None, n_sources_samples=None,
              out_stream=sys.stdout):
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

        results = np.zeros((n_runs, len(n_target_samples)), dtype=np.float64)
        for run_idx in range(n_runs):
            print("Run:", run_idx+1, file=self.out_stream)
            np.random.seed(self.seed)
            self.seed = int(np.random.uniform(high=2 ** 32))

            source_samples = None
            source_samples_probs_dseta = None
            source_samples_probs_p = None
            source_samples_probs_pi = None

            if source_tasks is not None and source_policies is not None and n_sources_samples is not None:
                source_samples = []
                source_samples_probs_dseta = []
                if self.v_estimator is not None and self.q_estimator is not None:
                    source_samples_probs_p = []
                    source_samples_probs_pi = []
                for i in range(len(source_tasks)):
                    samples = self.collect_samples(source_tasks[i], n_sources_samples[i], source_policies[i])
                    source_samples.append(samples)
                    source_samples_probs_dseta.append(source_tasks[i].env.dseta_distr[samples['fsi'], samples['ai']])
                    if self.v_estimator is not None and self.q_estimator is not None:
                        source_samples_probs_p.append(source_tasks[i].env.transition_matrix[samples['fsi'], samples['ai'], samples['nsi']])
                        source_samples_probs_pi.append(source_policies[i].choice_matrix[samples['nsi'], samples['nai']])
                self.gradient_estimator.add_sources()
                if self.v_estimator is not None and self.q_estimator is not None:
                    self.q_estimator.add_sources(source_samples)
                    self.v_estimator.add_sources(source_samples)
                if self.weights_estimator is not None:
                    self.weights_estimator.add_sources(source_samples)

            if self.weights_estimator is not None:
                idx_grid = np.dstack(np.meshgrid(np.arange(source_tasks[0].env.state_reps.shape[0]),
                                                 np.arange(source_tasks[0].env.action_reps.shape[0]),
                                                 indexing='ij')).reshape(-1, 2)
                phi_all_q = self.q_estimator.map_to_feature_space(source_tasks[0].env.state_reps[idx_grid[:,0]],
                                                                  source_tasks[0].env.action_reps[idx_grid[:,1]])
                self.weights_estimator.set_sources(source_samples, source_tasks, source_policies, source_samples_Qs,
                                                   phi_all_q)

            for size_idx, target_size in enumerate(n_target_samples):
                print("No. samples:", target_size, file=self.out_stream)
                alpha_1_target_opt, alpha_2_target_opt =\
                    self.optimize_policy_parameters(target_size, target_task,
                        source_samples, source_tasks, source_policies,
                        source_samples_probs_dseta, source_samples_probs_p, source_samples_probs_pi,
                        phi_source_q, phi_ns_source_q, phi_source_v, phi_ns_source_v)
                optimal_pi = self.policy_factory.create_policy(alpha_1_target_opt, alpha_2_target_opt)
                target_task.env.set_policy(optimal_pi, self.gamma)
                J1_opt = target_task.env.J
                results[run_idx, size_idx] = J1_opt
                print("Ended at:", [alpha_1_target_opt, alpha_2_target_opt], "J:", J1_opt, file=self.out_stream)
                self.gradient_estimator.clean_sources()
                self.q_estimator.clean_sources()
                self.v_estimator.clean_sources()

        return results



    def optimize_policy_parameters(self, target_size, target_task, source_samples=None, source_tasks=None, source_policies=None,
                                   source_sample_probs_dseta=None, source_sample_probs_p=None, source_sample_probs_pi=None,
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
                weights_dseta = [np.ones(target_size, np.float64)]
                if self.v_estimator is not None and self.q_estimator is not None:
                    weights_p = [np.ones(target_size, np.float64)]
                    weights_pi = [np.ones(target_size, np.float64)]
                for i in range(len(source_samples)):
                    weights_dseta.append(self.calculate_density_ratios_dseta(source_samples[i], source_tasks[i], target_task,
                                                                             source_policies[i], pol,
                                                                             source_sample_probs_dseta[i]))
                    if self.v_estimator is not None and self.q_estimator is not None:
                        weights_p.append(self.calculate_density_ratios_transition(source_samples[i], source_tasks[i], target_task,
                                                                                  source_policies[i], pol,
                                                                                  source_sample_probs_p[i]))
                        weights_pi.append(self.calculate_density_ratios_policy(source_samples[i], source_tasks[i], target_task,
                                                                               source_policies[i], pol,
                                                                               source_sample_probs_pi[i]))

                weights_dseta = np.concatenate(weights_dseta)
                if self.v_estimator is not None and self.q_estimator is not None:
                    weights_p = np.concatenate(weights_p)
                    weights_pi = np.concatenate(weights_pi)

                transfer_samples = {'fs': np.vstack([target_samples['fs']] + [ss['fs'] for ss in source_samples]),
                                  'a': np.vstack([target_samples['a']] + [ss['a'] for ss in source_samples]),
                                  'ns': np.vstack([target_samples['ns']] + [ss['ns'] for ss in source_samples]),
                                  'na': np.vstack([target_samples['na']] + [ss['na'] for ss in source_samples]),
                                  'r': np.concatenate([target_samples['r']] + [ss['r'] for ss in source_samples]),
                                  'fsi': np.concatenate([target_samples['fsi']] + [ss['fsi'] for ss in source_samples]),
                                  'ai': np.concatenate([target_samples['ai']] + [ss['ai'] for ss in source_samples]),
                                  'nsi': np.concatenate([target_samples['nsi']] + [ss['nsi'] for ss in source_samples]),
                                  'nai': np.concatenate([target_samples['nai']] + [ss['nai'] for ss in source_samples])}
                if self.v_estimator is not None and self.q_estimator is not None:
                    Qs = self.q_estimator.fit(transfer_samples, predict=True,
                                              source_weights=weights_dseta * weights_p * weights_pi,
                                              phi_source=phi_source_q, phi_ns_source=phi_ns_source_q)
                    Vs = self.v_estimator.fit(transfer_samples, predict=True,
                                              source_weights=weights_dseta * weights_p,
                                              phi_source=phi_source_v, phi_ns_source=phi_ns_source_v)
                else:
                    Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                    Vs = target_task.env.V[transfer_samples['fsi']]
                if self.weights_estimator is not None:
                    grad_J1 = self.gradient_estimator.estimate_gradient(target_samples, pol,
                                                                        Q=Qs[:target_size], V=Vs[:target_size])
                    weights_dseta = self.weights_estimator.estimate_weights(pol, target_task.env.power, target_size,
                                                                            self.q_estimator, grad_J1, target_task)
                    weights_dseta = np.append(np.ones(target_size, dtype=np.float64), weights_dseta)
                grad = self.gradient_estimator.estimate_gradient(transfer_samples, pol, Q=Qs, V=Vs, source_weights=weights_dseta)
                '''g = pol.log_gradient_matrix.copy()
                g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q * target_task.env.dseta_distr)
                g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))
                print(grad, grad_J1, g, alpha1, alpha2)'''
            else:
                if self.q_estimator is not None and self.v_estimator is not None:
                    Qs = self.q_estimator.fit(target_samples, predict=True)
                    Vs = self.v_estimator.fit(target_samples, predict=True)
                else:
                    Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                    Vs = target_task.env.V[target_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(target_samples, pol, Q=Qs, V=Vs)
            grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                              (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
            grad_norm = np.linalg.norm(grad)
            #grad = np.clip(grad, -1., 1.)
            grad /= np.linalg.norm(grad, ord=np.inf) if grad_norm != 0. else 1.
            iter += 1
            step_size -= (0.01 - 0.001) / max_iters
        if iter > max_iters:
            print("Did not converge;",grad_norm,iter, file=self.out_stream)
        return alpha1, alpha2



    def collect_samples(self, task, n_samples, policy):
        np.random.seed(self.seed)
        task.env._seed(self.seed)
        task.env.set_policy(policy, self.gamma)
        return task.env.sample_step(n_samples)



    def calculate_density_ratios_dseta(self, dataset, source_task, target_task, source_policy, target_policy,
                                       source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]
        target_sample_probs = target_task.env.dseta_distr[state_idx, action_idx]
        return target_sample_probs / source_sample_probs



    def calculate_density_ratios_transition(self, dataset, source_task, target_task, source_policy, target_policy,
                                            source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        next_state_idx = dataset['nsi']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
        target_sample_probs = target_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
        return target_sample_probs / source_sample_probs



    def calculate_density_ratios_policy(self, dataset, source_task, target_task, source_policy, target_policy,
                                        source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        next_state_idx = dataset['nsi']
        next_action_idx = dataset['nai']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_policy.choice_matrix[next_state_idx, next_action_idx]
        target_sample_probs = target_policy.choice_matrix[next_state_idx, next_action_idx]
        return target_sample_probs / source_sample_probs