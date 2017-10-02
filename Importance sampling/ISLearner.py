import numpy as np
import sys
import time



class ISLearner:
    def __init__(self, gamma, policy_factory, q_estimator, v_estimator, gradient_estimator, seed):
        self.gamma = gamma
        self.policy_factory = policy_factory
        self.q_estimator = q_estimator
        self.v_estimator = v_estimator
        self.gradient_estimator = gradient_estimator
        self.initial_seed = seed
        self.seed = seed



    def learn(self, target_task, n_target_samples, n_runs=1, source_tasks=None, source_policies=None, n_sources_samples=None,
              out_stream=sys.stdout):
        self.seed = self.initial_seed
        self.out_stream = out_stream
        results = np.zeros((n_runs, len(n_target_samples), 3), dtype=np.float64)
        for run_idx in range(n_runs):
            print("Run:", run_idx+1, file=self.out_stream)
            np.random.seed(self.seed)
            self.seed = int(np.random.uniform(high=2 ** 32))

            source_samples = None
            source_samples_probs_dseta = None
            source_samples_probs_p_sa = None
            source_samples_probs_r_sa = None
            source_samples_probs_delta = None
            source_samples_probs_p_s = None
            source_samples_probs_r_s = None
            phi_source_q = None
            phi_ns_source_q = None
            phi_source_v = None
            phi_ns_source_v = None

            if source_tasks is not None and source_policies is not None and n_sources_samples is not None:
                source_samples = []
                source_samples_probs_dseta = []
                if self.v_estimator is not None and self.q_estimator is not None:
                    source_samples_probs_p_sa = []
                    source_samples_probs_r_sa = []
                    source_samples_probs_delta = []
                    source_samples_probs_p_s = []
                    source_samples_probs_r_s = []
                    phi_source_q = []
                    phi_ns_source_q = []
                    phi_source_v = []
                    phi_ns_source_v = []
                for i in range(len(source_tasks)):
                    samples = self.collect_samples(source_tasks[i], n_sources_samples[i], source_policies[i])
                    source_samples.append(samples)
                    source_samples_probs_dseta.append(source_tasks[i].env.dseta_distr[samples['fsi'], samples['ai']])
                    if self.v_estimator is not None and self.q_estimator is not None:
                        source_samples_probs_p_sa.append(source_tasks[i].env.transition_matrix[samples['fsi'], samples['ai'], samples['nsi']] * source_policies[i].choice_matrix[samples['nsi'], samples['nai']])
                        source_samples_probs_r_sa.append(source_tasks[i].env.transition_matrix[samples['fsi'], samples['ai'], samples['nsi']])
                        source_samples_probs_delta.append(source_tasks[i].env.delta_distr[samples['fsi']])
                        source_samples_probs_p_s.append((source_policies[i].choice_matrix[samples['fsi'], :] * source_tasks[i].env.transition_matrix[samples['fsi'], :,samples['nsi']]).sum(axis=1))
                        source_samples_probs_r_s.append(source_policies[i].choice_matrix[samples['fsi'], samples['ai']] * source_tasks[i].env.transition_matrix[samples['fsi'], samples['ai'], samples['nsi']])
                        phi_source_q.append(self.q_estimator.map_to_feature_space(samples['fs'], samples['a']))
                        phi_ns_source_q.append(self.q_estimator.map_to_feature_space(samples['ns'], samples['na']))
                        phi_source_v.append(self.v_estimator.map_to_feature_space(samples['fs']))
                        phi_ns_source_v.append(self.v_estimator.map_to_feature_space(samples['ns']))
            if phi_source_q is not None and phi_ns_source_q is not None and phi_source_v is not None and phi_ns_source_v is not None:
                phi_source_q = np.vstack(phi_source_q)
                phi_ns_source_q = np.vstack(phi_ns_source_q)
                phi_source_v = np.vstack(phi_source_v)
                phi_ns_source_v = np.vstack(phi_ns_source_v)

            for size_idx, target_size in enumerate(n_target_samples):
                print("No. samples:", target_size, file=self.out_stream)
                alpha_1_target_opt, alpha_2_target_opt =\
                    self.optimize_policy_parameters(target_size, target_task,
                        source_samples, source_tasks, source_policies,
                        source_samples_probs_dseta, source_samples_probs_p_sa, source_samples_probs_r_sa,
                        source_samples_probs_delta, source_samples_probs_p_s, source_samples_probs_r_s,
                        phi_source_q, phi_ns_source_q, phi_source_v, phi_ns_source_v)
                optimal_pi = self.policy_factory.create_policy(alpha_1_target_opt, alpha_2_target_opt)
                target_task.env.set_policy(optimal_pi, self.gamma)
                J1_opt = target_task.env.J
                results[run_idx, size_idx] = np.array([alpha_1_target_opt, alpha_1_target_opt, J1_opt], dtype=np.float64)
                print("Coverged to:", [alpha_1_target_opt, alpha_2_target_opt], "J:", J1_opt, file=self.out_stream)



    def optimize_policy_parameters(self, target_size, target_task, source_samples=None, source_tasks=None, source_policies=None,
                                   source_sample_probs_dseta=None, source_sample_probs_p_sa=None, source_sample_probs_r_sa=None,
                                   source_sample_probs_delta=None, source_sample_probs_p_s=None, source_sample_probs_r_s=None,
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
                    weights_p_sa = [np.ones(target_size, np.float64)]
                    weights_r_sa = [np.ones(target_size, np.float64)]
                    weights_delta = [np.ones(target_size, np.float64)]
                    weights_p_s = [np.ones(target_size, np.float64)]
                    weights_r_s = [np.ones(target_size, np.float64)]
                for i in range(len(source_samples)):
                    weights_dseta.append(self.calculate_density_ratios_dseta(source_samples[i], source_tasks[i], target_task,
                                                                             source_policies[i], pol,
                                                                             source_sample_probs_dseta[i]))
                    if self.v_estimator is not None and self.q_estimator is not None:
                        weights_p_sa.append(self.calculate_density_ratios_transition_sa(source_samples[i], source_tasks[i], target_task,
                                                                                        source_policies[i], pol,
                                                                                        source_sample_probs_p_sa[i]))
                        weights_r_sa.append(self.calculate_density_ratios_r_sa(source_samples[i], source_tasks[i], target_task,
                                                                               source_policies[i], pol,
                                                                               source_sample_probs_r_sa[i]))
                        weights_delta.append(self.calculate_density_ratios_delta(source_samples[i], source_tasks[i], target_task,
                                                                                 source_policies[i], pol,
                                                                                 source_sample_probs_delta[i]))
                        weights_p_s.append(self.calculate_density_ratios_transition_s(source_samples[i], source_tasks[i], target_task,
                                                                                      source_policies[i], pol,
                                                                                      source_sample_probs_p_s[i]))
                        weights_r_s.append(self.calculate_density_ratios_r_s(source_samples[i], source_tasks[i], target_task,
                                                                             source_policies[i], pol,
                                                                             source_sample_probs_r_s[i]))

                weights_dseta = np.concatenate(weights_dseta)
                if self.v_estimator is not None and self.q_estimator is not None:
                    weights_p_sa = np.concatenate(weights_p_sa)
                    weights_r_sa = np.concatenate(weights_r_sa)
                    weights_delta = np.concatenate(weights_delta)
                    weights_p_s = np.concatenate(weights_p_s)
                    weights_r_s = np.concatenate(weights_r_s)

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
                                              weights_d=weights_dseta, weights_p=weights_p_sa, weights_r=weights_r_sa,
                                              phi_source=phi_source_q, phi_ns_source=phi_ns_source_q)
                    Vs = self.v_estimator.fit(transfer_samples, predict=True,
                                              weights_d=weights_delta, weights_p=weights_p_s, weights_r=weights_r_s,
                                              phi_source=phi_source_v, phi_ns_source=phi_ns_source_v)
                else:
                    Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                    Vs = target_task.env.V[transfer_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(transfer_samples, pol, Q=Qs, V=Vs, weights=weights_dseta)
            else:
                if self.v_estimator is not None and self.v_estimator is not None:
                    Qs = self.q_estimator.fit(target_samples, predict=True)
                    Vs = self.v_estimator.fit(target_samples, predict=True)
                else:
                    Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                    Vs = target_task.env.V[target_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(target_samples, pol, Q=Qs, V=Vs)
            grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                              (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
            grad_norm = np.linalg.norm(grad)
            iter += 1
            step_size -= (0.01 - 0.001) / max_iters
        #if iter > max_iters:
        #    self.out_logger.write("Did not converge")
        #    self.out_logger.write(grad_norm, iter)
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



    def calculate_density_ratios_transition_sa(self, dataset, source_task, target_task, source_policy, target_policy,
                                               source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        next_state_idx = dataset['nsi']
        next_action_idx = dataset['nai']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = \
                source_task.env.transition_matrix[state_idx, action_idx, next_state_idx] * source_policy.choice_matrix[
                    next_state_idx, next_action_idx]
        target_sample_probs = \
            target_task.env.transition_matrix[state_idx, action_idx, next_state_idx] * target_policy.choice_matrix[
                next_state_idx, next_action_idx]
        return target_sample_probs / source_sample_probs



    def calculate_density_ratios_r_sa(self, dataset, source_task, target_task, source_policy, target_policy,
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



    def calculate_density_ratios_delta(self, dataset, source_task, target_task, source_policy, target_policy,
                                       source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_task.env.delta_distr[state_idx]
        target_sample_probs = target_task.env.delta_distr[state_idx]
        return target_sample_probs / source_sample_probs



    def calculate_density_ratios_transition_s(self, dataset, source_task, target_task, source_policy, target_policy,
                                              source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        next_state_idx = dataset['nsi']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = \
                (source_policy.choice_matrix[state_idx, :] * source_task.env.transition_matrix[state_idx, :,
                                                             next_state_idx]).sum(axis=1)
        target_sample_probs = \
            (target_policy.choice_matrix[state_idx, :] * target_task.env.transition_matrix[state_idx, :,
                                                         next_state_idx]).sum(axis=1)
        return target_sample_probs / source_sample_probs



    def calculate_density_ratios_r_s(self, dataset, source_task, target_task, source_policy, target_policy,
                                     source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        next_state_idx = dataset['nsi']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_policy.choice_matrix[state_idx, action_idx] * \
                                  source_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
        target_sample_probs = target_policy.choice_matrix[state_idx, action_idx] * target_task.env.transition_matrix[
            state_idx, action_idx, next_state_idx]
        return target_sample_probs / source_sample_probs