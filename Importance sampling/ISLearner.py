import numpy as np
import sys
import time



class ISLearner:
    def __init__(self, gamma, policy_factory, q_estimator, v_estimator, gradient_estimator, weights_estimator,
                 app_w_critic_Q, app_w_critic_V, app_w_actor, seed):
        self.gamma = gamma
        self.policy_factory = policy_factory
        self.q_estimator = q_estimator
        self.v_estimator = v_estimator
        self.gradient_estimator = gradient_estimator
        self.weights_estimator = weights_estimator
        self.app_w_critic_Q = app_w_critic_Q
        self.app_w_critic_V = app_w_critic_V
        self.app_w_actor = app_w_actor
        if self.weights_estimator is not None:
            self.weights_estimator.set_flags(app_w_actor, app_w_critic_Q, app_w_critic_V)
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
        all_phi_Q = None
        all_phi_V = None

        if source_tasks is not None:
            if self.app_w_critic_Q or self.app_w_actor:
                idx_grid = np.dstack(np.meshgrid(np.arange(source_tasks[0].env.state_reps.shape[0]),
                                                 np.arange(source_tasks[0].env.action_reps.shape[0]),
                                                 indexing='ij')).reshape(-1, 2)
                all_phi_Q = self.q_estimator.map_to_feature_space(source_tasks[0].env.state_reps[idx_grid[:, 0]],
                                                                  source_tasks[0].env.action_reps[idx_grid[:, 1]])
                del idx_grid
            if self.app_w_critic_V:
                all_phi_V = self.v_estimator.map_to_feature_space(source_tasks[0].env.state_reps)

        for run_idx in range(n_runs):
            print("Run:", run_idx+1, file=self.out_stream)
            np.random.seed(self.seed)
            self.seed = int(np.random.uniform(high=2 ** 32))

            source_samples = None
            source_samples_probs_zeta = None
            source_samples_probs_p = None
            source_samples_probs_pi = None

            if source_tasks is not None and source_policies is not None and n_sources_samples is not None:
                source_samples = []
                source_samples_probs_zeta = []
                if self.v_estimator is not None and self.q_estimator is not None:
                    source_samples_probs_p = []
                    source_samples_probs_pi = []
                for i in range(len(source_tasks)):
                    samples = self.collect_samples(source_tasks[i], n_sources_samples[i], source_policies[i])
                    source_samples.append(samples)
                    source_samples_probs_zeta.append(source_tasks[i].env.zeta_distr[samples['fsi'], samples['ai']])
                    if self.v_estimator is not None and self.q_estimator is not None:
                        source_samples_probs_p.append(source_tasks[i].env.transition_matrix[samples['fsi'], samples['ai'], samples['nsi']])
                        source_samples_probs_pi.append(source_policies[i].choice_matrix[samples['nsi'], samples['nai']])
                self.gradient_estimator.add_sources()
                if self.v_estimator is not None and self.q_estimator is not None:
                    self.q_estimator.add_sources(source_samples)
                    self.v_estimator.add_sources(source_samples)
                if self.weights_estimator is not None:
                    self.weights_estimator.add_sources(source_samples, source_tasks, source_policies, all_phi_Q, all_phi_V)

            for size_idx, target_size in enumerate(n_target_samples):
                print("No. samples:", target_size, file=self.out_stream)
                alpha_1_target_opt, alpha_2_target_opt =\
                    self.optimize_policy_parameters(target_size, target_task,
                                                    source_samples, source_tasks, source_policies,
                                                    source_samples_probs_zeta, source_samples_probs_p, source_samples_probs_pi,
                                                    all_phi_Q)
                optimal_pi = self.policy_factory.create_policy(alpha_1_target_opt, alpha_2_target_opt)
                target_task.env.set_policy(optimal_pi, self.gamma)
                J1_opt = target_task.env.J
                results[run_idx, size_idx] = J1_opt
                print("Ended at:", [alpha_1_target_opt, alpha_2_target_opt], "J:", J1_opt, file=self.out_stream)
            self.gradient_estimator.clean_sources()
            if self.q_estimator is not None and self.v_estimator is not None:
                self.q_estimator.clean_sources()
                self.v_estimator.clean_sources()
            if self.weights_estimator is not None:
                self.weights_estimator.clean_sources()

        return results



    def optimize_policy_parameters(self, target_size, target_task, source_samples=None, source_tasks=None, source_policies=None,
                                   source_sample_probs_zeta=None, source_sample_probs_p=None, source_sample_probs_pi=None,
                                   all_phi_Q=None):
        step_size = 0.01
        max_iters = 2000
        iter = 1
        np.random.seed(self.seed)
        grad = np.zeros(2, dtype=np.float64)
        grad_norm = 1.
        alpha1 = np.random.uniform()
        alpha2 = np.random.uniform()
        print("Starting point:", [alpha1, alpha2], file=self.out_stream)
        n_uses = 0
        cum_fun_val = 0.
        fun_vals = np.zeros(max_iters, dtype=np.float64)

        while grad_norm > 1e-3 and iter <= max_iters:
            alpha1 += step_size * grad[0]
            alpha2 += step_size * grad[1]
            alpha1 = max(min(alpha1, 1.0), 0.0)
            alpha2 = max(min(alpha2, 1.0), 0.0)
            pol = self.policy_factory.create_policy(alpha1, alpha2)
            target_samples = self.collect_samples(target_task, target_size, pol)
            if source_samples is not None:
                transfer_samples = {'fs': np.vstack([target_samples['fs']] + [ss['fs'] for ss in source_samples]),
                                    'a': np.vstack([target_samples['a']] + [ss['a'] for ss in source_samples]),
                                    'ns': np.vstack([target_samples['ns']] + [ss['ns'] for ss in source_samples]),
                                    'na': np.vstack([target_samples['na']] + [ss['na'] for ss in source_samples]),
                                    'r': np.concatenate([target_samples['r']] + [ss['r'] for ss in source_samples]),
                                    'fsi': np.concatenate([target_samples['fsi']] + [ss['fsi'] for ss in source_samples]),
                                    'ai': np.concatenate([target_samples['ai']] + [ss['ai'] for ss in source_samples]),
                                    'nsi': np.concatenate([target_samples['nsi']] + [ss['nsi'] for ss in source_samples]),
                                    'nai': np.concatenate([target_samples['nai']] + [ss['nai'] for ss in source_samples])}
                # Critic
                if self.v_estimator is not None and self.q_estimator is not None:
                    if not self.app_w_critic_V or not self.app_w_critic_Q:
                        weights_zeta = []
                        weights_p = []
                        if not self.app_w_critic_Q:
                            weights_pi = []
                        for i in range(len(source_samples)):
                            weights_zeta.append(self.calculate_density_ratios_zeta(source_samples[i], source_tasks[i],
                                                                                   target_task, source_policies[i], pol,
                                                                                   source_sample_probs_zeta[i]))
                            weights_p.append(self.calculate_density_ratios_transition(source_samples[i], source_tasks[i],
                                                                                      target_task, source_policies[i], pol,
                                                                                      source_sample_probs_p[i]))
                            if not self.app_w_critic_Q:
                                weights_pi.append(self.calculate_density_ratios_policy(source_samples[i], source_tasks[i], target_task,
                                                                                       source_policies[i], pol,
                                                                                       source_sample_probs_pi[i]))

                        weights_zeta = np.concatenate(weights_zeta)
                        weights_p = np.concatenate(weights_p)
                        if not self.app_w_critic_Q:
                            weights_pi = np.concatenate(weights_pi)
                    if self.weights_estimator is not None and (self.app_w_critic_Q or self.app_w_critic_V):
                        self.weights_estimator.prepare_lstd(pol, target_task.env.power)
                    if self.app_w_critic_Q:
                        weights_q = self.weights_estimator.estimate_weights_lstdq(target_size)
                    else:
                        weights_q = weights_zeta*weights_p*weights_pi
                    if self.app_w_critic_V:
                        weights_v = self.weights_estimator.estimate_weights_lstdv(target_size)
                    else:
                        weights_v = weights_zeta*weights_p

                    Qs = self.q_estimator.fit(target_samples, predict=True, source_weights=weights_q)
                    Vs = self.v_estimator.fit(target_samples, predict=True, source_weights=weights_v)
                else:
                    Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                    Vs = target_task.env.V[transfer_samples['fsi']]
                # Actor
                if self.app_w_actor:
                    if self.q_estimator is not None:
                        all_target_Q = self.q_estimator.predict(None, None, all_phi_Q).reshape(source_tasks[0].env.Q.shape)
                    else:
                        all_target_Q = target_task.env.Q
                    self.weights_estimator.prepare_gradient(pol, target_task.env.power, all_target_Q, Vs[target_size:])

                    '''weights_zeta = self.weights_estimator.estimate_weights_gradient(target_size)
                    grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[transfer_samples['fsi'],
                                                                                                             transfer_samples['ai']],
                                                                     Q=Qs, V=Vs, source_weights=weights_zeta)'''
                    grad_J1 = self.gradient_estimator.estimate_gradient(target_samples,
                                                                        pol.log_gradient_matrix[target_samples['fsi'], target_samples['ai']],
                                                                        Q=Qs[:target_size], V=Vs[:target_size])
                    weights_zeta, use = self.weights_estimator.estimate_weights(target_samples, pol, target_size, Qs[:target_size],
                                                                                Vs[:target_size], grad_J1)
                    #print(use)
                    #cum_fun_val = 0.9*cum_fun_val + fun_val
                    #fun_vals[iter-1] = fun_val
                    if use:
                        grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[transfer_samples['fsi'], transfer_samples['ai']],
                                                                         Q=Qs, V=Vs, source_weights=weights_zeta)
                        n_uses += 1
                    else:
                        grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[target_samples['fsi'], target_samples['ai']],
                                                                         Q=Qs[:target_size], V=Vs[:target_size])
                else:
                    if weights_zeta is None:
                        weights_zeta = []
                        for i in range(len(source_samples)):
                            weights_zeta.append(self.calculate_density_ratios_zeta(source_samples[i], source_tasks[i],
                                                                                   target_task, source_policies[i], pol,
                                                                                   source_sample_probs_zeta[i]))
                        weights_zeta = np.concatenate(weights_zeta)
                    grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[transfer_samples['fsi'],
                                                                                                             transfer_samples['ai']],
                                                                     Q=Qs, V=Vs, source_weights=weights_zeta)
                '''g = pol.log_gradient_matrix.copy()
                g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q * target_task.env.zeta_distr)
                g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))/(1. - self.gamma)
                print(grad, grad_J1, g,alpha1, alpha2)'''
            else:
                if self.q_estimator is not None and self.v_estimator is not None:
                    Qs = self.q_estimator.fit(target_samples, predict=True)
                    Vs = self.v_estimator.fit(target_samples, predict=True)
                else:
                    Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                    Vs = target_task.env.V[target_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[target_samples['fsi'],
                                                                                                         target_samples['ai']],
                                                                 Q=Qs, V=Vs)
            grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                              (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
            grad_norm = np.linalg.norm(grad)
            grad /= np.linalg.norm(grad, ord=np.inf) if grad_norm != 0. else 1.
            iter += 1
            step_size -= (0.01 - 0.001) / max_iters
            weights_zeta = None
        #print(n_uses, iter, file=self.out_stream)
        #np.save('fun_vals_1_'+str(target_size), fun_vals)
        if iter > max_iters:
            print("Did not converge;",grad_norm,iter, file=self.out_stream)
        return alpha1, alpha2



    def collect_samples(self, task, n_samples, policy):
        np.random.seed(self.seed)
        task.env._seed(self.seed)
        task.env.set_policy(policy, self.gamma)
        return task.env.sample_step(n_samples)



    def calculate_density_ratios_zeta(self, dataset, source_task, target_task, source_policy, target_policy,
                                       source_sample_probs=None):
        target_task.env.set_policy(target_policy, self.gamma)
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        if source_sample_probs is None:
            source_task.env.set_policy(source_policy, self.gamma)
            source_sample_probs = source_task.env.zeta_distr[state_idx, action_idx]
        target_sample_probs = target_task.env.zeta_distr[state_idx, action_idx]
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