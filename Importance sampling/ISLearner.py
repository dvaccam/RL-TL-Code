import numpy as np
import sys
from functools import reduce
import time



class ISLearner:
    def __init__(self, gamma, policy_factory, q_estimator, v_estimator, gradient_estimator, weights_estimator,
                 app_w_critic_Q, app_w_critic_V, app_w_actor, seed, init_source, init_target):
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
        self.init_source = init_source
        self.init_target = init_target



    def learn(self, target_task, n_target_samples, n_runs=1, source_tasks=None, source_policies=None, n_sources_samples=None,
              out_stream=sys.stdout, name=None):
        self.seed = self.initial_seed
        self.out_stream = out_stream

        results = np.zeros((n_runs, len(n_target_samples), 2), dtype=np.float64)
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
                alpha_1_target_opt, alpha_2_target_opt, iters, ess_critic_v, ess_critic_q, ess_actor =\
                    self.optimize_policy_parameters(target_size, target_task,
                                                    source_samples, source_tasks, source_policies,
                                                    source_samples_probs_zeta, source_samples_probs_p, source_samples_probs_pi)
                if source_samples is not None:
                    np.save(name + '_ess_critic_v_' + str(run_idx + 1) + '_' + str(size_idx+1), ess_critic_v)
                    np.save(name + '_ess_critic_q_' + str(run_idx + 1) + '_' + str(size_idx+1), ess_critic_q)
                    np.save(name + '_ess_actor_' + str(run_idx + 1) + '_' + str(size_idx+1), ess_actor)
                optimal_pi = self.policy_factory.create_policy(alpha_1_target_opt, alpha_2_target_opt)
                target_task.env.set_policy(optimal_pi, self.gamma)
                J1_opt = target_task.env.J
                results[run_idx, size_idx] = J1_opt, iters
                print("Ended at:", [alpha_1_target_opt, alpha_2_target_opt], "J:", J1_opt, file=self.out_stream)
            self.gradient_estimator.clean_sources()
            if self.q_estimator is not None and self.v_estimator is not None:
                self.q_estimator.clean_sources()
                self.v_estimator.clean_sources()
            if self.weights_estimator is not None:
                self.weights_estimator.clean_sources()

        return results



    def optimize_policy_parameters(self, target_size, target_task, source_samples=None, source_tasks=None, source_policies=None,
                                   source_sample_probs_zeta=None, source_sample_probs_p=None, source_sample_probs_pi=None):
        step_size = 0.01
        max_iters = 2000
        iter = 1
        np.random.seed(self.seed)
        grad = np.zeros(2, dtype=np.float64)
        grad_norm = 1.
        if self.init_target == 'r':
            alpha1 = np.random.uniform()
            alpha2 = np.random.uniform()
        else:
            closest = np.argmin(np.abs(np.array([st.env.power for st in source_tasks]) - target_task.env.power))
            alpha1 = source_policies[closest].alpha1
            alpha2 = source_policies[closest].alpha2
        print("Starting point:", [alpha1, alpha2], file=self.out_stream)
        weights_zeta = None
        m = 0.
        v = 0.
        m_tilde = grad.copy()
        v_tilde = grad.copy()
        beta_1 = 0.9
        beta_2 = 0.999
        eps = 1e-8
        if source_samples is not None:
            ess_critic_v = np.empty(max_iters, dtype=np.float64)
            ess_critic_q = np.empty(max_iters, dtype=np.float64)
            ess_actor = np.empty(max_iters, dtype=np.float64)
        else:
            ess_critic_v = None
            ess_critic_q = None
            ess_actor = None

        while grad_norm > 1e-2 and iter <= max_iters:
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
                        if target_size != 0:
                            self.weights_estimator.prepare_lstd(pol, target_task.env.power)
                    if self.app_w_critic_Q:
                        if target_size != 0:
                            A_1, b_1 = self.q_estimator.produce_matrices(target_samples)
                            weights_q = self.weights_estimator.estimate_weights_lstdq(target_size, A_1, b_1)
                        else:
                            weights_q = np.ones(transfer_samples['fs'].shape[0])
                    else:
                        weights_q = weights_zeta*weights_p*weights_pi
                    if self.app_w_critic_V:
                        if target_size != 0:
                            A_1, b_1 = self.v_estimator.produce_matrices(target_samples)
                            weights_v = self.weights_estimator.estimate_weights_lstdv(target_size, A_1, b_1)
                        else:
                            weights_v = np.ones(transfer_samples['fs'].shape[0])
                    else:
                        weights_v = weights_zeta*weights_p
                    Qs = self.q_estimator.fit(target_samples, predict=True, source_weights=weights_q)
                    Vs = self.v_estimator.fit(target_samples, predict=True, source_weights=weights_v)
                    ess_critic_v[iter-1] = (np.linalg.norm(weights_v, ord=1)/np.linalg.norm(weights_v))**2
                    ess_critic_q[iter-1] = (np.linalg.norm(weights_q, ord=1)/np.linalg.norm(weights_q))**2
                else:
                    Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                    Vs = target_task.env.V[transfer_samples['fsi']]
                # Actor
                if self.app_w_actor:
                    if target_size != 0:
                        grad_J1 = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[target_samples['fsi'],
                                                                                                                    target_samples['ai']],
                                                                            Q=Qs[:target_size], V=Vs[:target_size])
                        self.weights_estimator.prepare_gradient(pol, target_task.env.power, Qs[target_size:], Vs[target_size:])
                        weights_zeta = self.weights_estimator.estimate_weights_gradient(target_size, grad_J1)
                    else:
                        weights_zeta = np.ones(transfer_samples['fs'].shape[0])
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
                ess_actor[iter-1] = (np.linalg.norm(weights_zeta, ord=1)/np.linalg.norm(weights_zeta))**2
            else:
                if target_size == 0:
                    break
                if self.q_estimator is not None and self.v_estimator is not None:
                    Qs = self.q_estimator.fit(target_samples, predict=True)
                    Vs = self.v_estimator.fit(target_samples, predict=True)
                else:
                    Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                    Vs = target_task.env.V[target_samples['fsi']]
                grad = self.gradient_estimator.estimate_gradient(target_samples, pol.log_gradient_matrix[target_samples['fsi'],
                                                                                                         target_samples['ai']],
                                                                 Q=Qs, V=Vs)
            '''g = pol.log_gradient_matrix.copy()
            g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q * target_task.env.zeta_distr)
            g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))/(1. - self.gamma)
            print(grad, g, alpha1, alpha2, target_task.env.J)'''
            grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                              (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
            m = beta_1*m + (1. - beta_1)*grad
            v = beta_2*v + (1. - beta_2)*(grad**2)
            m_tilde = m/(1. - beta_1**iter)
            v_tilde = v/(1. - beta_2**iter)
            grad = (m_tilde/(np.sqrt(v_tilde) + eps))
            grad_norm = np.linalg.norm(grad)
            if iter % 40 == 0:
                print(iter)
                sys.stdout.flush()
            iter += 1
            step_size -= (0.01 - 0.001) / max_iters
            weights_zeta = None
        print("Finished at", iter-1, "iterations", file=self.out_stream)
        if iter > max_iters:
            print("Did not converge;",grad_norm, file=self.out_stream)
        if source_tasks is not None:
            return alpha1, alpha2, iter - 1, ess_critic_v[:iter-1], ess_critic_q[:iter-1], ess_actor[:iter-1]
        else:
            return alpha1, alpha2, iter - 1, None, None, None



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