import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PolicyFactoryMC import PolicyFactoryMC, PolicyMC
from LSTD_Q_Estimator import LSTD_Q_Estimator
from LSTD_V_Estimator import LSTD_V_Estimator
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.dummy import Pool
import copy
import threading as thr
import sys



def rescale_state(x):
    return x * (max_pos - min_pos) / 1.8



def _collect_sample(i, task, dataset):
    d = task.env.sample_step(1000, return_idx=True, include_next_action=True)
    dataset['fs'][2*i*1000:2*(i+1)*1000] = d['fs'].flatten()
    dataset['a'][i*1000:(i+1)*1000] = d['a']
    dataset['ns'][2*i*1000:2*(i+1)*1000] = d['ns'].flatten()
    dataset['na'][i*1000:(i+1)*1000] = d['na']
    dataset['r'][i*1000:(i+1)*1000] = d['r']
    dataset['fsi'][i*1000:(i+1)*1000] = d['fsi']
    dataset['ai'][i*1000:(i+1)*1000] = d['ai']
    dataset['nsi'][i*1000:(i+1)*1000] = d['nsi']
    dataset['nai'][i*1000:(i+1)*1000] = d['nai']



def collect_samples(task, n_samples, seed, policy):
    np.random.seed(seed)
    task.env._seed(seed)
    task.env.set_policy(policy, gamma)
    return task.env.sample_step(n_samples, return_idx=True, include_next_action=True)



def collect_episodes(task, n_episodes, max_episode_length, seed, policy, render):
    np.random.seed(seed)
    task.env._seed(seed)
    #aux = np.zeros(n_episodes, dtype=np.int64)
    # Sampling from task
    if True:
        # Data structures for storing samples
        if task.observation_space.shape[0] > 1:
            first_states = np.zeros((n_episodes, max_episode_length, task.observation_space.shape[0]), dtype=np.float64)
            next_states = np.zeros((n_episodes, max_episode_length, task.observation_space.shape[0]), dtype=np.float64)
        else:
            first_states = np.zeros((n_episodes, max_episode_length), dtype=np.float64)
            next_states = np.zeros((n_episodes, max_episode_length), dtype=np.float64)
        if task.action_space.shape[0] > 1:
            actions = np.zeros((n_episodes, max_episode_length, task.action_space.shape[0]), dtype=np.float64)
        else:
            actions = np.zeros((n_episodes, max_episode_length), dtype=np.float64)
        rewards = np.ones((n_episodes, max_episode_length), dtype=np.float64)
        for i in range(n_episodes):
            init_state = task.reset()
            first_state = init_state
            for t in range(max_episode_length):
                if render:
                    task.render()
                action = np.array([policy.produce_action(first_state)])
                next_state, reward, done, info = task.step(action)
                first_states[i, t] = first_state
                actions[i, t] = action
                next_states[i, t] = next_state
                rewards[i, t] = reward
                first_state = next_state
                if done:
                    #aux[i] = t
                    break
            if t + 1 < max_episode_length:
                rewards[i,t+1] = 1
    else:
        first_states = sct.RawArray('d', np.zeros(n_episodes*max_episode_length*task.observation_space.shape[0], dtype=np.float64))
        actions = sct.RawArray('d', np.zeros(n_episodes*max_episode_length, dtype=np.float64))
        next_states = sct.RawArray('d', np.zeros(n_episodes*max_episode_length*task.observation_space.shape[0], dtype=np.float64))
        rewards = sct.RawArray('d', np.ones(n_episodes*max_episode_length, dtype=np.float64))
        lock = thr.Lock()
        inps = [(i, task, max_episode_length, policy, first_states, actions, next_states, rewards, lock) for i in range(n_episodes)]
        n_jobs = 2
        pool = mp.dummy.Pool(n_jobs)
        pool.starmap(_collect_sample, inps, chunksize=int(len(inps)/n_jobs))
        first_states = np.frombuffer(first_states).reshape((n_episodes, max_episode_length, 2))
        actions = np.frombuffer(actions).reshape((n_episodes, max_episode_length))
        next_states = np.frombuffer(next_states).reshape((n_episodes, max_episode_length, 2))
        rewards = np.frombuffer(rewards).reshape((n_episodes, max_episode_length))

    #aux = next_states[:,:,0].max(axis=1)
    #plt.hist(aux, bins='sqrt')
    #plt.hist(aux[aux != 0], bins='sqrt')
    #plt.show()

    return (first_states, actions, next_states, rewards)



'''
Issues:
    - With supports: actual distributions (for policy and states) are different from the ones used here; the actual ones
    have higher densities on the borders. This is a problem because the collected samples follow that distribution and not
    the assumed one.
    - Transitions that zero the velocity: actual distribution (for states) will have higher density for zero velocity
'''
def calculate_density_ratios_dseta(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    action_idx = dataset['ai']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]
    target_sample_probs = target_task.env.dseta_distr[state_idx, action_idx]
    return target_sample_probs/source_sample_probs



def calculate_density_ratios_transition_sa(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    action_idx = dataset['ai']
    next_state_idx = dataset['nsi']
    next_action_idx = dataset['nai']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs =\
            source_task.env.transition_matrix[state_idx, action_idx, next_state_idx] * source_policy.choice_matrix[next_state_idx, next_action_idx]
    target_sample_probs = \
        target_task.env.transition_matrix[state_idx, action_idx, next_state_idx] * target_policy.choice_matrix[next_state_idx, next_action_idx]
    return target_sample_probs/source_sample_probs



def calculate_density_ratios_r_sa(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    action_idx = dataset['ai']
    next_state_idx = dataset['nsi']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = source_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
    target_sample_probs = target_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
    return target_sample_probs / source_sample_probs



def calculate_density_ratios_delta(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = source_task.env.delta_distr[state_idx]
    target_sample_probs = target_task.env.delta_distr[state_idx]
    return target_sample_probs/source_sample_probs



def calculate_density_ratios_transition_s(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    next_state_idx = dataset['nsi']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = \
            (source_policy.choice_matrix[state_idx, :] * source_task.env.transition_matrix[state_idx, :, next_state_idx]).sum(axis=1)
    target_sample_probs = \
        (target_policy.choice_matrix[state_idx, :] * target_task.env.transition_matrix[state_idx,:,next_state_idx]).sum(axis=1)
    return target_sample_probs/source_sample_probs



def calculate_density_ratios_r_s(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset['fsi']
    action_idx = dataset['ai']
    next_state_idx = dataset['nsi']
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = source_policy.choice_matrix[state_idx, action_idx]*source_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
    target_sample_probs = target_policy.choice_matrix[state_idx, action_idx] * target_task.env.transition_matrix[state_idx, action_idx, next_state_idx]
    return target_sample_probs / source_sample_probs



# No need to calculate reward of source samples on source model, we need reward of source samples in target model only
def estimate_J(dataset, gamma, task=None, weights=None):
    if task is None:
        rewards = dataset[3].copy()
        if weights is not None:
            rewards *= weights
        J = rewards.mean()/(1-gamma)
        return J
    else:
        '''R = np.zeros(task.env.R.shape, dtype=np.float64)
        dseta = np.zeros(task.env.R.shape, dtype=np.float64)
        for i in range(len(dataset[0])):
            R[task.env.state_to_idx[dataset[0][i][0]][dataset[0][i][1]], task.env.action_to_idx[dataset[1][i][0]]] += dataset[3][i]
            dseta[task.env.state_to_idx[dataset[0][i][0]][dataset[0][i][1]], task.env.action_to_idx[dataset[1][i][0]]] += 1
        counts = dseta.copy()
        counts[counts == 0] = 1.
        R /= counts
        dseta /= dseta.sum()
        J1 = np.average(R, weights=dseta)/(1-gamma)'''
        state_idx = np.array([task.env.state_to_idx[s[0]][s[1]] for s in dataset[0]])
        action_idx = np.array([task.env.action_to_idx[a[0]] for a in dataset[1]])
        rewards = task.env.R[state_idx, action_idx]
        J = rewards.mean()/(1-gamma)
        return J



def optimize_parameters(target_size, target_task, pf, lstd_q, lstd_v, source_policy=None, source_samples=None,
                        source_sample_probs_d_sa=None, source_sample_probs_p_sa=None, source_sample_probs_r_sa=None,
                        source_sample_probs_d_s=None, source_sample_probs_p_s=None, source_sample_probs_r_s=None,
                        phi_source_q=None, phi_ns_source_q=None, phi_source_v=None, phi_ns_source_v=None):
    step_size = 0.01
    max_iters = 2000
    i = 1
    np.random.seed(seed)
    grad = np.zeros(2, dtype=np.float64)
    grad_norm = 1.
    alpha1 = np.random.uniform()
    alpha2 = np.random.uniform()

    while grad_norm > 1e-3 and i <= max_iters:
        alpha1 += step_size * grad[0]
        alpha2 += step_size * grad[1]
        alpha1 = max(min(alpha1, 1.0), 0.0)
        alpha2 = max(min(alpha2, 1.0), 0.0)
        pol = pf.create_policy(alpha1, alpha2)
        target_samples = collect_samples(target_task, target_size, seed, pol)
        if source_samples is not None:
            weights_d_sa = calculate_density_ratios_dseta(source_samples, source_task, target_task, source_policy, pol,
                                                          source_sample_probs_d_sa)
            weights_d_sa = np.append(np.ones(target_size, dtype=np.float64), weights_d_sa)
            transfer_samples = {'fs':np.vstack((target_samples['fs'], source_samples['fs'])),
                                'a':np.vstack((target_samples['a'], source_samples['a'])),
                                'ns':np.vstack((target_samples['ns'], source_samples['ns'])),
                                'na':np.vstack((target_samples['na'], source_samples['na'])),
                                'r':np.concatenate((target_samples['r'], source_samples['r'])),
                                'fsi':np.concatenate((target_samples['fsi'], source_samples['fsi'])),
                                'ai':np.concatenate((target_samples['ai'], source_samples['ai'])),
                                'nsi':np.concatenate((target_samples['nsi'], source_samples['nsi'])),
                                'nai':np.concatenate((target_samples['nai'], source_samples['nai']))}
            if lstd_v is not None and lstd_q is not None:
                weights_p_sa = calculate_density_ratios_transition_sa(source_samples, source_task, target_task,
                                                                      source_policy, pol, source_sample_probs_p_sa)
                weights_p_sa = np.append(np.ones(target_size, dtype=np.float64), weights_p_sa)
                weights_r_sa = calculate_density_ratios_r_sa(source_samples, source_task, target_task,
                                                             source_policy, pol, source_sample_probs_r_sa)
                weights_r_sa = np.append(np.ones(target_size, dtype=np.float64), weights_r_sa)
                weights_d_s = calculate_density_ratios_delta(source_samples, source_task, target_task, source_policy,
                                                             pol, source_sample_probs_d_s)
                weights_d_s = np.append(np.ones(target_size, dtype=np.float64), weights_d_s)
                weights_p_s = calculate_density_ratios_transition_s(source_samples, source_task, target_task,
                                                                    source_policy, pol, source_sample_probs_p_s)
                weights_p_s = np.append(np.ones(target_size, dtype=np.float64), weights_p_s)
                weights_r_s = calculate_density_ratios_r_s(source_samples, source_task, target_task,
                                                           source_policy, pol, source_sample_probs_r_s)
                weights_r_s = np.append(np.ones(target_size, dtype=np.float64), weights_r_s)
                lstd_q.fit_slow(transfer_samples, weights_d=weights_d_sa, weights_p=weights_p_sa, weights_r=weights_r_sa,
                                phi_source=phi_source_q, phi_ns_source=phi_ns_source_q)
                Qs = lstd_q.transform(transfer_samples['fs'], transfer_samples['a'], phi_source=phi_source_q)
                lstd_v.fit_slow(transfer_samples, weights_d=weights_d_s, weights_p=weights_p_s, weights_r=weights_r_s,
                                phi_source=phi_source_v, phi_ns_source=phi_ns_source_v)
                Vs = lstd_v.transform(transfer_samples['fs'], phi_source=phi_source_v)
            else:
                Qs = target_task.env.Q[transfer_samples['fsi'], transfer_samples['ai']]
                Vs = target_task.env.V[transfer_samples['fsi']]
            grad = estimate_gradient(transfer_samples, pol, Qs, Vs, weights=weights_d_sa, baseline_type=1)
        else:
            if lstd_v is not None and lstd_q is not None:
                lstd_q.fit_slow(target_samples)
                Qs = lstd_q.transform(target_samples['fs'], target_samples['a'])
                lstd_v.fit_slow(target_samples)
                Vs = lstd_v.transform(target_samples['fs'])
            else:
                Qs = target_task.env.Q[target_samples['fsi'], target_samples['ai']]
                Vs = target_task.env.V[target_samples['fsi']]
            grad = estimate_gradient(target_samples, pol, Qs, Vs, baseline_type=1)
        '''if i % 100 == 0 or i == 1:
            g = pol.log_gradient_matrix.copy()
            g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q * target_task.env.dseta_distr)
            g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))
            print(grad, g, alpha1, alpha2)'''
        grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                          (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
        grad_norm = np.linalg.norm(grad)
        #grad /= grad_norm if grad_norm != 0 else 1
        i += 1
        step_size -= (0.01-0.001)/max_iters
    if i > max_iters:
        print("Did not converge")
        print(grad_norm, i)
    return alpha1, alpha2



def estimate_Q_TD(dataset, gamma, policy, episodic, lam):
    e = np.zeros(policy.choice_matrix.shape, dtype=np.float64)
    Q = e.copy()
    alpha = 0.1
    if episodic:
        con = 1
        for ep in range(dataset[0].shape[0]):
            e = np.zeros(policy.choice_matrix.shape, dtype=np.float64)
            alpha = 0.1
            first_states = dataset[0][ep]
            actions = dataset[1][ep]
            next_states = dataset[2][ep]
            rewards = dataset[3][ep]
            for t in range(first_states.shape[0]):
                if rewards[t] == 1.:
                    break
                con += 1
                fs_idx = source_task.env.state_to_idx[first_states[t][0]][first_states[t][1]]
                a_idx = source_task.env.action_to_idx[actions[t]]
                ns_idx = source_task.env.state_to_idx[next_states[t][0]][next_states[t][1]]
                if t == first_states.shape[0] - 1 or rewards[t+1] == 1.:
                    na_idx = source_task.env.action_to_idx[policy.produce_action(next_states[t])]
                else:
                    na_idx = source_task.env.action_to_idx[actions[t+1]]
                delta = rewards[t] + gamma*Q[ns_idx, na_idx] - Q[fs_idx, a_idx]
                e[fs_idx, a_idx] += 1
                Q += alpha*delta*e
                e *= gamma*lam
    else:
        first_states = dataset[0]
        actions = dataset[1]
        next_states = dataset[2]
        rewards = dataset[3]
        fs_idx = dataset[4]
        a_idx = dataset[5]
        ns_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in next_states])
        na_idx = np.array([np.random.choice(policy.factory.action_reps.shape[0], p=policy.choice_matrix[ns])
                                     for ns in ns_idx])
        for t in range(first_states.shape[0]):
            delta = rewards[t] + gamma * Q[ns_idx[t], na_idx[t]] - Q[fs_idx[t], a_idx[t]]
            e[fs_idx[t], a_idx[t]] += 1
            Q += alpha * delta * e
            e *= gamma * lam
    return Q




# No need to calculate gradient of source samples on source model, we need gradient of source samples in target model only
def estimate_gradient(dataset, policy, Q, V, weights=None, baseline_type=0):
    state_idx = dataset['fsi']
    action_idx = dataset['ai']
    grads = policy.log_gradient_matrix[state_idx, action_idx]
    if baseline_type == 0:
        grad = (grads.T * Q).T
        if weights is not None:
            grad = (grad.T * weights).T
        grad = grad.mean(axis=0)
    if baseline_type == 1:
        grad = (grads.T * Q).T
        baseline = V
        grad = grad - (grads.T*baseline).T
        if weights is not None:
            grad  = (grad .T * weights).T
        grad = grad.mean(axis=0)
    if baseline_type == 2:
        den = grads ** 2
        grad = (grads.T * Q).T
        baseline = grad ** 2
        if weights is not None:
            baseline = (baseline.T * weights).T
            den = (den.T * weights).T
        baseline = baseline.mean(axis=0)
        den = den.mean(axis=0)
        den[den == 0] = 1
        baseline /= den
        grad  = grad - grads * baseline
        if weights is not None:
            grad = (grad .T * weights).T
        grad = grad.mean(axis=0)
    return grad






gamma = 0.99
min_pos = -10.
max_pos = 10.
min_act = -1.0
max_act = -min_act
seed = 9876
power_source = rescale_state(0.0015)
power_target = rescale_state(0.002)
alpha_1_source = 0.
alpha_2_source = 0.
alpha_1_target = 1.
alpha_2_target = 1.
action_noise = (max_act - min_act)*0.2
max_episode_length = 200
n_source_samples = 50000
n_target_samples = 5000
n_action_bins = 10 + 1
n_position_bins = 20 + 1
n_velocity_bins = 20 + 1

# Creation of source task
source_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_source, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins, position_noise=0.025, velocity_noise=0.025)
# Creation of target task
target_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_target, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins, position_noise=0.025, velocity_noise=0.025)
# Policy factory
pf = PolicyFactoryMC(model='S', action_noise=action_noise, max_speed=source_task.env.max_speed, min_act=min_act, max_act=max_act,
                     min_pos=min_pos, bottom_pos=source_task.env.bottom_position, max_pos=max_pos,
                     action_bins=source_task.env.action_bins, action_reps=source_task.env.action_reps,
                     state_reps=source_task.env.state_reps, state_to_idx=source_task.env.state_to_idx)
# Defining source policy
source_policy = pf.create_policy(alpha_1_source, alpha_2_source)
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

lstd_q = LSTD_Q_Estimator(7, 7, 2, 0.45, True, gamma, 0., min_pos, max_pos, source_task.env.min_speed, source_task.env.max_speed,
                          min_act, max_act)
lstd_v = LSTD_V_Estimator(6, 6, 0.2, True, gamma, 0., min_pos, max_pos, source_task.env.min_speed, source_task.env.max_speed)

#collect_episodes(source_task, 10, max_episode_length, seed, source_policy, True)

'''source_task.env.set_policy(pf.create_policy(0., 0.), gamma)
a1 = source_task.env.J
source_task.env.set_policy(pf.create_policy(1., 0.), gamma)
a2 = source_task.env.J
source_task.env.set_policy(pf.create_policy(1., 1.), gamma)
a3 = source_task.env.J
source_task.env.set_policy(pf.create_policy(0., 1.), gamma)
a4 = source_task.env.J
source_task.env.set_policy(pf.create_policy(.5, .5), gamma)
a5 = source_task.env.J

target_task.env.set_policy(pf.create_policy(0., 0.), gamma)
b1 = target_task.env.J
target_task.env.set_policy(pf.create_policy(1., 0.), gamma)
b2 = target_task.env.J
target_task.env.set_policy(pf.create_policy(1., 1.), gamma)
b3 = target_task.env.J
target_task.env.set_policy(pf.create_policy(0., 1.), gamma)
b4 = target_task.env.J
target_task.env.set_policy(pf.create_policy(.5, .5), gamma)
b5 = target_task.env.J
# Collecting source episodes
print("Collecting", n_source_samples, "samples from source task...")
source_samples = collect_samples(source_task, n_source_samples, seed, source_policy, False)
source_task.close()
print("Done sampling from source task")

# Collecting target episodes
print("Collecting", n_target_samples, "samples from target task...")
target_samples = collect_samples(target_task, n_target_samples, seed, target_policy, False)
target_task.close()
print("Done sampling from target task")


J_1, var_J_1 = estimate_J(source_samples, gamma)
print("Estimated performance on source task:", J_1,"; empirical variance of estimate:",var_J_1)

J_2, var_J_2 = estimate_J(target_samples, gamma)
print("Estimated performance on target task:", J_2,"; empirical variance of estimate:",var_J_2)

reduced_target_samples = (target_samples[0][:10], target_samples[1][:10], target_samples[2][:10], target_samples[3][:10])
J_1_reduced, var_J_1_reduced = estimate_J(reduced_target_samples, gamma)
print("Estimated performance on target task (reduced dataset):", J_1_reduced,"; empirical variance of estimate:",var_J_1_reduced)

transfer_samples = (np.vstack((reduced_target_samples[0],source_samples[0])), np.vstack((reduced_target_samples[1],source_samples[1])), np.vstack((reduced_target_samples[2],source_samples[2])), np.concatenate((reduced_target_samples[3],source_samples[3])))
weights = calculate_density_ratios(source_samples, source_task, target_task, source_policy, target_policy)
weights = np.append(np.ones(10, dtype=np.float64), weights)
J_1_is, var_J_1_is = estimate_J(transfer_samples, gamma, weights=weights)
print("Estimated performance on target task with transfer:", J_1_is, "; empirical variance of estimate:", var_J_1_is)'''

m1 = source_task.env.state_reps[:,0] < source_task.env.max_position
xs = source_task.env.state_reps[:,0][m1]
ys = source_task.env.state_reps[:,1][m1]

'''source_task.env.set_policy(source_policy, gamma)
for act in range(n_action_bins - 1):
    zs = source_task.env.Q[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.velocity_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()
plt.close()
zs = source_task.env.V[m1].flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
aux = int(m1.sum() / source_task.env.velocity_reps.shape[0])
sur = ax.plot_surface(xs.reshape((-1,aux)), ys.reshape((-1,aux)), zs.reshape((-1,aux)))
plt.show()
plt.close()'''

'''idx_grid = np.dstack(np.meshgrid(np.arange(source_task.env.state_reps.shape[0]),
                                 np.arange(source_task.env.action_reps.shape[0]),
                                 indexing='ij')).reshape(-1, 2)
for nk in range(2, 10+1):
    for eps in np.arange(0.01, 1.01, 0.01):
        lstd_q = LSTD_Q_Estimator(nk, nk, 2, eps, True, gamma, 0., min_pos, max_pos, source_task.env.min_speed,
                                  source_task.env.max_speed, min_act, max_act)
        lstd_q.calculate_theta(source_task, source_policy)
        Q = lstd_q.transform(source_task.env.state_reps[idx_grid[:,0]], source_task.env.action_reps[idx_grid[:,1]]).reshape(source_task.env.Q.shape)
        print(nk, eps, (((Q - source_task.env.Q) ** 2) * source_task.env.dseta_distr).sum())
for nk in range(2, 10+1):
    for eps in np.arange(0.01, 1.01, 0.01):
        lstd_v = LSTD_V_Estimator(nk, nk, eps, True, gamma, 0., min_pos, max_pos, source_task.env.min_speed,
                                  source_task.env.max_speed)
        lstd_v.calculate_theta(source_task, source_policy)
        V = lstd_v.transform(source_task.env.state_reps)
        print(nk, eps, (((V - source_task.env.V) ** 2) * source_task.env.delta_distr).sum())
lstd_q.calculate_theta(source_task, source_policy)
Q = lstd_q.transform(source_task.env.state_reps[idx_grid[:,0]], source_task.env.action_reps[idx_grid[:,1]]).reshape(source_task.env.Q.shape)
for act in range(n_action_bins - 1):
    zs = Q[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-30,120)
    aux = int(m1.sum() / source_task.env.velocity_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()
plt.close()'''

'''a = collect_samples(source_task, 10000, seed, source_policy)
w = calculate_density_ratios_dseta(a, source_task, target_task, source_policy, target_policy)
w2 = calculate_density_ratios_transition_sa(a, source_task, target_task, source_policy, target_policy)
w3 = calculate_density_ratios_delta(a, source_task, target_task, source_policy, target_policy)
w4 = calculate_density_ratios_transition_s(a, source_task, target_task, source_policy, target_policy)'''

sys.stdout = open('out.log', 'w')

np.random.seed(seed)

target_sizes = list(range(100, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 70000 + 1, 10000))
n_seeds = 5
results_noIS = np.zeros((n_seeds, len(target_sizes), 3), dtype=np.float64)
results_noIS_app = np.zeros((n_seeds, len(target_sizes), 3), dtype=np.float64)
results_IS = np.zeros((n_seeds, len(target_sizes), 3), dtype=np.float64)
results_IS_app = np.zeros((n_seeds, len(target_sizes), 3), dtype=np.float64)
for i in range(n_seeds):
    seed = int(np.random.uniform(high=2**32))

    if i != 0:
        for idx, target_size in enumerate(target_sizes):
            alpha_1_target_opt, alpha_2_target_opt = optimize_parameters(target_size, target_task, pf, None, None)
            optimal_pi = pf.create_policy(alpha_1_target_opt, alpha_2_target_opt)
            target_task.env.set_policy(optimal_pi, gamma)
            J1_opt = target_task.env.J
            results_noIS[i, idx] = np.array([alpha_1_target_opt, alpha_1_target_opt, J1_opt], dtype=np.float64)
            print("No IS: TS", target_size, "a1", alpha_1_target_opt, "a2", alpha_2_target_opt, "J", J1_opt)
            sys.stdout.flush()

        for idx, target_size in enumerate(target_sizes):
            alpha_1_target_opt, alpha_2_target_opt = optimize_parameters(target_size, target_task, pf, lstd_q, lstd_v)
            optimal_pi = pf.create_policy(alpha_1_target_opt, alpha_2_target_opt)
            target_task.env.set_policy(optimal_pi, gamma)
            J1_opt = target_task.env.J
            results_noIS_app[i, idx] = np.array([alpha_1_target_opt, alpha_1_target_opt, J1_opt], dtype=np.float64)
            print("No IS app: TS", target_size, "a1", alpha_1_target_opt, "a2", alpha_2_target_opt, "J", J1_opt)
            sys.stdout.flush()

    '''results_J = []
    results_G = []
    results_G_b1 = []
    results_G_b2 = []
    for j in range(25):
        results_J.append([])
        results_G.append([])
        results_G_b1.append([])
        results_G_b2.append([])
        seed = int(np.random.uniform(high=2 ** 32))
        source_samples = collect_samples(source_task, int(1e5), seed, source_policy, False)
        source_task.env.set_policy(source_policy, gamma)
        state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in source_samples[0]])
        action_idx = np.array([source_task.env.action_to_idx[a[0]] for a in source_samples[1]])
        source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]
        w = calculate_density_ratios(source_samples, source_task, target_task, source_policy, target_policy, source_sample_probs=source_sample_probs)
        target_samples = collect_samples(target_task, int(1e5), seed, target_policy, False)

        J = calculate_J(target_task, target_policy, gamma)

        g = target_policy.log_gradient_matrix.copy()
        g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q*target_task.env.dseta_distr)
        g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))
        for i, size in enumerate([1e4, 2e4, 3e4, 4e4, 5e4, 1e5]):
            print(size)
            J_1, _ = estimate_J((source_samples[0][:int(size)], source_samples[1][:int(size)], source_samples[2][:int(size)], source_samples[3][:int(size)]),
                                gamma)
            J_2, _ = estimate_J((target_samples[0][:int(size)], target_samples[1][:int(size)], target_samples[2][:int(size)], target_samples[3][:int(size)]), gamma)
            J_2_is, _ = estimate_J((source_samples[0][:int(size)], source_samples[1][:int(size)], source_samples[2][:int(size)], source_samples[3][:int(size)]), gamma, weights=w[:int(size)])
            results_J[j].append([J_1, J_2, J_2_is])
            print(J_1, J_2, J_2_is, J)

            g_2 = estimate_gradient((target_samples[0][:int(size)], target_samples[1][:int(size)], target_samples[2][:int(size)], target_samples[3][:int(size)]),
                                    target_policy, target_task, baseline_type=0)
            g_2_is = estimate_gradient((source_samples[0][:int(size)], source_samples[1][:int(size)], source_samples[2][:int(size)], source_samples[3][:int(size)]),
                                       target_policy, target_task, baseline_type=0, weights=w[:int(size)])
            results_G[j].append([g_2, g_2_is])
            print(g_2, g_2_is, g)

            g_2 = estimate_gradient((target_samples[0][:int(size)], target_samples[1][:int(size)], target_samples[2][:int(size)],
                                     target_samples[3][:int(size)]),
                                    target_policy, target_task, baseline_type=1)
            g_2_is = estimate_gradient((source_samples[0][:int(size)], source_samples[1][:int(size)], source_samples[2][:int(size)],
                                        source_samples[3][:int(size)]),
                                       target_policy, target_task, baseline_type=1, weights=w[:int(size)])
            results_G_b1[j].append([g_2, g_2_is])
            print(g_2, g_2_is, g)

            g_2 = estimate_gradient((target_samples[0][:int(size)], target_samples[1][:int(size)], target_samples[2][:int(size)],
                                     target_samples[3][:int(size)]),
                                    target_policy, target_task, baseline_type=2)
            g_2_is = estimate_gradient((source_samples[0][:int(size)], source_samples[1][:int(size)], source_samples[2][:int(size)],
                                        source_samples[3][:int(size)]),
                                       target_policy, target_task, baseline_type=2, weights=w[:int(size)])
            results_G_b2[j].append([g_2, g_2_is])
            print(g_2, g_2_is, g)

    results_J = np.array(results_J)
    results_G = np.array(results_G)
    results_G_b1 = np.array(results_G_b1)
    results_G_b2 = np.array(results_G_b2)
    print(results_J.var(axis=0))
    print(results_G.var(axis=0))
    print(results_G_b1.var(axis=0))
    print(results_G_b2.var(axis=0))'''

    source_samples = collect_samples(source_task, n_source_samples, seed, source_policy)
    source_sample_probs_d_sa = source_task.env.dseta_distr[source_samples['fsi'], source_samples['ai']]

    if i != 0:
        for idx, target_size in enumerate(target_sizes):
            alpha_1_target_opt_transfer, alpha_2_target_opt_transfer =\
                optimize_parameters(target_size, target_task, pf, None, None, source_policy, source_samples,
                                    source_sample_probs_d_sa)
            optimal_pi_transfer = pf.create_policy(alpha_1_target_opt_transfer, alpha_2_target_opt_transfer)
            target_task.env.set_policy(optimal_pi_transfer, gamma)
            J1_opt_transfer = target_task.env.J
            results_IS[i,idx] = np.array([alpha_1_target_opt_transfer, alpha_2_target_opt_transfer, J1_opt_transfer], dtype=np.float64)
            print("IS: TS", target_size, "a1", alpha_1_target_opt_transfer, "a2", alpha_2_target_opt_transfer, "J", J1_opt_transfer)
            sys.stdout.flush()

    source_sample_probs_p_sa = source_task.env.transition_matrix[source_samples['fsi'], source_samples['ai'], source_samples['nsi']] * source_policy.choice_matrix[source_samples['nsi'], source_samples['nai']]
    source_sample_probs_r_sa = source_task.env.transition_matrix[source_samples['fsi'], source_samples['ai'], source_samples['nsi']]
    source_sample_probs_d_s = source_task.env.delta_distr[source_samples['fsi']]
    source_sample_probs_p_s = (source_policy.choice_matrix[source_samples['fsi'],:] * source_task.env.transition_matrix[source_samples['fsi'],:,source_samples['nsi']]).sum(axis=1)
    source_sample_probs_r_s = source_policy.choice_matrix[source_samples['fsi'], source_samples['ai']] * source_task.env.transition_matrix[source_samples['fsi'], source_samples['ai'], source_samples['nsi']]
    phi_source_q = lstd_q.map_to_feature_space(source_samples['fs'], source_samples['a'])
    phi_ns_source_q = lstd_q.map_to_feature_space(source_samples['ns'], source_samples['na'])
    phi_source_v = lstd_v.map_to_feature_space(source_samples['fs'])
    phi_ns_source_v = lstd_v.map_to_feature_space(source_samples['ns'])

    for idx, target_size in enumerate(target_sizes):
        if i != 0 or target_size != 100:
            alpha_1_target_opt_transfer, alpha_2_target_opt_transfer = \
                optimize_parameters(target_size, target_task, pf, lstd_q, lstd_v, source_policy, source_samples,
                                        source_sample_probs_d_sa, source_sample_probs_p_sa, source_sample_probs_r_sa,
                                        source_sample_probs_d_s, source_sample_probs_p_s, source_sample_probs_r_s,
                                        phi_source_q, phi_ns_source_q, phi_source_v, phi_ns_source_v)
            optimal_pi_transfer = pf.create_policy(alpha_1_target_opt_transfer, alpha_2_target_opt_transfer)
            target_task.env.set_policy(optimal_pi_transfer, gamma)
            J1_opt_transfer = target_task.env.J
            results_IS_app[i, idx] = np.array([alpha_1_target_opt_transfer, alpha_2_target_opt_transfer, J1_opt_transfer],dtype=np.float64)
            print("IS app: TS", target_size, "a1", alpha_1_target_opt_transfer, "a2", alpha_2_target_opt_transfer, "J", J1_opt_transfer)
            sys.stdout.flush()

np.save('learning_noIS', np.array(results_noIS))
np.save('learning_noIS_app', np.array(results_noIS_app))
np.save('learning_IS', np.array(results_IS))
np.save('learning_IS_app', np.array(results_IS_app))