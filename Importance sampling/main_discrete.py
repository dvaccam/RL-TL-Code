import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PolicyFactoryMC import PolicyFactoryMC, PolicyMC
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.dummy import Pool
import copy
import threading as thr
import sys



def rescale_state(x):
    return x * (max_pos - min_pos) / 1.8



def _collect_sample(i, task, max_episode_length, policy, first_states, actions, next_states, rewards, lock):
    lock.acquire()
    init_state = task.reset()
    task_c = copy.deepcopy(task)
    lock.release()
    first_state = init_state
    for t in range(max_episode_length):
        action = np.array([policy.produce_action(first_state)])
        next_state, reward, done, info = task_c.step(action)
        first_states[i*2*max_episode_length + 2*t], first_states[i*2*max_episode_length + 2*t + 1] = first_state
        actions[i*max_episode_length + t] = action
        next_states[i*2*max_episode_length + 2*t], next_states[i*2*max_episode_length + 2*t + 1] = next_state
        rewards[i*max_episode_length + t] = reward
        first_state = next_state
        if done:
            break
    if t + 1 < max_episode_length:
        rewards[t + 1] = 1
    return t



def collect_samples(task, n_samples, seed, policy, include_terminal=True):
    np.random.seed(seed)
    task.env._seed(seed)
    task.env.set_policy(policy, gamma)
    return task.env.sample_step(n_samples, True, include_terminal)



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
def calculate_density_ratios(dataset, source_task, target_task, source_policy, target_policy, source_sample_probs=None):
    target_task.env.set_policy(target_policy, gamma)
    state_idx = dataset[4]
    action_idx = dataset[5]
    if source_sample_probs is None:
        source_task.env.set_policy(source_policy, gamma)
        source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]
    target_sample_probs = target_task.env.dseta_distr[state_idx, action_idx]
    return target_sample_probs/source_sample_probs



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



def optimize_parameters(target_size, target_task, pf, source_policy=None, source_samples=None, source_sample_probs=None):
    step_size = 0.01
    max_iters = 2000
    i = 1
    np.random.seed(seed)
    grad = np.zeros(2, dtype=np.float64)
    grad_norm = 1
    alpha1 = source_policy.alpha1 if source_policy is not None else np.random.uniform()
    alpha2 = source_policy.alpha2 if source_policy is not None else np.random.uniform()

    while grad_norm > 1e-3 and i <= max_iters:
        alpha1 += step_size * grad[0]
        alpha2 += step_size * grad[1]
        alpha1 = max(min(alpha1, 1.0), 0.0)
        alpha2 = max(min(alpha2, 1.0), 0.0)
        pol = pf.create_policy(alpha1, alpha2)
        target_samples = collect_samples(target_task, target_size, seed, pol)
        if source_samples is not None:
            weights = calculate_density_ratios(source_samples, source_task, target_task, source_policy, pol,
                                               source_sample_probs)
            weights = np.append(np.ones(target_size, dtype=np.float64), weights)
            transfer_samples = (np.vstack((target_samples[0], source_samples[0])),
                                np.vstack((target_samples[1], source_samples[1])),
                                np.vstack((target_samples[2], source_samples[2])),
                                np.concatenate((target_samples[3], source_samples[3])),
                                np.concatenate((target_samples[4], source_samples[4])),
                                np.concatenate((target_samples[5], source_samples[5])))
            grad = estimate_gradient(transfer_samples, pol, target_task, weights, baseline_type=1)
        else:
            Q = estimate_Q(target_samples, gamma, pol)
            grad = estimate_gradient(target_samples, pol, target_task, baseline_type=1)
        '''if i % 100 == 0 or i == 1:
            J = calculate_J(target_task, pol, gamma)
            ts = collect_samples(target_task, 10000, seed, pol, False)
            g = estimate_gradient(ts, pol, target_task)
            print(grad, g)
            print(alpha1, alpha2, grad_norm, J)'''
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



def map_to_feature_space(s, a):
    p1 = np.array([source_task.env.position_reps[0]])
    p2 = np.array([source_task.env.position_reps[-1]])
    p3 = np.array([(p2 + p1) / 2]).flatten()
    v1 = np.array([source_task.env.velocity_reps[0]])
    v2 = np.array([source_task.env.velocity_reps[-1]])
    v3 = np.array([(v2 + v1) / 2]).flatten()
    epsp = rescale_state(1.8 * 7)
    epsv = rescale_state(.014 * 7)
    if s.ndim == 2:
        dp1 = np.abs(s[:,0] - p1)
        dp2 = np.abs(s[:,0] - p2)
        dp3 = np.abs(s[:,0] - p3)
        dv1 = np.abs(s[:,1] - v1)
        dv2 = np.abs(s[:,1] - v2)
        dv3 = np.abs(s[:,1] - v3)
        d = np.stack((dp1 ** 2 / epsp + dv1 ** 2 / epsv + 1, dp1 ** 2 / epsp + dv2 ** 2 / epsv + 1,
                      dp2 ** 2 / epsp + dv1 ** 2 / epsv + 1, dp2 ** 2 / epsp + dv2 ** 2 / epsv + 1,
                      dp3 ** 2 / epsp + dv3 ** 2 / (2 * epsv))).T
        d = np.stack((np.exp(-dp1 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp1 ** 2 / epsp - dv2 ** 2 / epsv),
                      np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2 ** 2 / epsp - dv2 ** 2 / epsv),
                      dp3 ** 2 / epsp + dv3 ** 2 / (2 * epsv))).T
        phi = np.hstack((1. / (1. + d[:, :-1]), (d[:, -1] / (1. + d[:, -1])).reshape((-1, 1))))[:, :-1]
        phi = np.hstack((d[:, :-1], (d[:, -1] / (1. + d[:, -1])).reshape((-1, 1))))[:, :-1]
    else:
        dp1 = np.abs(s[0] - p1)
        dp2 = np.abs(s[0] - p2)
        dp3 = np.abs(s[0] - p3)
        dv1 = np.abs(s[1] - v1)
        dv2 = np.abs(s[1] - v2)
        dv3 = np.abs(s[1] - v3)
        d = np.array([dp1 ** 2 / epsp + dv1 ** 2 / epsv + 1, dp1 ** 2 / epsp + dv2 ** 2 / epsv + 1,
                      dp2 ** 2 / epsp + dv1 ** 2 / epsv + 1, dp2 ** 2 / epsp + dv2 ** 2 / epsv + 1,
                      dp3 ** 2 / epsp + dv3 ** 2 / (2 * epsv)]).flatten()
        d = np.array([np.exp(-dp1 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp1 ** 2 / epsp - dv2 ** 2 / epsv),
                      np.exp(-dp2 ** 2 / epsp - dv1 ** 2 / epsv), np.exp(-dp2 ** 2 / epsp - dv2 ** 2 / epsv),
                      dp3 ** 2 / epsp + dv3 ** 2 / (2 * epsv)]).flatten()
        phi = np.append(1. / (1. + d[:-1]), d[-1] / (1. + d[-1]))[:-1]
        phi = np.append(d[:-1], d[-1] / (1. + d[-1]))[:-1]
    return phi



def estimate_Q_TD_fa(dataset, gamma, policy, episodic, lam):
    n_feats = 4
    theta = np.zeros(n_feats, dtype=np.float64)
    theta = np.array([50., 50., 60., 60.])
    A = np.zeros((n_feats,n_feats), dtype=np.float64)
    b = np.zeros(n_feats, dtype=np.float64)
    if episodic:
        con = 1
        for ep in range(dataset[0].shape[0]):
            first_states = dataset[0][ep]
            actions = dataset[1][ep]
            next_states = dataset[2][ep]
            rewards = dataset[3][ep]
            for t in range(first_states.shape[0]):
                if rewards[t] == 1.:
                    break
                con += 1
                phi = map_to_feature_space(first_states[t], actions[t])
                if t != first_states.shape[0] - 1 and rewards[t + 1] != 1.:
                    phi_ns = map_to_feature_space(next_states[t], actions[t+1])
                else:
                    phi_ns = map_to_feature_space(next_states[t], policy.produce_action(next_states[t]))
                if t == 0:
                    z = phi.copy()
                A += z.reshape((-1,1)).dot((phi - gamma*phi_ns).reshape((1,-1)))
                b += z*rewards[t]
                z = lam*gamma*z + phi_ns
    else:
        first_states = dataset[0]
        actions = dataset[1]
        next_states = dataset[2]
        rewards = dataset[3]
        dseta = np.zeros(source_task.env.R.shape, dtype=np.float64)
        tm = np.zeros((source_task.env.state_reps.shape[0],
                       source_task.env.action_reps.shape[0],
                       source_task.env.state_reps.shape[0]), dtype=np.float64)
        R = dseta.copy()
        for t in range(first_states.shape[0]):
            phi = map_to_feature_space(first_states[t], actions[t])
            phi_ns = map_to_feature_space(next_states[t], policy.produce_action(next_states[t]))
            if t == 0:
                z = phi.copy()
            else:
                z = lam * gamma * z + phi
            A += z.reshape((-1, 1)).dot((phi - gamma * phi_ns).reshape((1, -1)))
            b += z * rewards[t]
    theta = np.linalg.inv(A).dot(b)
    print(theta)
    def Q(s, a):
        phi_s = map_to_feature_space(s, a)
        return theta.dot(phi_s)
    return Q



def estimate_Q(dataset, gamma, policy):
    next_state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in dataset[2]])
    next_actions_idx = np.array([np.random.choice(policy.factory.action_reps.shape[0], p=policy.choice_matrix[ns])
                                 for ns in next_state_idx])
    lam = 0.7
    phi = map_to_feature_space(dataset[0], dataset[1])
    phi_next = map_to_feature_space(dataset[2], policy.factory.action_reps[next_actions_idx].reshape((-1,1)))
    gl = np.power(gamma*lam, np.arange(next_state_idx.shape[0])[::-1])
    z = (phi.T*gl).T.cumsum(axis=0)
    r = dataset[3]
    b = (z.T*r).T.mean(axis=0)
    a = (phi - gamma*phi_next)
    a = np.array([z[i].reshape((-1,1)).dot(a[i].reshape((1,-1))) for i in range(a.shape[0])]).mean(axis=0)
    theta = np.linalg.inv(a).dot(b)
    print(theta)
    def Q(s, a):
        phi_s = map_to_feature_space(s, a)
        return theta.dot(phi_s)
    return Q



def estimate_Q1(dataset, gamma, policy):
    next_state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in dataset[2]])
    next_actions_idx = np.array([np.random.choice(policy.factory.action_reps.shape[0], p=policy.choice_matrix[ns])
                                 for ns in next_state_idx])
    phi = map_to_feature_space(dataset[0], dataset[1])
    phi_next = map_to_feature_space(dataset[2], policy.factory.action_reps[next_actions_idx].reshape((-1,1)))
    a = phi.T.dot(phi - gamma*phi_next)
    r = dataset[3]
    b = phi.T.dot(r)
    theta = np.linalg.inv(a).dot(b)
    print(theta)
    def Q(s, a):
        phi_s = map_to_feature_space(s, a)
        return theta.dot(phi_s)
    return Q



def calculate_theta(task, gamma, policy):
    task.env.set_policy(policy, gamma)
    idx_grid = np.dstack(np.meshgrid(np.arange(task.env.state_reps.shape[0]), np.arange(task.env.action_reps.shape[0]), indexing='ij')).reshape(-1, 2)
    phi = map_to_feature_space(task.env.state_reps[idx_grid[:, 0]],
                              task.env.action_reps[idx_grid[:, 1]].reshape((-1,1)))
    D = np.diag(task.env.dseta_distr.flatten())
    dseta_pi = np.zeros((idx_grid.shape[0], idx_grid.shape[0]), dtype=np.float64)
    for i in range(dseta_pi.shape[0]):
        p = task.env.transition_matrix[idx_grid[i,0], idx_grid[i,1]]
        for j in range(dseta_pi.shape[1]):
            dseta_pi[i,j] = p[idx_grid[j,0]]*policy.choice_matrix[idx_grid[j,0], idx_grid[j,1]]
    A = phi.T.dot(D.dot(phi - gamma*dseta_pi.dot(phi)))
    R = task.env.R.flatten()
    b = phi.T.dot(D.dot(R))
    theta = np.linalg.inv(A).dot(b)
    print(theta)
    return theta




# No need to calculate gradient of source samples on source model, we need gradient of source samples in target model only
def estimate_gradient(dataset, policy, task, weights=None, baseline_type=0):
    state_idx = dataset[4]
    action_idx = dataset[5]
    grads = policy.log_gradient_matrix[state_idx, action_idx]
    Qs = task.env.Q[state_idx, action_idx]
    if baseline_type == 0:
        grad = (grads.T * Qs).T
        if weights is not None:
            grad = (grad.T * weights).T
        grad = grad.mean(axis=0)
    if baseline_type == 1:
        grad = (grads.T * Qs).T
        baseline = task.env.V[state_idx]
        grad = grad - (grads.T*baseline).T
        if weights is not None:
            grad  = (grad .T * weights).T
        grad  = grad .mean(axis=0)
    if baseline_type == 2:
        den = grads ** 2
        grad = (grads.T * Qs).T
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
            grad  = (grad .T * weights).T
        grad  = grad .mean(axis=0)
    return grad






gamma = 0.99
min_pos = -10.
max_pos = 10.
min_act = -1.0
max_act = -min_act
seed = 9876
power_source = rescale_state(0.5)
power_target = rescale_state(0.2)
alpha_1_source = 1.
alpha_2_source = 1.
alpha_1_target = 1.
alpha_2_target = 0.
action_noise = (max_act - min_act)*0.2
max_episode_length = 200
n_source_samples = 10000
n_target_samples = 5000
n_action_bins = 4 + 1
n_position_bins = 4 + 1
n_velocity_bins = 4 + 1

# Creation of source task
source_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_source, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins, position_noise=0.1, velocity_noise=0.1)
# Creation of target task
target_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_target, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins)
# Policy factory
pf = PolicyFactoryMC(model='S', action_noise=action_noise, max_speed=source_task.env.max_speed, min_act=min_act, max_act=max_act,
                     min_pos=min_pos, bottom_pos=source_task.env.bottom_position, max_pos=max_pos,
                     action_bins=source_task.env.action_bins, action_reps=source_task.env.action_reps,
                     state_reps=source_task.env.state_reps, state_to_idx=source_task.env.state_to_idx)
# Defining source policy
source_policy = pf.create_policy(alpha_1_source, alpha_2_source)
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

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

'''source_task.env.set_policy(source_policy, gamma)
m1 = source_task.env.state_reps[:,0] < source_task.env.goal_position
xs = source_task.env.state_reps[:,0][m1]
ys = source_task.env.state_reps[:,1][m1]
for act in range(n_action_bins-1):
    zs = source_task.env.Q[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1,aux)), ys.reshape((-1,aux)), zs.reshape((-1,aux)))
plt.show()

xs = source_task.env.state_reps[:30,0]
ys = source_task.env.action_reps
xs, ys = np.dstack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2).T
zs = source_task.env.Q[:30].flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_wireframe(xs, ys, zs)
#plt.show()

xs = source_task.env.state_reps[450:450+30,0]
ys = source_task.env.action_reps
xs, ys = np.dstack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2).T
zs = source_task.env.Q[450:450+30].flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_wireframe(xs, ys, zs)
#plt.show()

xs = source_task.env.state_reps[-30:,0]
ys = source_task.env.action_reps
xs, ys = np.dstack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2).T
zs = source_task.env.Q[-30:].flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sur = ax.plot_wireframe(xs, ys, zs)
plt.show()'''

m1 = source_task.env.state_reps[:,0] < source_task.env.goal_position
xs = source_task.env.state_reps[:,0][m1]
ys = source_task.env.state_reps[:,1][m1]
theta = calculate_theta(source_task, gamma, source_policy)
Q_opt = np.array([map_to_feature_space(source_task.env.state_reps[i], source_task.env.action_reps[j]).dot(theta)
                    for i in np.arange(source_task.env.state_reps.shape[0])
                    for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
for act in range(1):
    zs = Q_opt[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
#plt.show()
ds = collect_episodes(source_task, 1000, max_episode_length, seed, source_policy, False)
Q_td = estimate_Q_TD(ds, gamma, source_policy, True, 0.)
Q_f = estimate_Q_TD_fa(ds, gamma, source_policy, True, 0.)
Q_td_fa = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j])
                    for i in np.arange(source_task.env.state_reps.shape[0])
                    for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
ds = collect_samples(source_task, 1000*50, seed, source_policy, include_terminal=True)
Q_td1 = estimate_Q_TD(ds, gamma, source_policy, False, 0.)
Q_f = estimate_Q_TD_fa(ds, gamma, source_policy, False, 0.)
Q_td_fa1 = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j])
                    for i in np.arange(source_task.env.state_reps.shape[0])
                    for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
for act in range(1):
    zs = Q_td[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = Q_td_fa[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = Q_td_fa1[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()
Q_f = estimate_Q(ds, gamma, source_policy)
Q = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j])
              for i in np.arange(source_task.env.state_reps.shape[0])
              for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)


a = collect_samples(source_task, 1000*50, seed, source_policy, include_terminal=False)
Q_f = estimate_Q(a, gamma, source_policy)
Q = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
Q_f1 = estimate_Q1(a, gamma, source_policy)
Q1 = np.array([Q_f1(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
m1 = source_task.env.state_reps[:,0] < source_task.env.goal_position
xs = source_task.env.state_reps[:,0][m1]
ys = source_task.env.state_reps[:,1][m1]
for act in range(1):
    zs = Q[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = Q1[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()

a = collect_samples(source_task, 1000, seed, source_policy, include_terminal=False)
Q_f = estimate_Q(a, gamma, source_policy)
Q = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
Q_f1 = estimate_Q1(a, gamma, source_policy)
Q1 = np.array([Q_f1(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)

for act in range(1):
    zs = Q[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = Q1[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()

a = collect_samples(source_task, 10000, seed, source_policy, include_terminal=False)
Q_f = estimate_Q(a, gamma, source_policy)
Q = np.array([Q_f(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)
Q_f1 = estimate_Q1(a, gamma, source_policy)
Q1 = np.array([Q_f1(source_task.env.state_reps[i], source_task.env.action_reps[j]) for i in np.arange(source_task.env.state_reps.shape[0]) for j in np.arange(source_task.env.action_reps.shape[0])]).reshape(source_task.env.R.shape)

for act in range(1):
    zs = Q[m1,act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = Q1[m1, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(m1.sum() / source_task.env.action_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()

#b = collect_samples(target_task, 10000, 200, seed, target_policy, False)
#w = calculate_density_ratios(a, source_task, target_task, source_policy, target_policy)

#sys.stdout = open('out.log', 'w')

results_noIS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):
    alpha_1_target_opt, alpha_2_target_opt = optimize_parameters(target_size, target_task, pf)
    optimal_pi = pf.create_policy(alpha_1_target_opt, alpha_2_target_opt)
    target_task.env.set_policy(optimal_pi, gamma)
    J1_opt = target_task.env.J
    results_noIS.append([alpha_1_target_opt, alpha_1_target_opt, J1_opt])
    print("No IS: TS", target_size, "a1", alpha_1_target_opt, "a2", alpha_2_target_opt, "J", J1_opt)
    #sys.stdout.flush()
#np.save('learning_noIS', np.array(results_noIS))

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
state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in source_samples[0]])
action_idx = np.array([source_task.env.action_to_idx[a[0]] for a in source_samples[1]])
source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]

results_IS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):
    alpha_1_target_opt_transfer, alpha_2_target_opt_transfer =\
        optimize_parameters(target_size, target_task, pf, source_policy, source_samples, source_sample_probs)
    optimal_pi_transfer = pf.create_policy(alpha_1_target_opt_transfer, alpha_2_target_opt_transfer)
    target_task.env.set_policy(optimal_pi_transfer, gamma)
    J1_opt_transfer = target_task.env.J
    results_IS.append([alpha_1_target_opt_transfer, alpha_2_target_opt_transfer, J1_opt_transfer])
    print("IS: TS", target_size, "a1", alpha_1_target_opt_transfer, "a2", alpha_2_target_opt_transfer, "J", J1_opt_transfer)
#np.save('learning_IS', np.array(results_IS))