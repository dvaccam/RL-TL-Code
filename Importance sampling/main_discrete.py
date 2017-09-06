import gym
import numpy as np
import math
import time
from scipy.stats import norm, beta
import matplotlib.pyplot as plt
from PolicyFactoryMC import PolicyFactoryMC, PolicyMC
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.dummy import Pool
import copy
import threading as thr
import sys


def norm_pdf(x, loc=0, scale=1.0):
    return np.exp(((x - loc) / scale) ** 2) / (np.sqrt(2.0 * np.pi) * scale)



def beta_pdf(x, mu, sigma2, l, u):
    mu_std = (mu - l) / (u - l)
    sigma2_std = sigma2 / ((u - l) ** 2)
    a = (mu_std ** 2 - mu_std ** 3 - mu_std * sigma2_std) / sigma2_std
    b = (1 - mu_std) * a / mu_std
    return beta.pdf(x, a=a, b=b, loc=l, scale=u-l)



def rescale_state(x):
    return x * (max_pos - min_pos) / 1.8



def rescale_action(x):
    return (max_act - min_act) * x / 2.0



# Maps from [-1,1] to an appropriate mu in [min_act,max_act]
def transform_to_mu_action(x):
    mu = (max_act - min_act)*(x + 1)/2 + min_act
    eps_act = rescale_action(1e-2)
    mu_max = (min_act + max_act + np.sqrt((min_act + max_act) ** 2 - 4 * (min_act * max_act + action_noise ** 2))) / 2 - eps_act
    mu_min = (min_act + max_act - np.sqrt((min_act + max_act) ** 2 - 4 * (min_act * max_act + action_noise ** 2))) / 2 + eps_act
    return min(max(mu, mu_min), mu_max)



def _collect_sample(i, task, max_episode_length, policy, first_states, actions, next_states, rewards, lock):
    lock.acquire()
    init_state = task.reset()
    task_c = copy.deepcopy(task)
    lock.release()
    first_state = init_state
    for t in range(max_episode_length):
        # action = np.array([min(max(policy(first_state), min_act), max_act)])
        action = np.array([policy.produce_action(first_state)])
        if np.any(task_c.env.state != first_state):
            dfg=1
        next_state, reward, done, info = task_c.step(action)
        first_states[i*2*max_episode_length + 2*t], first_states[i*2*max_episode_length + 2*t + 1] = first_state
        actions[i*max_episode_length + t] = action
        next_states[i*2*max_episode_length + 2*t], next_states[i*2*max_episode_length + 2*t + 1] = next_state
        rewards[i*max_episode_length + t] = reward
        first_state = next_state
        if done:
            # aux[i] = t
            break
    if t + 1 < max_episode_length:
        rewards[t + 1] = 1
    return t



def collect_samples(task, n_samples, seed, policy, render):
    np.random.seed(seed)
    task.env._seed(seed)
    #counts = np.zeros(task.env.R.shape, dtype=np.float64)
    task.env.set_policy(policy, gamma)
    #aux = np.zeros(n_episodes, dtype=np.int64)
    # Sampling from task
    if True:
        # Data structures for storing samples
        first_states = []
        actions = []
        next_states = []
        rewards = []
        for i in range(n_samples):
            first_state, action, next_state, reward = task.env.sample_step()
            first_states.append(first_state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            #counts[task.env.state_to_idx[first_state[0]][first_state[1]], task.env.action_to_idx[action[0]]] += 1
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
    #counts *= (1-gamma if gamma != 1 else 1.)/n_samples
    return (np.array(first_states), np.array(actions), np.array(next_states), np.array(rewards))



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
                #action = np.array([min(max(policy(first_state), min_act), max_act)])
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
    state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in dataset[0]])
    action_idx = np.array([source_task.env.action_to_idx[a[0]] for a in dataset[1]])
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
        return J, 0
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
        return J, 0



def calculate_J(task, policy, gamma):
    P_pi = np.transpose(task.env.transition_matrix, axes=(2, 0, 1)).copy()
    P_pi = (P_pi * policy.choice_matrix).sum(axis=2).T
    R_pi = (task.env.R * policy.choice_matrix).sum(axis=1)
    V = np.eye(task.env.transition_matrix.shape[0]) - gamma * P_pi
    V = np.linalg.inv(V)
    J = V.dot(R_pi).dot(task.env.initial_state_distr)
    return J



# Note: empirical discounted distribution of states and the delta (with terminal states going to themselves and zero-reward)
# differ only in the terminal states, exactly because the ciclic transition is absent in the samples. This does not affect the
# final result as such states have zero reward
def calculate_delta_distr(task, policy, gamma):
    P_pi_T = np.transpose(task.env.transition_matrix, axes=(2, 0, 1)).copy()
    P_pi_T = (P_pi_T * policy.choice_matrix).sum(axis=2)
    delta_distr = np.eye(task.env.transition_matrix.shape[0]) - gamma*P_pi_T
    delta_distr = np.linalg.inv(delta_distr)
    delta_distr = (1-gamma if gamma != 1 else 1)*delta_distr.dot(task.env.initial_state_distr)
    return  delta_distr



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
        target_samples = collect_samples(target_task, target_size, seed, pol, False)
        if source_samples is not None:
            weights = calculate_density_ratios(source_samples, source_task, target_task, source_policy, pol,
                                               source_sample_probs)
            weights = np.append(np.ones(target_size, dtype=np.float64), weights)
            transfer_samples = (np.vstack((target_samples[0], source_samples[0])),
                                np.vstack((target_samples[1], source_samples[1])),
                                np.vstack((target_samples[2], source_samples[2])),
                                np.concatenate((target_samples[3], source_samples[3])))
            grad = estimate_gradient(transfer_samples, pol, target_task, weights, baseline_type=1)
        else:
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
        grad /= grad_norm if grad_norm != 0 else 1
        i += 1
        step_size -= (0.01-0.001)/max_iters
        #target_samples = collect_samples(target_task, 10000, max_episode_length, seed, pol, False)
    if i > max_iters:
        print("Did not converge")
        print(grad_norm, i)
    return alpha1, alpha2



# No need to calculate gradient of source samples on source model, we need gradient of source samples in target model only
def estimate_gradient(dataset, policy, task, weights=None, baseline_type=0):
    state_idx = np.array([task.env.state_to_idx[s[0]][s[1]] for s in dataset[0]])
    action_idx = np.array([task.env.action_to_idx[a[0]] for a in dataset[1]])
    grads = policy.gradient_matrix[state_idx, action_idx]
    Qs = task.env.Q[state_idx, action_idx]
    if baseline_type == 0:
        grad = (grads.T * Qs).T
        if weights is not None:
            grad = (grad.T * weights).T
        grad = grad.mean(axis=0)
    if baseline_type == 1:
        grad = (grads.T * Qs).T
        P_pi = np.transpose(task.env.transition_matrix, axes=(2, 0, 1)).copy()
        P_pi = (P_pi * policy.choice_matrix).sum(axis=2).T
        R_pi = (task.env.R * policy.choice_matrix).sum(axis=1)
        V = np.eye(task.env.transition_matrix.shape[0]) - gamma * P_pi
        V = np.linalg.inv(V).dot(R_pi)
        baseline = V[state_idx]
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






gamma = 0.95
min_pos = -10.
max_pos = 10.
min_act = -1.0
max_act = -min_act
seed = 9876
power_source = rescale_state(0.005)
power_target = rescale_state(0.0055)
alpha_1_source = 0.
alpha_2_source = 1.
alpha_1_target = 1.
alpha_2_target = 0.
n_source_episodes = 10000
action_noise = (max_act - min_act)*0.2
max_episode_length = 200
n_source_samples = 10000
n_target_samples = 5000

n_action_bins = 10 + 1
n_position_bins = 30 + 1
n_velocity_bins = 30 + 1

# Creation of source task
source_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_source, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins)
# Creation of target task
target_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_target, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins)
# Policy factory
pf = PolicyFactoryMC(model='S', action_noise=action_noise, max_speed=source_task.env.max_speed, min_act=min_act, max_act=max_act,
                     action_bins=source_task.env.action_bins, action_reps=source_task.env.action_reps,
                     state_reps=source_task.env.state_reps, state_to_idx=source_task.env.state_to_idx)
# Defining source policy
source_policy = pf.create_policy(alpha_1_source, alpha_2_source)
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

#collect_episodes(source_task, 10, max_episode_length, seed, source_policy, True)

'''xs = np.linspace(source_task.env.min_speed, source_task.env.max_speed, 1000)
ys1 = [source_task.env.transition_model_pdf(np.array([0,x]), np.array([0,0]), np.array([max_act]), True)[1] for x in xs]
ys2 = [target_task.env.transition_model_pdf(np.array([0,x]), np.array([0,0]), np.array([max_act]), True)[1] for x in xs]
plt.plot(xs, ys1, 'b', xs, ys2, 'r')
plt.show()'''


'''J_1_r = calculate_J(source_task, source_policy, gamma)
a = calculate_J(source_task, pf.create_policy(0., 0.), gamma)
b = calculate_J(source_task, pf.create_policy(1., 1.), gamma)
c = calculate_J(source_task, pf.create_policy(1., 0.), gamma)
d = calculate_J(source_task, pf.create_policy(.5, .5), gamma)
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


'''#r = np.load('results10000.npy')
#r1 = np.load('results1000.npy')
s = collect_samples(source_task, int(1000), 15, seed, source_policy, False)
w = calculate_density_ratios(s, source_task, target_task, source_policy, target_policy)
g = estimate_gradient(s, target_policy, weights=w)
g1 = estimate_gradient(s, target_policy, weights=w, baseline_type=1)
g2 = estimate_gradient(s, target_policy, weights=w, baseline_type=2)

s = collect_samples(source_task, int(1000), 15, 100, source_policy, False)
w = calculate_density_ratios(s, source_task, target_task, source_policy, target_policy)
g_ = estimate_gradient(s, target_policy, weights=w)
g1_ = estimate_gradient(s, target_policy, weights=w, baseline_type=1)
g2_ = estimate_gradient(s, target_policy, weights=w, baseline_type=2)

s = collect_samples(source_task, int(100000), 15, seed, source_policy, False)
w = calculate_density_ratios(s, source_task, target_task, source_policy, target_policy)
G = estimate_gradient(s, target_policy, weights=w)
G1 = estimate_gradient(s, target_policy, weights=w, baseline_type=1)
G2 = estimate_gradient(s, target_policy, weights=w, baseline_type=2)'''

'''g = estimate_gradient(t, target_policy)
g1 = estimate_gradient(t, target_policy, baseline_type=1)
g2 = estimate_gradient(t, target_policy, baseline_type=2)

t = collect_samples(target_task, int(1000), 10, 100, target_policy, False)
g_ = estimate_gradient(t, target_policy)
g1_ = estimate_gradient(t, target_policy, baseline_type=1)
g2_ = estimate_gradient(t, target_policy, baseline_type=2)

t = collect_samples(target_task, int(100000), 10, seed, target_policy, False)
G = estimate_gradient(t, target_policy)
G1 = estimate_gradient(t, target_policy, baseline_type=1)
G2 = estimate_gradient(t, target_policy, baseline_type=2)'''
#a = collect_samples(source_task, 10000, seed, source_policy, True)
#b = collect_samples(target_task, 10000, 200, seed, target_policy, False)
#w = calculate_density_ratios(a, source_task, target_task, source_policy, target_policy)

#sys.stdout = open('out.log', 'w')

results_noIS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):
    alpha_1_target_opt, alpha_2_target_opt = optimize_parameters(target_size, target_task, pf)
    optimal_pi = pf.create_policy(alpha_1_target_opt, alpha_2_target_opt)
    J1_opt = calculate_J(target_task, optimal_pi, gamma)
    results_noIS.append([alpha_1_target_opt, alpha_1_target_opt, J1_opt])
    print("No IS: TS", target_size, "a1", alpha_1_target_opt, "a2", alpha_2_target_opt, "J", J1_opt)
    #sys.stdout.flush()
#np.save('learning_noIS_1', np.array(results_noIS))

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
    
    g = target_policy.gradient_matrix.copy()
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

source_samples = collect_samples(source_task, n_source_samples, seed, source_policy, False)
state_idx = np.array([source_task.env.state_to_idx[s[0]][s[1]] for s in source_samples[0]])
action_idx = np.array([source_task.env.action_to_idx[a[0]] for a in source_samples[1]])
source_sample_probs = source_task.env.dseta_distr[state_idx, action_idx]

results_IS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):
    alpha_1_target_opt_transfer, alpha_2_target_opt_transfer =\
        optimize_parameters(target_size, target_task, pf, source_policy, source_samples, source_sample_probs)
    optimal_pi_transfer = pf.create_policy(alpha_1_target_opt_transfer, alpha_2_target_opt_transfer)
    J1_opt_transfer = calculate_J(target_task, optimal_pi_transfer, gamma)
    results_IS.append([alpha_1_target_opt_transfer, alpha_2_target_opt_transfer, J1_opt_transfer])
    print("IS: TS", target_size, "a1", alpha_1_target_opt_transfer, "a2", alpha_2_target_opt_transfer, "J", J1_opt_transfer)
#np.save('learning_IS_1', np.array(results_IS))