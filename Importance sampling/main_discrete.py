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
from GradientEstimator import GradientEstimator
from ISLearner import ISLearner
from BatchLearner import BatchLearner
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



# No need to calculate reward of source samples on source model, we need reward of source samples in target model only
def estimate_J(dataset, gamma, task=None, weights=None):
    if task is None:
        if isinstance(dataset, dict):
            rewards = dataset['r'].copy()
            if weights is not None:
                rewards *= weights
            J = rewards.mean()/(1-gamma)
            return J
        if isinstance(dataset, tuple):
            rewards = dataset[-1].copy()
            mask = rewards == 1.
            rewards[mask] = 0.
            discounts = np.power(gamma, np.arange(rewards.shape[1]))
            discounted_rewards = rewards * discounts
            discounted_rewards = discounted_rewards[np.logical_not(mask)]
            J = discounted_rewards.mean()/(1-gamma)
            return J
    else:
        state_idx = dataset['fsi']
        action_idx = dataset['ai']
        rewards = task.env.R[state_idx, action_idx]
        J = rewards.mean()/(1-gamma)
        return J



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






# Try: source task 0.2; source policy (0. 1.); different seed
gamma = 0.99
min_pos = -10.
max_pos = 10.
min_act = -1.0
max_act = -min_act
seed = 9876
power_sources = [rescale_state(0.2), rescale_state(0.007), rescale_state(0.0015)]
power_target = rescale_state(0.002)
alpha_1_sources = [0., 0., 0.]
alpha_2_sources = [0., 0., 0.]
alpha_1_target = 1.
alpha_2_target = 1.
action_noise = (max_act - min_act)*0.2
max_episode_length = 200
n_source_samples = [25000]*3
n_target_samples = 5000
n_action_bins = 10 + 1
n_position_bins = 20 + 1
n_velocity_bins = 20 + 1

# Creation of source tasks
source_tasks = [gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                         max_action=max_act, power=power_source, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                         n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins, position_noise=0.025, velocity_noise=0.025)
                for power_source in power_sources]
# Creation of target task
target_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_target, seed=seed, model='S', discrete=True, n_position_bins=n_position_bins,
                       n_velocity_bins=n_velocity_bins, n_action_bins=n_action_bins, position_noise=0.025, velocity_noise=0.025)
# Policy factory
pf = PolicyFactoryMC(model='S', action_noise=action_noise, max_speed=target_task.env.max_speed, min_act=min_act, max_act=max_act,
                     action_bins=target_task.env.action_bins, action_reps=target_task.env.action_reps,
                     state_reps=target_task.env.state_reps, state_to_idx=target_task.env.state_to_idx)
# Defining source policy
source_policies = [pf.create_policy(alpha_1_sources[i], alpha_2_sources[i]) for i in range(len(alpha_1_sources))]
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

lstd_q = LSTD_Q_Estimator(7, 7, 2, 0.45, True, gamma, 0., min_pos, max_pos, target_task.env.min_speed, target_task.env.max_speed,
                          min_act, max_act)
lstd_v = LSTD_V_Estimator(6, 6, 0.2, True, gamma, 0., min_pos, max_pos, target_task.env.min_speed, target_task.env.max_speed)
grad_est = GradientEstimator(baseline_type=1)

'''xs = source_tasks[0].env.state_reps[:,0]
ys = source_tasks[0].env.state_reps[:,1]
target_task.env.set_policy(target_policy, gamma)
source_tasks[0].env.set_policy(source_policies[0], gamma)
for act in range(n_action_bins - 1):
    zs = source_tasks[0].env.dseta_distr[:, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(source_tasks[0].env.state_reps.shape[0] / source_tasks[0].env.velocity_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
    zs = target_task.env.dseta_distr[:, act].flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    aux = int(source_tasks[0].env.state_reps.shape[0] / source_tasks[0].env.velocity_reps.shape[0])
    sur = ax.plot_surface(xs.reshape((-1, aux)), ys.reshape((-1, aux)), zs.reshape((-1, aux)))
plt.show()
plt.close()'''

'''epis = collect_episodes(source_task, 10000, max_episode_length, seed, source_policy, False)
J1 = estimate_J(epis, gamma)
samps = collect_samples(source_task, 1000*50, seed, source_policy)
J2 = estimate_J(samps, gamma)'''

'''for i in range(len(source_tasks)):
    source_task = source_tasks[i]
    source_task.env.set_policy(pf.create_policy(0., 0.), gamma)
    a1 = source_task.env.J
    source_task.env.set_policy(pf.create_policy(1., 0.), gamma)
    a2 = source_task.env.J
    source_task.env.set_policy(pf.create_policy(1., 1.), gamma)
    a3 = source_task.env.J
    source_task.env.set_policy(pf.create_policy(0., 1.), gamma)
    a4 = source_task.env.J
    source_task.env.set_policy(pf.create_policy(.5, .5), gamma)
    a5 = source_task.env.J
    print(power_sources[i], a1, a2, a3, a4, a5)

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
print(b1, b2, b3, b4, b5)'''

'''learner = Learner(gamma, pf, lstd_q, lstd_v, grad_est, seed)
a = learner.collect_samples(source_tasks[0], n_source_samples[0], source_policies[0])
w = learner.calculate_density_ratios_dseta(a, source_tasks[0], target_task, source_policies[0], target_policy)
w2 = learner.calculate_density_ratios_transition_sa(a, source_tasks[0], target_task, source_policies[0], target_policy)
w3 = learner.calculate_density_ratios_delta(a, source_tasks[0], target_task, source_policies[0], target_policy)
w4 = learner.calculate_density_ratios_transition_s(a, source_tasks[0], target_task, source_policies[0], target_policy)
w5 = learner.calculate_density_ratios_r_sa(a, source_tasks[0], target_task, source_policies[0], target_policy)
w6 = learner.calculate_density_ratios_r_s(a, source_tasks[0], target_task, source_policies[0], target_policy)
g = target_policy.log_gradient_matrix.copy()
g = np.transpose(g, axes=(2, 0, 1)) * (target_task.env.Q * target_task.env.dseta_distr)
g = np.transpose(g, axes=(1, 2, 0)).sum(axis=(0, 1))
Qs = lstd_q.fit(a, predict=True, weights_d=w, weights_p=w2, weights_r=w5)
Vs = lstd_v.fit(a, predict=True, weights_d=w3, weights_p=w4, weights_r=w6)
grad = grad_est.estimate_gradient(a, target_policy, weights=w, Q=Qs, V=Vs)
print(grad, g)'''

target_sizes = list(range(200, 1000, 200)) + list(range(1000, 10000, 1000)) + list(range(10000, 50000 + 1, 10000))
n_runs = 1
out_logger = open('results.log', 'w', buffering=1)
for i in [0]:#range(len(source_tasks)):
    print("Task:", power_sources[i], file=out_logger)
    print("IS app", file=out_logger)
    learner = ISLearner(gamma, pf, lstd_q, lstd_v, grad_est, seed)
    learner.learn(target_task, target_sizes, n_runs, [source_tasks[i]], [source_policies[i]], [n_source_samples[i]], out_logger)
    #np.save('learning_IS_app_' + str(i+1), np.array(results_IS_app))

    '''print("IS", file=out_logger)
    learner = Learner(gamma, pf, None, None, grad_est, seed)
    results_IS = learner.learn(target_task, target_sizes, n_runs, [source_tasks[i]], [source_policies[i]], [n_source_samples[i]], out_logger)
    np.save('learning_IS_' + str(i+1), np.array(results_IS))'''

print("All tasks", file=out_logger)
print("IS app", file=out_logger)
learner = ISLearner(gamma, pf, lstd_q, lstd_v, grad_est, seed)
learner.learn(target_task, target_sizes, n_runs, source_tasks, source_policies, n_source_samples, out_logger)
#np.save('learning_IS_app', np.array(results_IS_app))

'''print("No IS app", file=out_logger)
learner = Learner(gamma, pf, lstd_q, lstd_v, grad_est, seed)
results_noIS_app = learner.learn(target_task, target_sizes, n_runs, None, None, None, out_logger)
np.save('learning_noIS_app', np.array(results_noIS_app))'''

'''print("IS", file=out_logger)
learner = Learner(gamma, pf, None, None, grad_est, seed)
results_IS = learner.learn(target_task, target_sizes, n_runs, source_tasks, source_policies, n_source_samples, out_logger)
np.save('learning_IS', np.array(results_IS))

print("No IS", file=out_logger)
learner = Learner(gamma, pf, None, None, grad_est, seed)
results_noIS = learner.learn(target_task, target_sizes, n_runs, None, None, None, out_logger)
np.save('learning_noIS', np.array(results_noIS))'''