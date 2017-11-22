import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PolicyFactoryMC import PolicyFactoryMC
from LSTD_Q_Estimator import LSTD_Q_Estimator
from LSTD_V_Estimator import LSTD_V_Estimator
from GradientEstimator import GradientEstimator
from MinMaxWeightsEstimator import MinMaxWeightsEstimator
from MinEstimator import MinWeightsEstimator
from ISLearner import ISLearner
from BatchLearner import BatchLearner
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.dummy import Pool
import copy
import threading as thr
import sys
from itertools import compress
from functools import reduce
import os



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





gamma = 0.99
min_pos = -10.
max_pos = 10.
min_act = -1.0
max_act = -min_act
seed = 9876
power_sources = [rescale_state(0.0025), rescale_state(0.0015),
                 rescale_state(0.007), rescale_state(0.0007),
                 rescale_state(0.07), rescale_state(1e-5)]
#power_sources = [rescale_state(0.007)]
power_target = rescale_state(0.002)
alpha_1_sources = [0.]*len(power_sources)
alpha_2_sources = [1.]*len(power_sources)
alpha_1_sources_opt = [0.63, 0.45, 0.76, 0.22, 0.46, 0.]
#alpha_1_sources_opt = [0.76]
alpha_2_sources_opt = [0.16, 0.1, 0.13, 0.04, 0., 0.]
#alpha_2_sources_opt = [0.13]
alpha_1_target = 0.5
alpha_2_target = 0.1
action_noise = (max_act - min_act)*0.2
max_episode_length = 200
n_source_samples = [25000]*len(power_sources)
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
# Defining source policies
source_policies_rand = [pf.create_policy(alpha_1_sources[i], alpha_2_sources[i]) for i in range(len(alpha_1_sources))]
source_policies_opt = [pf.create_policy(alpha_1_sources_opt[i], alpha_2_sources_opt[i]) for i in range(len(alpha_1_sources_opt))]
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

lstd_q = LSTD_Q_Estimator(3, 3, 3, 0.4, True, gamma, 0., min_pos, max_pos, target_task.env.min_speed, target_task.env.max_speed,
                          min_act, max_act)
lstd_v = LSTD_V_Estimator(3, 3, 0.4, True, gamma, 0., min_pos, max_pos, target_task.env.min_speed, target_task.env.max_speed)
grad_est = GradientEstimator(gamma=gamma, baseline_type=1)
weights_est = MinWeightsEstimator(gamma)

#epis = collect_episodes(target_task, 10, max_episode_length, seed, target_policy, False)

'''xs = np.linspace(0., 1., 11)
ys = np.linspace(0., 1., 11)
Js = np.empty((7,11,11), dtype=np.float64)
for x_idx, x in enumerate(xs):
    for y_idx, y in enumerate(ys):
        print(x, y)
        pol = pf.create_policy(x, y)
        source_tasks[0].env.set_policy(pol, gamma)
        source_tasks[1].env.set_policy(pol, gamma)
        source_tasks[2].env.set_policy(pol, gamma)
        source_tasks[3].env.set_policy(pol, gamma)
        source_tasks[4].env.set_policy(pol, gamma)
        source_tasks[5].env.set_policy(pol, gamma)
        target_task.env.set_policy(pol, gamma)
        Js[0,x_idx, y_idx] = source_tasks[0].env.J
        Js[1,x_idx, y_idx] = source_tasks[1].env.J
        Js[2,x_idx, y_idx] = source_tasks[2].env.J
        Js[3,x_idx, y_idx] = source_tasks[3].env.J
        Js[4,x_idx, y_idx] = source_tasks[4].env.J
        Js[5,x_idx, y_idx] = source_tasks[5].env.J
        Js[6,x_idx, y_idx] = target_task.env.J
np.save('Js', Js)
Js = np.load('Js.npy')
xs, ys = np.meshgrid(xs, ys)
for i in range(len(source_tasks)+1):
    zs = Js[i].T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sur = ax.plot_surface(xs, ys, zs)
plt.show()
plt.close()'''


target_sizes = [0] + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000))
n_runs = 5
alg = str(sys.argv[1]) # Algorithm, folder, init_source, init_target, sources, [start, end]
folder = sys.argv[2]
if not os.path.exists(folder):
    os.makedirs(folder)

if alg == 'NoTransfer':
    del source_tasks, source_policies_rand, source_policies_opt
    name = folder + '/NoTransfer'
    out_stream = open(name + '.log', 'w', buffering=1)
    learner = ISLearner(gamma, pf, lstd_q, lstd_v, grad_est, None, False, False, False, seed, 'x', 'r')
    results = learner.learn(target_task, target_sizes, n_runs, None, None, None, out_stream)
    np.save(name, np.array(results))
elif alg in ['IS', 'Min', 'Batch']:
    init_source = sys.argv[3]  # 'r' for random, 'o' for optimal
    init_target = sys.argv[4]  # 'r' for random, 's' for source
    assert(init_source in ['r', 'o'] and init_target in ['r', 's'])
    sources = list(map(int, list(sys.argv[5])))
    name = init_source + init_target
    for i in range(len(source_tasks)-1, -1, -1):
        if i not in sources:
            del source_tasks[i], source_policies_rand[i], source_policies_opt[i]
    if alg == 'IS':
        name = folder + '/IS_' + name
        learner = ISLearner(gamma, pf, lstd_q, lstd_v, grad_est, None, False, False, False, seed, init_source, init_target)
    elif alg == 'Min':
        name = folder + '/Min_' + name
        if len(sys.argv) > 6:
            start = int(sys.argv[6])
            end = int(sys.argv[7])
            for _ in range(start):
                np.random.seed(seed)
                seed = int(np.random.uniform(high=2**32))
            n_runs = end - start
            name = name + '_(' + str(start) + '-' + str(end) + ')'
        learner = ISLearner(gamma, pf, lstd_q, lstd_v, grad_est, weights_est, True, True, True, seed, init_source, init_target)
    elif alg == 'Batch':
        name = folder + '/Batch_' + name
        learner = BatchLearner(gamma, pf, lstd_q, lstd_v, grad_est, seed, init_source, init_target)

    out_stream = open(name + '.log', 'a', buffering=1)
    for i in range(len(source_tasks)):
        print("Task:", source_tasks[i].env.power, file=out_stream)
        if isinstance(learner, ISLearner):
            results = learner.learn(target_task, target_sizes, n_runs, [source_tasks[i]],
                                    [source_policies_rand[i] if init_source == 'r' else source_policies_opt[i]],
                                    [n_source_samples[i]], out_stream, name + '_' + '{0:.5f}'.format(source_tasks[i].env.power))
        else:
            results = learner.learn(target_task, target_sizes, n_runs, [source_tasks[i]],
                                    [source_policies_rand[i] if init_source == 'r' else source_policies_opt[i]],
                                    n_source_samples[0], out_stream, name + '_' + '{0:.5f}'.format(source_tasks[i].env.power))
        np.save(name + '_' + '{0:.5f}'.format(source_tasks[i].env.power), np.array(results))

    if len(source_tasks) > 1:
        print("All tasks", file=out_stream)
        name = name + '_' + reduce(lambda x, y: x + '_' + y, ['{0:.5f}'.format(st.env.power) for st in source_tasks])
        if isinstance(learner, ISLearner):
            results = learner.learn(target_task, target_sizes, n_runs, source_tasks, source_policies_rand if init_source == 'r' else source_policies_opt,
                                    n_source_samples, out_stream, name)
        else:
            results = learner.learn(target_task, target_sizes, n_runs, source_tasks, source_policies_rand if init_source == 'r' else source_policies_opt,
                                    n_source_samples[0], out_stream, name)
        np.save(name, np.array(results))