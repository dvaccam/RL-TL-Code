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



def collect_samples(task, n_episodes, max_episode_length, seed, policy, render):
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
def calculate_density_ratios(dataset, source_task, target_task, source_policy, target_policy, source_episode_probs=None, target_episode_model_probs=None):
    mask = dataset[3] != 1
    source_probs_cond = lambda samp: source_task.env.transition_model_pdf(samp[3:5], samp[:2], samp[2:3])*\
                                     source_policy.action_pdf(samp[2:3], samp[:2]) if samp[5] != 1 else 1
    target_probs_cond = lambda samp: target_task.env.transition_model_pdf(samp[3:5], samp[:2], samp[2:3])*\
                                     target_policy.action_pdf(samp[2:3], samp[:2]) if samp[5] != 1 else 1
    #st = time.time()
    #samples = np.dstack(dataset)
    if source_episode_probs is None:
        source_sample_probs = np.ones((dataset[0].shape[0], dataset[0].shape[1]), dtype=np.float64)
        all_source_probs = source_task.env.transition_model_pdf(dataset[2][mask], dataset[0][mask],
                                                                dataset[1][mask].reshape(-1, 1))
        all_source_probs *= source_policy.action_pdf(dataset[1][mask].reshape(-1, 1), dataset[0][mask])
        source_sample_probs[mask] = all_source_probs
        #source_sample_probs = np.apply_along_axis(source_probs_cond, 2, samples)
        source_episode_probs = source_sample_probs.prod(axis=1)*(1/(source_task.env.max_initial_state - source_task.env.min_initial_state))
    if target_episode_model_probs is None:
        target_sample_probs = np.ones((dataset[0].shape[0], dataset[0].shape[1]), dtype=np.float64)
        all_target_probs = target_task.env.transition_model_pdf(dataset[2][mask], dataset[0][mask],
                                                                dataset[1][mask].reshape(-1, 1))
        all_target_probs *= target_policy.action_pdf(dataset[1][mask].reshape(-1, 1), dataset[0][mask])
        target_sample_probs[mask] = all_target_probs
        #target_sample_probs = np.apply_along_axis(target_probs_cond, 2, samples)
        target_episode_probs = target_sample_probs.prod(axis=1) * (1 / (target_task.env.max_initial_state - target_task.env.min_initial_state))
    else:
        target_probs_pol_cond =  lambda samp: target_policy.action_pdf(samp[2:3], samp[:2]) if samp[5] != 1 else 1
        target_sample_pol_probs = np.ones((dataset[0].shape[0], dataset[0].shape[1]), dtype=np.float64)
        all_target_probs = target_policy.action_pdf(dataset[1][mask].reshape(-1, 1), dataset[0][mask])
        target_sample_pol_probs[mask] = all_target_probs
        #target_sample_pol_probs = np.apply_along_axis(target_probs_pol_cond, 2, samples)
        target_episode_probs = target_sample_pol_probs.prod(axis=1)*target_episode_model_probs * (1 / (target_task.env.max_initial_state - target_task.env.min_initial_state))
    #et = time.time()
    #print(et-st)
    return target_episode_probs/source_episode_probs



# No need to calculate reward of source samples on source model, we need reward of source samples in target model only
def estimate_J(dataset, gamma, weights=None):
    rewards = dataset[-1].copy()
    rewards[rewards == 1] = 0
    discounts = np.power(gamma, np.arange(rewards.shape[1]))
    cumm_discounted_rewards = rewards.dot(discounts)
    if weights is not None:
        cumm_discounted_rewards *= weights
    J = cumm_discounted_rewards.mean()
    variance = cumm_discounted_rewards.var()
    return J, variance



def optimize_parameters(target_size, target_task, pf, source_policy=None, source_samples=None, source_episode_probs=None, target_episode_model_probs=None):
    step_size = 0.01
    max_iters = 2000
    i = 1
    np.random.seed(seed)
    grad = np.zeros(2, dtype=np.float64)
    grad_norm = 1
    alpha1 = source_policy.alpha1 if source_policy is not None else np.random.uniform()
    alpha2 = source_policy.alpha2 if source_policy is not None else np.random.uniform()

    while grad_norm > 1e-3 and i <= max_iters: #TODO: add baseline_type in estimate_gradient
        alpha1 += step_size * grad[0]
        alpha2 += step_size * grad[1]
        alpha1 = max(min(alpha1, 1.0), 0.0)
        alpha2 = max(min(alpha2, 1.0), 0.0)
        pol = pf.create_policy(alpha1, alpha2)
        target_samples = collect_samples(target_task, target_size, max_episode_length, seed, pol, False)
        if source_samples is not None:
            weights = calculate_density_ratios(source_samples, source_task, target_task, source_policy, pol,
                                               source_episode_probs, target_episode_model_probs)
            weights = np.append(np.ones(target_size, dtype=np.float64), weights)
            transfer_samples = (np.vstack((target_samples[0], source_samples[0])),
                                np.vstack((target_samples[1], source_samples[1])),
                                np.vstack((target_samples[2], source_samples[2])),
                                np.vstack((target_samples[3], source_samples[3])))
            grad = estimate_gradient(transfer_samples, pol, weights, baseline_type=1)
        else:
            grad = estimate_gradient(target_samples, pol, baseline_type=1)
        '''if i % 100 == 11 or i == 1:
            ts = collect_samples(target_task, 10000, max_episode_length, seed, pol, False)
            g = estimate_gradient(ts, pol, baseline_type=1)
            print(grad, g)
            print(alpha1, alpha2)'''
        grad *= np.array([(0 < alpha1 < 1) or ((alpha1 < 1 or grad[0] < 0) and (alpha1 > 0 or grad[0] > 0)),
                          (0 < alpha2 < 1) or ((alpha2 < 1 or grad[1] < 0) and (alpha2 > 0 or grad[1] > 0))])
        grad_norm = np.sqrt((grad ** 2).sum())
        grad /= grad_norm if grad_norm != 0 else 1
        i += 1
        step_size -= (0.01-0.001)/max_iters
        #target_samples = collect_samples(target_task, 10000, max_episode_length, seed, pol, False)
        #J, _ = estimate_J(target_samples, gamma)
        #print(alpha1, alpha2, grad_norm, i)
    if i > max_iters:
        print("Did not converge")
        print(grad_norm, i)
    return alpha1, alpha2



# No need to calculate gradient of source samples on source model, we need gradient of source samples in target model only
def estimate_gradient(dataset, policy, weights=None, baseline_type=0):
    rewards = dataset[-1].copy()
    rewards[rewards == 1] = 0
    discounts = np.power(gamma, np.arange(rewards.shape[1]))
    cumm_discounted_rewards = rewards.dot(discounts)
    #st = time.time()
    log_grads_sample = np.zeros((rewards.shape[0], rewards.shape[1], 2), dtype=np.float64)
    mask = dataset[-1] != 1
    all_grads = policy.log_gradient_paramaters(dataset[0][mask], dataset[1][mask].reshape((-1,1)))
    log_grads_sample[mask] = all_grads
    #et = time.time()
    #print(et-st)
    #st = time.time()
    '''samples = np.dstack(dataset)
    log_grads_sample = np.apply_along_axis(lambda s: policy.log_gradient_paramaters(s[:2], s[2:3]), 2, samples)'''
    log_grads_episode = log_grads_sample.sum(axis=1)
    #et = time.time()
    #print(et-st)
    if log_grads_episode.shape[0] > 1:
        if baseline_type == 0:
            grad = (log_grads_episode.T * cumm_discounted_rewards).T
            if weights is not None:
                grad = (grad.T*weights).T
            grad = grad.mean(axis=0)
        if baseline_type == 1:
            baseline = ((log_grads_episode ** 2).T*cumm_discounted_rewards).T
            den = (log_grads_episode ** 2)
            if weights is not None:
                baseline = (baseline.T*weights).T
                den = (den.T*weights).T
            baseline = baseline.mean(axis=0)
            den = den.mean(axis=0)
            den[den == 0] = 1
            baseline /= den
            grad = (log_grads_episode.T*cumm_discounted_rewards).T - log_grads_episode*baseline
            if weights is not None:
                grad = (grad.T*weights).T
            grad = grad.mean(axis=0)

        if baseline_type == 2:
            cum_grads = np.cumsum(log_grads_sample, axis=1)
            den = cum_grads**2
            baseline = np.transpose(np.transpose(cum_grads, axes=(2, 0, 1)) * (rewards*discounts), axes=(1, 2, 0))
            if weights is not None:
                baseline = np.transpose(np.transpose(baseline, axes=(2, 1, 0)) * weights, axes=(2, 1, 0))
                den = np.transpose(np.transpose(den, axes=(2,1,0))*weights, axes=(2,1,0))
            masked_baseline = np.ma.array(baseline, mask=np.logical_not(np.dstack((mask, mask))))
            baseline = np.asarray(np.ma.average(masked_baseline, axis=0))
            masked_den = np.ma.array(den, mask=np.logical_not(np.dstack((mask, mask))))
            den = np.asarray(np.ma.average(masked_den, axis=0))
            den[den == 0] = 1
            baseline /= den
            grad = np.transpose(np.transpose(cum_grads,axes=(2, 0, 1))*(rewards*discounts), axes=(1,2,0)) + cum_grads*baseline
            grad = grad.sum(axis=1)
            if weights is not None:
                grad = (grad.T * weights).T
            grad = grad.mean(axis=0)
    else:
        grad = (log_grads_episode * cumm_discounted_rewards).ravel()
    return grad




gamma = 0.95
min_pos = -40.
max_pos = 40.
min_act = -1.0
max_act = -min_act
seed = 100#int(time.time())
power_source = rescale_state(0.005)
power_target = rescale_state(0.0055)
alpha_1_source = 0.
alpha_2_source = 1.
alpha_1_target = 1.
alpha_2_target = 0.
n_source_episodes = 20000
n_target_episodes = 50000
action_noise = (max_act - min_act)*0.2#rescale_action(2*0.05/5)
max_episode_length = 200


# Creation of source task
source_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_source, seed=seed, model='G')
# Creation of target task
target_task = gym.make('MountainCarContinuous-v0', min_position=min_pos, max_position=max_pos, min_action=min_act,
                       max_action=max_act, power=power_target, seed=seed, model='G')
# Policy factory
pf = PolicyFactoryMC(model='G', action_noise=action_noise, max_speed=source_task.env.max_speed, min_act=min_act, max_act=max_act)
# Defining source policy
source_policy = pf.create_policy(alpha_1_source, alpha_2_source)
# Defining target policy
target_policy = pf.create_policy(alpha_1_target, alpha_2_target)

'''xs = np.linspace(source_task.env.min_speed, source_task.env.max_speed, 1000)
ys1 = [source_task.env.transition_model_pdf(np.array([0,x]), np.array([0,0]), np.array([max_act]), True)[1] for x in xs]
ys2 = [target_task.env.transition_model_pdf(np.array([0,x]), np.array([0,0]), np.array([max_act]), True)[1] for x in xs]
plt.plot(xs, ys1, 'b', xs, ys2, 'r')
plt.show()'''


'''# Collecting source episodes
print("Collecting", n_source_episodes, "episodes from source task...")
source_samples = collect_samples(source_task, n_source_episodes, max_episode_length, seed, source_policy, False)
source_task.close()
print("Done sampling from source task")

# Collecting target episodes
print("Collecting", n_target_episodes, "episodes from target task...")
target_samples = collect_samples(target_task, n_target_episodes, max_episode_length, seed, target_policy, False)
target_task.close()
print("Done sampling from target task")



J_1, var_J_1 = estimate_J(source_samples, gamma)
print("Estimated performance on source task:", J_1,"; empirical variance of estimate:",var_J_1)

J_2, var_J_2 = estimate_J(target_samples, gamma)
print("Estimated performance on target task:", J_2,"; empirical variance of estimate:",var_J_2)

reduced_target_samples = (target_samples[0][:10], target_samples[1][:10], target_samples[2][:10], target_samples[3][:10])
J_1_reduced, var_J_1_reduced = estimate_J(reduced_target_samples, gamma)
print("Estimated performance on target task (reduced dataset):", J_1_reduced,"; empirical variance of estimate:",var_J_1_reduced)

transfer_samples = (np.vstack((reduced_target_samples[0],source_samples[0])), np.vstack((reduced_target_samples[1],source_samples[1])), np.vstack((reduced_target_samples[2],source_samples[2])), np.vstack((reduced_target_samples[3],source_samples[3])))
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
#a = collect_samples(source_task, 10000, 200, seed, source_policy, False)
#b = collect_samples(target_task, 10000, 200, seed, target_policy, False)
#w = calculate_density_ratios(a, source_task, target_task, source_policy, target_policy)
'''s = collect_samples(source_task, 10000, 200, seed, source_policy, False)
a = estimate_J(s, gamma)
s = collect_samples(source_task, 10000, 200, seed, pf.create_policy(0.,1.), False)
b = estimate_J(s, gamma)
s = collect_samples(source_task, 10000, 200, seed, pf.create_policy(1.,0.), False)
c = estimate_J(s, gamma)
s = collect_samples(source_task, 10000, 200, seed, pf.create_policy(0.,0.), False)
d = estimate_J(s, gamma)'''

#sys.stdout = open('out.log', 'w')

'''results_noIS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):#[1] + list(range(100, n_target_episodes+1, 100 )):
    alpha_1_target_opt, alpha_2_target_opt = optimize_parameters(target_size, target_task, pf)
    optimal_pi = pf.create_policy(alpha_1_target_opt, alpha_2_target_opt)
    target_samples = collect_samples(target_task, n_target_episodes, max_episode_length, seed, optimal_pi, False)
    J1_opt, _ = estimate_J(target_samples, gamma)
    results_noIS.append([alpha_1_target_opt, alpha_1_target_opt, J1_opt])
    print("No IS: TS", target_size, "a1", alpha_1_target_opt, "a2", alpha_2_target_opt, "J", J1_opt)
    #sys.stdout.flush()
np.save('learning_noIS_1', np.array(results_noIS))'''

ts = collect_samples(target_task, 10000, max_episode_length, seed, target_policy, False)
estimate_gradient(ts, target_policy, baseline_type=1)
source_samples = collect_samples(source_task, int(5e4), max_episode_length, seed, source_policy, False)
w = calculate_density_ratios(source_samples, source_task, target_task, source_policy, target_policy)
for i in [1e4, 2e4, 3e4, 4e4, 5e4]:
    g = estimate_gradient(source_samples[:int(i)], target_policy, weights=w[:int(i)], baseline_type=1)
    target_samples = collect_samples(target_task, 10000, max_episode_length, seed, target_policy, False)
    g1 = estimate_gradient(target_samples, target_policy, baseline_type=1)
    print(g, g1)



mask = source_samples[3] != 1
source_sample_probs = np.ones((source_samples[0].shape[0], source_samples[0].shape[1]), dtype=np.float64)
all_source_probs = source_task.env.transition_model_pdf(source_samples[2][mask], source_samples[0][mask],
                                                        source_samples[1][mask].reshape(-1, 1))
all_source_probs *= source_policy.action_pdf(source_samples[1][mask].reshape(-1, 1), source_samples[0][mask])
source_sample_probs[mask] = all_source_probs
source_episode_probs = source_sample_probs.prod(axis=1)*(1/(source_task.env.max_initial_state - source_task.env.min_initial_state))
target_sample_model_probs = np.ones((source_samples[0].shape[0], source_samples[0].shape[1]), dtype=np.float64)
all_target_probs = target_task.env.transition_model_pdf(source_samples[2][mask], source_samples[0][mask],
                                                        source_samples[1][mask].reshape(-1, 1))
target_sample_model_probs[mask] = all_target_probs
target_episode_model_probs = target_sample_model_probs.prod(axis=1)

results_IS = []
for target_size in list(range(10, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 10000+1, 1000)):#[1] + list(range(10, n_target_episodes+1, 10)):
    alpha_1_target_opt_transfer, alpha_2_target_opt_transfer =\
        optimize_parameters(target_size, target_task, pf, source_policy, source_samples, source_episode_probs, target_episode_model_probs)
    optimal_pi_transfer = pf.create_policy(alpha_1_target_opt_transfer, alpha_2_target_opt_transfer)
    target_samples = collect_samples(target_task, n_target_episodes, max_episode_length, seed, optimal_pi_transfer, False)
    J1_opt_transfer, _ = estimate_J(target_samples, gamma)
    results_IS.append([alpha_1_target_opt_transfer, alpha_2_target_opt_transfer, J1_opt_transfer])
    print("IS: TS", target_size, "a1", alpha_1_target_opt_transfer, "a2", alpha_2_target_opt_transfer, "J", J1_opt_transfer)
np.save('learning_IS_1', np.array(results_IS))