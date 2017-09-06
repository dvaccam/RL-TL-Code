import gym
import numpy as np
import utils
import scipy.optimize as opt
import itertools as itt
import time


def policy(state):
    if state[0] < -0.9 or state[1] > 0 or (abs(state[1]) < 0.001 and state[0] < -0.4):
        action = 2
    else:
        action = 0
    return action

gamma = 0.9
goal_pos = -0.35

# Creation of source task
source_task = gym.make('MountainCar-v0', discretize=(1000, 1000), aceleration=0.001, seed=1000, goal_position=goal_pos)
# Data structures for storing samples
n_source = 5#1000
source_samples = [] #93, 115
# Dictionary to check if an initial position is observed during sampling
init_positions = source_task.env.bins[0][(-0.6 <= source_task.env.bins[0]) & (source_task.env.bins[0] <= -0.4)]
seen_init_positions_source = {s0:False for s0 in init_positions}
#Sampling from source task
print("Collecting ", n_source, " samples from source task...")
for i in range(n_source):
    episode = []
    init_state = source_task.reset()
    seen_init_positions_source[init_state[0]] = True
    first_state = init_state
    for t in range(500):
        source_task.render()
        action = policy(first_state)
        next_state, reward, done, info = source_task.step(action)
        episode.append((first_state, action, next_state, reward))
        first_state = next_state
        if done:
            break
    source_samples.append(episode)
print("Done sampling from source task")

# Check there was at least one episode for each initial position
not_seen_states = 0
for k,v in seen_init_positions_source.items():
    if not v:
        not_seen_states += 1#print(k)
print("Not seen positions: ", not_seen_states, "/", len(seen_init_positions_source))



# Creation of target task
target_acel = 0.0025
target_task = gym.make('MountainCar-v0', discretize=(1000, 1000), aceleration=target_acel, seed=1, goal_position=goal_pos)
# Data structures for storing samples
n_target = 1
target_samples = []
# Dictionary to check if an initial position is observed during sampling
seen_init_positions_target = {s0:False for s0 in init_positions}
# Sampling from target task
print("Collecting ", n_target, " samples from target task...")
for i in range(n_target):
    episode = []
    init_state = target_task.reset()
    seen_init_positions_target[init_state[0]] = True
    first_state = init_state
    for t in range(500):
        # target_task.render()
        if first_state[0] < -0.9 or first_state[1] > 0 or (abs(first_state[1]) < 0.001 and first_state[0] < -0.5):
            action = 2
        else:
            action = 0
        next_state, reward, done, info = target_task.step(action)
        episode.append((first_state, action, next_state, reward))
        first_state = next_state
        if done:
            break
    target_samples.append(episode)
print("Done sampling from target task")
# Check there was at least one episode for each initial state
not_seen_states = 0
for k,v in seen_init_positions_target.items():
    if not v:
        not_seen_states += 1#print(k)
print("Not seen states: ", not_seen_states, "/", len(seen_init_positions_target))



# Calculate real value function on target task
V = {}
print("Calculating real value function for target task...")
for s in init_positions:
    V[s] = 0
    target_task = gym.make('MountainCar-v0', discretize=(1000, 1000), aceleration=target_acel, initial_position=s, goal_position=goal_pos)
    init_state = target_task.reset()
    first_state = init_state
    for t in range(500):
        # target_task.render()
        if first_state[0] < -0.9 or first_state[1] > 0 or (abs(first_state[1]) < 0.001 and first_state[0] < -0.5):
            action = 2
        else:
            action = 0
        next_state, reward, done, info = target_task.step(action)
        V[s] += np.power(gamma, t)*reward
        first_state = next_state
        if done:
            break
print("Done calculating real value function for target task")
# Calculate real expected reward for policy
Jt = np.array(list(V.values())).mean()
print("Expected reward for the policy:", Jt)
print("Estimated expected reward:", utils.estimate_J(target_samples, gamma))



def B(D_source, D_target, gamma, ext, target_task, Jt):# Ext is a mutidimensional array with the new states
    D_c = list(D_target)
    diff = 0
    for ep in range(len(ext)):
        ep_c = []
        for d in range(len(ext[ep])):
            init_pos, init_vel = D_source[ep][d][0]
            action = D_source[ep][d][1]
            vel = init_vel + (action - 1) * target_acel + np.cos(3 * init_pos) * (-0.0025)
            vel = target_task.env.bins[1][np.array(abs(target_task.env.bins[1] - vel)).argmin()]
            pos = init_pos + vel
            pos = target_task.env.bins[0][np.array(abs(target_task.env.bins[0] - pos)).argmin()]
            if (pos == target_task.env.min_position and vel < 0): vel = 0
            dis = np.sqrt(((np.array([pos, vel]) - ext[ep][d])**2).sum())
            ep_c.append((D_source[ep][d][0], D_source[ep][d][1], ext[ep][d], pow(2, dis)*-1))
            diff += (D_source[ep][d][3] - ep_c[d][3])
        D_c.append(ep_c)
    diff /= len(D_source)
    J_cap = utils.estimate_J(D_c, gamma)
    eps = abs(J_cap - Jt)
    return eps + diff

def par_f(D_source, D_target, gamma, x, target_task, Jt):
    ext = []
    offset = 0
    for e in range(len(D_source)):
        ep = []
        for d in range(len(D_source[e])):
            ep.append(np.array([x[offset + 2*d], x[offset + 2*d + 1]]))
        offset += 2*len(D_source[e])
        ext.append(ep)
    return B(D_source, D_target, gamma, ext, target_task, Jt)

def B_1(D_source, D_target, gamma, ext, sample_mask, target_task, Jt):# Ext is a list of lists as D_source, but with the state as a numpy array instead of a tuple
    #rounded_mask = np.array(np.around(sample_mask)).astype(dtype=bool)
    D_c = list(D_target)
    diff = 0
    for ep in range(len(ext)):
        #if rounded_mask[ep]:
        ep_c = []
        for d in range(len(ext[ep])):
            init_pos, init_vel = D_source[ep][d][0]
            action = D_source[ep][d][1]
            vel = init_vel + (action - 1) * target_acel + np.cos(3 * init_pos) * (-0.0025)
            vel = target_task.env.bins[1][np.array(abs(target_task.env.bins[1] - vel)).argmin()]
            pos = init_pos + vel
            pos = target_task.env.bins[0][np.array(abs(target_task.env.bins[0] - pos)).argmin()]
            if (pos == target_task.env.min_position and vel < 0): vel = 0
            dis = np.sqrt(((np.array([pos, vel]) - ext[ep][d])**2).sum())
            ep_c.append((D_source[ep][d][0], D_source[ep][d][1], ext[ep][d], pow(2, dis)*-1))
            diff += sample_mask[ep]*(D_source[ep][d][3] - ep_c[d][3])
        D_c.append(ep_c)
    diff /= sample_mask.sum() if sample_mask.sum() != 0 else 1
    J_cap = utils.estimate_J(D_c, gamma, weights=np.hstack((np.ones(len(D_target)), sample_mask)))
    eps = abs(J_cap - Jt)
    return eps + diff

def par_f1(D_source, D_target, gamma, x, target_task, Jt):
    ext = []
    offset = 0
    for e in range(len(D_source)):
        ep = []
        for d in range(len(D_source[e])):
            ep.append(np.array([x[offset + 2 * d], x[offset + 2 * d + 1]]))
        offset += 2 * len(D_source[e])
        ext.append(ep)
    sample_mask = x[-len(D_source):]
    return B_1(D_source, D_target, gamma, ext, sample_mask, target_task, Jt)


np.random.seed(200)



st = time.time()
ex = []
for e in source_samples:
    for d in e:
        ex.extend([d[2][0], d[2][1]])
ex = np.array(ex)
dim = ex.shape[0]
a = par_f(source_samples, target_samples, gamma, ex, target_task, Jt)
print("Bound by using source samples as extension:", a)
print("Estimated expected reward with direct tranfer:", utils.estimate_J(target_samples+source_samples, gamma))
print("Error by direct transfer:", abs(utils.estimate_J(target_samples+source_samples, gamma) - Jt))

f = lambda x:par_f(D_source=source_samples, D_target=target_samples, gamma=gamma, Jt=Jt, target_task=target_task, x=x)
bounds = []
for _ in range(int(dim/2)):
    bounds.extend([(target_task.env.min_position, target_task.env.max_position), (-target_task.env.max_speed, target_task.env.max_speed)])

s0 = np.random.uniform(low=target_task.env.min_position, high=target_task.env.max_position, size=int(dim/2))
v0 = np.random.uniform(low=-target_task.env.max_speed, high=target_task.env.max_speed, size=int(dim/2))
x0 = np.empty(dim, dtype=s0.dtype)
x0[0::2] = s0
x0[1::2] = v0
best = opt.minimize(f, x0, bounds=bounds, options={'maxfun':min(10000*len(source_samples), 1e5)})
print("Def:", best.fun, best.x, best.success)
x_opt_all = best.x
print("Full extenstion:", time.time()-st)
print("-------")



st = time.time()
minim = 100
min_idx = None
for n in range(len(source_samples)):
    for idx in itt.combinations(range(len(source_samples)), n+1):
        sub = np.array(source_samples)[np.array(idx)].tolist()
        ex = []
        for e in sub:
            for d in e:
                ex.extend([d[2][0], d[2][1]])
        ex = np.array(ex)
        dim = ex.shape[0]
        a = par_f(sub, target_samples, gamma, ex, target_task, Jt)
        print(idx)
        print("Bound by using source samples as extension:", a)
        print("Estimated expected reward with direct tranfer:", utils.estimate_J(target_samples+sub, gamma))
        print("Error by direct transfer:", abs(utils.estimate_J(target_samples+sub, gamma) - Jt))

        f = lambda x:par_f(D_source=sub, D_target=target_samples, gamma=gamma, Jt=Jt, target_task=target_task, x=x)
        bounds = []
        for _ in range(int(dim/2)):
            bounds.extend([(target_task.env.min_position, target_task.env.max_position), (-target_task.env.max_speed, target_task.env.max_speed)])

        s0 = np.random.uniform(low=target_task.env.min_position, high=target_task.env.max_position, size=int(dim/2))
        v0 = np.random.uniform(low=-target_task.env.max_speed, high=target_task.env.max_speed, size=int(dim/2))
        x0 = np.empty(dim, dtype=s0.dtype)
        x0[0::2] = s0
        x0[1::2] = v0
        best = opt.minimize(f, x0, bounds=bounds, options={'maxfun':min(10000*len(sub), 1e5)})
        if best.fun < minim:
            minim = best.fun
            min_idx = idx
        print("Def:", best.fun, best.success)
        print("-------")
print("Best one:",minim, min_idx)
print("Combinatorial:", time.time()-st)
print("-------")

st = time.time()
'''ex = []
for e in source_samples:
    for d in e:
        ex.extend([d[2][0], d[2][1]])
ex.extend([0,1])
ex = np.array(ex)
dim = ex.shape[0]
a = par_f1(source_samples, target_samples, gamma, ex, target_task, Jt)
print("Bound by using some source samples as extension:", a)
print("Error by direct transfer:", abs(utils.estimate_J(target_samples+source_samples, gamma, np.array([1,0,1])) - Jt))'''
dim = ex.shape[0] + len(source_samples)


f = lambda x:par_f1(D_source=source_samples, D_target=target_samples, gamma=gamma, Jt=Jt, target_task=target_task, x=x)
bounds = []
for _ in range(int((dim-len(source_samples))/2)):
    bounds.extend([(target_task.env.min_position, target_task.env.max_position), (-target_task.env.max_speed, target_task.env.max_speed)])
bounds.extend([(0,1) for _ in range(len(source_samples))])

reps = 1
for i in range(reps):
    s0 = np.random.uniform(low=target_task.env.min_position, high=target_task.env.max_position, size=int((dim-len(source_samples))/2))
    v0 = np.random.uniform(low=-target_task.env.max_speed, high=target_task.env.max_speed, size=int((dim-len(source_samples))/2))
    x0 = np.empty(dim-len(source_samples), dtype=s0.dtype)
    x0[0::2] = s0
    x0[1::2] = v0
    w0 = np.random.uniform(low=0, high=1, size=len(source_samples))
    #x0 = x_opt_all
    w0 = np.ones(len(source_samples))
    best = opt.minimize(f, np.hstack((x0, w0)), bounds=bounds, options={'maxfun':min(10000*len(source_samples), 1e5)})
    print("Def:", best.fun, best.x, best.success)
print("Full optimization:", time.time()-st)