import numpy as np
import scipy.sparse as sps

#S and A are array-like; R and pi are |S|*|A| matrices, P is |S|*|A|*|S|
def value_iteration(S, A, P, R, gamma, pi, max_iter=1000, tol=0.001):
    V = np.zeros_like(S)
    delta_V = tol + 1
    it = 1
    R_pi = np.array((R*pi).sum(axis=1)).flatten()
    P_pi = np.array([np.array(pi[s1,:].dot(P[s1,:,:])).flatten() for s1 in S])
    while it <= max_iter and delta_V > tol:
        V1 = R_pi + gamma*P_pi.dot(V)
        delta_V = np.abs(V-V1).max()
        V = V1
        print(V1)


# D is a episodic dataset of samples
def estimate_J(D, gamma, weights=None):
    if weights is None:
        weights = np.ones(len(D))
    init_states_count = {}
    init_states_reward = {}
    for ep in range(len(D)):
        t = 0
        try:
            init_states_count[D[ep][0][0][0]] += 1
        except KeyError:
            init_states_count[D[ep][0][0][0]] = 1
            init_states_reward[D[ep][0][0][0]] = 0
        for obs in D[ep]:
            init_states_reward[D[ep][0][0][0]] += weights[ep]*np.power(gamma, t)*obs[3]
            t += 1

    init_states_mean_reward = {}
    for k,v in init_states_count.items():
        init_states_mean_reward[k] = init_states_reward[k]/ init_states_count[k]
    #return np.average(list(init_states_mean_reward.values()), weights=list(init_states_count.values()))
    return np.sum(list(init_states_reward.values()))/weights.sum()