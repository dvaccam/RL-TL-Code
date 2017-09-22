from __future__ import division
import numpy as np
from scipy.special import erf
import time


class PolicyMC:
    def __init__(self, alpha1, alpha2, factory):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.factory = factory
        if self.factory.model == 'S':
            self.build_choice_matrix()
            self.build_log_grad_matrix()



    def produce_action(self, state):
        alp = self.alpha1*self.factory.max_act if state[1] >= 0 else -self.alpha2*self.factory.min_act
        mu =  alp * (state[1] / self.factory.max_speed)
        if self.factory.model == 'G':
            #return self.factory.max_act
            noise = np.random.randn()
            return self.factory.action_noise * noise + mu
        elif self.factory.model == 'S':
            #return self.factory.action_reps[-1]
            state_idx = self.factory.state_to_idx[state[0]][state[1]]
            return np.random.choice(self.factory.action_reps, p=self.choice_matrix[state_idx])



    def action_pdf(self, action, state):
        if action.size == 1 and state.size == 2:
            mu = (self.alpha1 * self.factory.max_act if state[1] >= 0 else -self.alpha2 * self.factory.min_act) * state[1] / self.factory.max_speed
            if self.factory.model == 'G':
                ex = (action[0] - mu) / self.factory.action_noise
                res = np.exp(-ex*ex/2.0)/(np.sqrt(2*np.pi)*self.factory.action_noise)
                return res
        elif action.ndim == 2 and state.ndim == 2:
            a1_mask = state[:,1] >= 0
            a2_mask = np.logical_not(a1_mask)
            mus = self.alpha1 * self.factory.max_act * a1_mask - self.alpha2 * self.factory.min_act * a2_mask
            mus *= state[:,1] / self.factory.max_speed
            if self.factory.model == 'G':
                exs = (action.flatten() - mus) / self.factory.action_noise
                res = np.exp(-exs*exs/2.0)/(np.sqrt(2*np.pi)*self.factory.action_noise)
                return res



    def log_gradient_paramaters(self, state, action): #TODO: Update with new policy
        if state.size == 2 and action.size == 1:
            vel_norm = state[1] / self.factory.max_speed
            if state[1] >= 0:
                return np.array([vel_norm*(action[0] - self.alpha1*vel_norm)/(self.factory.action_noise**2), 0])
            else:
                return np.array([0, vel_norm * (action[0] - self.alpha2 * vel_norm) / (self.factory.action_noise ** 2)])
        elif state.ndim == 2 and action.ndim == 2:
            a1_mask = state[:, 1] >= 0
            a2_mask = np.logical_not(a1_mask)
            log_grads = np.vstack((a1_mask*state[:,1]*(action.flatten() - self.alpha1 * state[:,1]/self.factory.max_speed),
                                   a2_mask*state[:,1]*(action.flatten() - self.alpha2 * state[:,1]/self.factory.max_speed)))
            log_grads = log_grads.T / (self.factory.max_speed*self.factory.action_noise**2)
            return log_grads



    def build_choice_matrix(self):
        self.choice_matrix = np.zeros((self.factory.state_reps.shape[0], self.factory.action_reps.shape[0]), dtype=np.float64)
        a1_mask = self.factory.state_reps[:,1] >= 0
        a2_mask = np.logical_not(a1_mask)
        mu = (a1_mask * self.alpha1 + a2_mask * self.alpha2)*(self.factory.state_reps[:,1] / self.factory.max_speed)
        self.choice_matrix[:,0] = (1 + erf((self.factory.action_bins[1] - mu)/(self.factory.action_noise*np.sqrt(2.))))/2.
        self.choice_matrix[:,-1] = (1 - erf((self.factory.action_bins[-2] - mu) / (self.factory.action_noise * np.sqrt(2.))))/2.
        mu_rep = np.dstack([mu]*(self.factory.action_reps.shape[0] - 2))
        self.choice_matrix[:,1:-1] = (erf((self.factory.action_bins[2:-1] - mu_rep) / (self.factory.action_noise * np.sqrt(2.))) - erf((self.factory.action_bins[1:-2] - mu_rep) / (self.factory.action_noise * np.sqrt(2.))))/2.
                    


    def build_log_grad_matrix(self):
        self.log_gradient_matrix = np.zeros((self.factory.state_reps.shape[0], self.factory.action_reps.shape[0], 2), dtype=np.float64)
        all_grads = np.zeros((self.factory.state_reps.shape[0], self.factory.action_reps.shape[0]), dtype=np.float64)
        a1_mask = self.factory.state_reps[:, 1] >= 0
        a2_mask = np.logical_not(a1_mask)
        vel_norm = self.factory.state_reps[:, 1] / self.factory.max_speed
        mu = (a1_mask * self.alpha1 + a2_mask * self.alpha2) * vel_norm
        all_grads[:,0] = (-np.exp(-((mu - self.factory.action_bins[1])/self.factory.action_noise)**2)*vel_norm) / (self.factory.action_noise*np.sqrt(2*np.pi)*self.choice_matrix[:,0])
        all_grads[:,-1] = (np.exp(-((mu - self.factory.action_bins[-2])/self.factory.action_noise) ** 2) * vel_norm) / (self.factory.action_noise * np.sqrt(2 * np.pi) * self.choice_matrix[:,-1])
        mu_rep = np.hstack([mu.reshape((-1,1))] * (self.factory.action_reps.shape[0] - 2))
        all_grads[:,1:-1] = ((-np.exp(-((mu_rep - self.factory.action_bins[2:-1])/self.factory.action_noise)**2) + np.exp(-((mu_rep - self.factory.action_bins[1:-2])/self.factory.action_noise)**2)).T*vel_norm).T/(self.factory.action_noise*np.sqrt(2*np.pi)*self.choice_matrix[:,1:-1])
        self.log_gradient_matrix[a1_mask,:,0] = all_grads[a1_mask]
        self.log_gradient_matrix[a2_mask,:,1] = all_grads[a2_mask]



class PolicyFactoryMC:
    def __init__(self, model, action_noise, max_speed, min_act, max_act, bottom_pos, min_pos, max_pos, action_bins=None, action_reps=None, state_reps=None,
                 state_to_idx=None):
        self.model = model
        self.action_noise = action_noise
        self.max_speed = max_speed
        self.min_act = min_act
        self.max_act = max_act
        self.bottom_pos = bottom_pos
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.action_bins = action_bins
        self.action_reps = action_reps
        self.state_reps = state_reps
        self.state_to_idx = state_to_idx
        
        
        
    def create_policy(self, alpha1, alpha2):
        return PolicyMC(alpha1, alpha2, self)



    def rescale_action(self, x):
        return (self.max_act - self.min_act) * x / 2.0