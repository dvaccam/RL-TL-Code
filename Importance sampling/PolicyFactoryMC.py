from __future__ import division
import numpy as np
from scipy.stats import norm, beta
from scipy.special import erf


class PolicyMC:
    def __init__(self, alpha1, alpha2, factory):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.factory = factory
        if self.factory.model == 'S':
            self.build_choice_matrix()
            self.build_grad_matrix()
        


    def produce_action(self, state):
        #return self.factory.max_act
        mu = (self.alpha1*self.factory.max_act if state[1] >= 0 else -self.alpha2*self.factory.min_act) * state[1] / self.factory.max_speed
        if self.factory.model == 'G':
            noise = np.random.randn()
            # print(action_noise*noise + alp*state[1]/rescale_state(0.07), norm.pdf(noise)/action_noise,
            #      np.abs(action_noise * noise), alp*state[1]/rescale_state(0.07))
            return self.factory.action_noise * noise + mu
        elif self.factory.model == 'B':
            eps_act = self.factory.rescale_action(1e-2)
            mu_max = (self.factory.min_act + self.factory.max_act +
                      np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                              4 * (self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 - eps_act
            mu_min = (self.factory.min_act + self.factory.max_act -
                      np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                              4 * (self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 + eps_act
            mu = min(max(mu, mu_min), mu_max)
            mu = (mu - self.factory.min_act)/(self.factory.max_act - self.factory.min_act)
            sigma2 = self.factory.action_noise ** 2
            sigma2 = sigma2 / ((self.factory.max_act - self.factory.min_act)**2)
            a = (mu**2-mu**3-mu*sigma2)/sigma2
            b = (1-mu)*a/mu
            noise = np.random.beta(a=a, b=b)
            act = (self.factory.max_act - self.factory.min_act)*noise + self.factory.min_act
            while act <= self.factory.min_act or act >= self.factory.max_act:
                noise = np.random.beta(a=a, b=b)
                act = (self.factory.max_act - self.factory.min_act)*noise + self.factory.min_act
            #print(beta.pdf(noise, a=a, b=b)/(max_act - min_act))
            return act
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
                #return norm.pdf(action[0], loc=mu, scale=self.factory.action_noise)
            elif self.factory.model == 'B':
                eps_act = self.factory.rescale_action(1e-2)
                mu_max = (self.factory.min_act + self.factory.max_act +
                          np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                                  4 * (self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 - eps_act
                mu_min = (self.factory.min_act + self.factory.max_act -
                          np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                                  4 * (self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 + eps_act
                mu = min(max(mu, mu_min), mu_max)
                mu_std = (mu - self.factory.min_act) / (self.factory.max_act - self.factory.min_act)
                sigma2_std = self.factory.action_noise**2 / ((self.factory.max_act - self.factory.min_act) ** 2)
                a = (mu_std ** 2 - mu_std ** 3 - mu_std * sigma2_std) / sigma2_std
                b = (1 - mu_std) * a / mu_std
                return beta.pdf(action[0], a=a, b=b, loc=self.factory.min_act, scale=self.factory.max_act - self.factory.min_act)
        elif action.ndim == 2 and state.ndim == 2:
            a1_mask = state[:,1] >= 0
            a2_mask = np.logical_not(a1_mask)
            mus = self.alpha1 * self.factory.max_act * a1_mask - self.alpha2 * self.factory.min_act * a2_mask
            mus *= state[:,1] / self.factory.max_speed
            if self.factory.model == 'G':
                exs = (action.flatten() - mus) / self.factory.action_noise
                res = np.exp(-exs*exs/2.0)/(np.sqrt(2*np.pi)*self.factory.action_noise)
                return res
            elif self.factory.model == 'B-':
                eps_act = self.factory.rescale_action(1e-2)
                mu_max = (self.factory.min_act + self.factory.max_act +
                          np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                                  4 * (
                                  self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 - eps_act
                mu_min = (self.factory.min_act + self.factory.max_act -
                          np.sqrt((self.factory.min_act + self.factory.max_act) ** 2 -
                                  4 * (
                                  self.factory.min_act * self.factory.max_act + self.factory.action_noise ** 2))) / 2 + eps_act
                mu = min(max(mu, mu_min), mu_max)
                mu_std = (mu - self.factory.min_act) / (self.factory.max_act - self.factory.min_act)
                sigma2_std = self.factory.action_noise ** 2 / ((self.factory.max_act - self.factory.min_act) ** 2)
                a = (mu_std ** 2 - mu_std ** 3 - mu_std * sigma2_std) / sigma2_std
                b = (1 - mu_std) * a / mu_std
                return beta.pdf(action[0], a=a, b=b, loc=self.factory.min_act,
                                scale=self.factory.max_act - self.factory.min_act)



    def log_gradient_paramaters(self, state, action):
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
        for i in range(self.choice_matrix.shape[0]):
            state = self.factory.state_reps[i]
            mu = (self.alpha1 * self.factory.max_act if state[1] >= 0 else -self.alpha2 * self.factory.min_act) * state[1] / self.factory.max_speed
            for j in range(self.choice_matrix.shape[1]):
                if j == 0:
                    self.choice_matrix[i,j] = (1 + erf((self.factory.action_bins[j+1] - mu)/(self.factory.action_noise*np.sqrt(2.))))/2.
                elif j == self.choice_matrix.shape[1] - 1:
                    self.choice_matrix[i, j] = (1 - erf((self.factory.action_bins[j] - mu) / (self.factory.action_noise * np.sqrt(2.)))) / 2.
                else:
                    self.choice_matrix[i, j] = (1 + erf((self.factory.action_bins[j + 1] - mu) / (self.factory.action_noise * np.sqrt(2.)))) / 2. - (1 + erf((self.factory.action_bins[j] - mu)/(self.factory.action_noise*np.sqrt(2.))))/2.



    def build_grad_matrix(self):
        self.gradient_matrix = np.zeros((self.factory.state_reps.shape[0], self.factory.action_reps.shape[0], 2), dtype=np.float64)
        
        for i in range(self.gradient_matrix.shape[0]):
            state = self.factory.state_reps[i]
            mu = (self.alpha1 * self.factory.max_act if state[1] >= 0 else -self.alpha2 * self.factory.min_act) * state[1] / self.factory.max_speed
            vel_norm = state[1]/self.factory.max_speed
            for j in range(self.gradient_matrix.shape[1]):
                if j == 0:
                    action_bin = self.factory.action_bins[j+1]
                    val = (-np.exp(-((mu - action_bin)/self.factory.action_noise)**2)*vel_norm)/(self.factory.action_noise*np.sqrt(2*np.pi)*self.choice_matrix[i,j])
                elif j == self.choice_matrix.shape[1] - 1:
                    action_bin = self.factory.action_bins[j]
                    val = (np.exp(-((mu - action_bin)/self.factory.action_noise)**2)*vel_norm)/(self.factory.action_noise*np.sqrt(2*np.pi)*self.choice_matrix[i,j])
                else:
                    left_action_bin = self.factory.action_bins[j]
                    right_action_bin = self.factory.action_bins[j+1]
                    val = ((-np.exp(-((mu - right_action_bin)/self.factory.action_noise)**2) + np.exp(-((mu - left_action_bin)/self.factory.action_noise)**2))*vel_norm)/(self.factory.action_noise*np.sqrt(2*np.pi)*self.choice_matrix[i,j])
                if vel_norm >= 0:
                    self.gradient_matrix[i,j] = np.array([val, 0])
                else:
                    self.gradient_matrix[i,j] = np.array([0, val])



class PolicyFactoryMC:
    def __init__(self, model, action_noise, max_speed, min_act, max_act, action_bins=None, action_reps=None, state_reps=None,
                 state_to_idx=None):
        self.model = model
        self.action_noise = action_noise
        self.max_speed = max_speed
        self.min_act = min_act
        self.max_act = max_act
        self.action_bins = action_bins
        self.action_reps = action_reps
        self.state_reps = state_reps
        self.state_to_idx = state_to_idx
        
        
        
    def create_policy(self, alpha1, alpha2):
        return PolicyMC(alpha1, alpha2, self)



    def rescale_action(self, x):
        return (self.max_act - self.min_act) * x / 2.0