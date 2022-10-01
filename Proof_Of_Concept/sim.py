import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from qutip import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DIM_RES = 10
DIM_Q = 2
A = destroy(DIM_RES)
WR = 7.062 #GHz frequency of resonator
WQ = 5.092 #GHz frequency of qubit
G = 102.9 #MHz coupling strength
K = 5.35 #MHz kappa, photon decay rate
X = 0.856 #MHz kai, dispersive shift
P1PH = 36.1 #MHz power required to maintain a population of one photon
PNORM = 4.0 #no units (number of photons)
MAX_AMP = 10 # Constant determined by the AWG in the lab

T_INITIAL = 0.0
T_FINAL = 2.0
T_LIST = np.linspace(T_INITIAL, T_FINAL, 50)

H_JC = WR*tensor(A.dag()*A, qeye(DIM_Q)) + tensor(qeye(DIM_RES), WQ*sigmaz()/2) + X*tensor(A.dag()*A, sigmaz()) + X/2*tensor(qeye(DIM_RES), sigmaz())
H_D1 = tensor(A.dag(), qeye(DIM_Q))
H_D2 = tensor(A, qeye(DIM_Q))

INITIAL_STATE = tensor(coherent(DIM_RES, 2), fock(2,0)) # second input in coherent squared gives the init num of photons


class SingleQNKEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SingleQNKEnv, self).__init__()
        self.count = 0
        
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        
        self.observation_space = spaces.Box(np.array([-1, -1, 0]), np.array([1, 1, 20]),
                                            dtype=np.float32)

    def step(self, action):
        info = {}
        if self.count == 0:
            self.past_action = np.array([0, 0], dtype=np.float32)
            self.past_sum = self.nat_decay
        
        input_arr = np.zeros_like(T_LIST)

        for i in range(25):
            input_arr[i] = action[0]*MAX_AMP
        for i in range(25, 50):
            input_arr[i] = action[1]*MAX_AMP
        
        S1_inp = input_arr*np.exp(-1j*WR*T_LIST)
        S2_inp = input_arr*np.exp(1j*WR*T_LIST)
        
        S1 = Cubic_Spline(T_LIST[0], T_LIST[-1], S1_inp)
        S2 = Cubic_Spline(T_LIST[0], T_LIST[-1], S2_inp)
        
        H_experimental = [H_JC, [H_D1, S1], [H_D2, S2]]
        result_experimental = mesolve(H_experimental, INITIAL_STATE, T_LIST, c_ops=[np.sqrt(K)*tensor(A, qeye(DIM_Q))], e_ops=[tensor(A.dag()*A, qeye(DIM_Q))])
        
        self.reward = -np.sum(result_experimental.expect[0])
        self.observation = np.array([self.past_action[0], self.past_action[1], self.past_sum/10], dtype=np.float32)

        self.past_sum = -self.reward
        self.past_action = action
        
        self.count += 1
        self.done = True
        
        return self.observation, self.reward, self.done, info
    
    def reset(self):
        self.done = False
        
        result_reset = mesolve(H_JC, INITIAL_STATE, T_LIST, c_ops=[np.sqrt(K)*tensor(A, qeye(DIM_Q))], e_ops=[tensor(A.dag()*A, qeye(DIM_Q))])
        self.nat_decay = np.sum(result_reset.expect[0])        

        self.observation = np.array([0, 0, self.nat_decay/10], dtype=np.float32)
        
        return self.observation
    
    def render(self, mode='human'):
        
        results_list = []
        for t in T_LIST:
            results_list.append(PNORM*np.exp(-K*t))
        
        input_arr = np.zeros_like(T_LIST)

        for i in range(25):
            input_arr[i] = self.past_action[0]*MAX_AMP
        for i in range(25, 50):
            input_arr[i] = self.past_action[1]*MAX_AMP
        
        S1_inp = input_arr*np.exp(-1j*WR*T_LIST)
        S2_inp = input_arr*np.exp(1j*WR*T_LIST)
        
        S1 = Cubic_Spline(T_LIST[0], T_LIST[-1], S1_inp)
        S2 = Cubic_Spline(T_LIST[0], T_LIST[-1], S2_inp)
        
        H_experimental = [H_JC, [H_D1, S1], [H_D2, S2]]
        result_experimental = mesolve(H_experimental, INITIAL_STATE, T_LIST, c_ops=[np.sqrt(K)*tensor(A, qeye(DIM_Q))], e_ops=[tensor(A.dag()*A, qeye(DIM_Q))])

        fig, axes = plt.subplots(figsize=(10,5))
        axes.plot(T_LIST, results_list)
        axes.plot(result_experimental.times, result_experimental.expect[0])
        axes.set_title('Photon Population in Cavity with Experimental Data', fontsize=30)
        axes.set_xlabel(r'$t$', fontsize=20)
        axes.set_ylabel(r'$a.dag*a$', fontsize=20)
        axes.legend(fontsize=15)
        plt.show()