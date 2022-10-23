from re import A
import gym
from gym import spaces
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

KAI = 1 #MHz
KAPPA = 1 #MHz
KERR = 0.01#MHz

N_CRIT = 15

MAX_SIG = 1.5

T_START = 0 #microseconds
T_END = 2 #microseconds
T_LENGTH = 100
T_LIST = np.linspace(T_START, T_END, T_LENGTH + 1)

INIT_BLOB_RADIUS = 0.25
IDEAL_RATIO = 3
SIGMA = 0

INIT_STATE = np.array([0,0])

def KERR_GROUND(alpha, t, real_in, imag_in):
    alphar, alphai = alpha
    d_alphar_dt = -KAI/2*alphai - KAPPA/2*alphar - np.sqrt(KAPPA)*real_in(t) + KERR*(alphar**2 + alphai**2)
    d_alphai_dt = KAI/2*alphar - KAPPA/2*alphai - np.sqrt(KAPPA)*imag_in(t)
    return [d_alphar_dt, d_alphai_dt]

def KERR_EXCITED(alpha, t, real_in, imag_in):
    alphar, alphai = alpha
    d_alphar_dt = KAI/2*alphai - KAPPA/2*alphar - np.sqrt(KAPPA)*real_in(t) + KERR*(alphar**2 + alphai**2)
    d_alphai_dt = -KAI/2*alphar - KAPPA/2*alphai - np.sqrt(KAPPA)*imag_in(t)
    return [d_alphar_dt, d_alphai_dt]

class PhaseEnv(gym.Env):
    def __init__(self):
        super(PhaseEnv, self).__init__()

        lower_bound = np.zeros(2*(T_LENGTH + 1)) - 1
        upper_bound = np.zeros(2*(T_LENGTH + 1)) + 1

        self.action_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-1000, -1000, -1000, -1000]), np.array([1000, 1000, 1000, 1000]), dtype=np.float32)
    
    def step(self, action):
        info = {}

        real_portion = action[0:T_LENGTH + 1]*MAX_SIG
        imag_portion = action[T_LENGTH + 1:]*MAX_SIG

        real_func = interp1d(T_LIST, real_portion, bounds_error=False, fill_value="extrapolate")
        imag_func = interp1d(T_LIST, imag_portion, bounds_error=False, fill_value="extrapolate")

        results_g = odeint(KERR_GROUND, INIT_STATE, T_LIST, args=(real_func, imag_func))
        results_e = odeint(KERR_GROUND, INIT_STATE, T_LIST, args=(real_func, imag_func))

        distance = np.sqrt( (results_g[:,0] - results_g[:,0])**2 + (results_e[:,1] - results_e[:,1])**2 )
        photon_g = results_g[:,0]**2 + results_g[:,1]**2
        photon_e = results_e[:,0]**2 + results_e[:,1]**2
        penalty = 0
        if (max(photon_e) + max(photon_g))/2 > N_CRIT:
            penalty = 100
        index = np.where(distance == max(distance))[-1][-1]
        final_time = T_END
        for i in range(T_LENGTH - index):
            if (photon_e[i + index - 1] + photon_g[i + index - 1])/2 < 0.1:
                final_time = T_LIST[i + index - 1]
                break
        
        ratio = max(distance)/(2*INIT_BLOB_RADIUS)
        if ratio <= 1:
            penalty += 100
        avg_final = (results_g[-1,0]**2 + results_g[-1,1]**2 + results_e[-1,0]**2 + results_e[-1,1]**2)/2

        if avg_final < 0.1:
            avg_final = 0

        if ratio >= IDEAL_RATIO:
            self.reward = -(final_time) - avg_final - penalty
        else:
            self.reward = -(final_time) - avg_final - (IDEAL_RATIO - ratio) - penalty

        self.observation = np.array([avg_final, final_time, ratio, (max(photon_e) + max(photon_g))/2], dtype=np.float32)

        self.done = True

        return self.observation, self.reward, self.done, info
    
    def reset(self):
        self.done = False

        self.observation = np.array([1/3*N_CRIT, T_END, IDEAL_RATIO, 2/3*N_CRIT], dtype=np.float32)

        return self.observation
    
    def grapher(self, action, mode):
        real_portion = action[0:T_LENGTH + 1]
        imag_portion = action[T_LENGTH + 1:]

        real_func = interp1d(T_LIST, real_portion, bounds_error=False, fill_value="extrapolate")
        imag_func = interp1d(T_LIST, imag_portion, bounds_error=False, fill_value="extrapolate")

        results_g = odeint(KERR_GROUND, INIT_STATE, T_LIST, args=(real_func, imag_func))
        results_e = odeint(KERR_GROUND, INIT_STATE, T_LIST, args=(real_func, imag_func))

        distance = np.sqrt( (results_g[:,0] - results_g[:,0])**2 + (results_e[:,1] - results_e[:,1])**2 )
        photon_g = results_g[:,0]**2 + results_g[:,1]**2
        photon_e = results_e[:,0]**2 + results_e[:,1]**2
        penalty = 0
        if (max(photon_e) + max(photon_g))/2 > N_CRIT:
            penalty = 100
        index = np.where(distance == max(distance))[-1][-1]
        final_time = T_END
        for i in range(T_LENGTH - index):
            if (photon_e[i + index - 1] + photon_g[i + index - 1])/2 < 0.1:
                final_time = T_LIST[i + index - 1]
                break
        
        ratio = max(distance)/(2*INIT_BLOB_RADIUS)
        if ratio <= 1:
            penalty += 100
        avg_final = (results_g[-1,0]**2 + results_g[-1,1]**2 + results_e[-1,0]**2 + results_e[-1,1]**2)/2

        if avg_final < 0.1:
            avg_final = 0

        if ratio >= IDEAL_RATIO:
            self.reward = -(final_time) - avg_final - penalty
        else:
            self.reward = -(final_time) - avg_final - (IDEAL_RATIO - ratio) - penalty
        
        print(f"Reward: {self.reward}")

        if mode == 0:
            plt.clf()
            plt.plot(T_LIST, results_g[:,0], label="Real Part of Kerr Ground State")
            plt.plot(T_LIST, results_g[:,1], label="Imag Part of Kerr Ground State")
            plt.plot(T_LIST, results_e[:,0], label="Real Part of Kerr Excited State")
            plt.plot(T_LIST, results_e[:,1], label="Imag Part of Kerr Excited State")
            plt.xlabel("Time")
            plt.ylabel("Resonator State")
            plt.grid(True)
            plt.legend(loc='best')
            plt.show()
        if mode == 1:
            plt.clf()
            plt.plot(results_g[:,1], results_g[:,0], label="Ground State")
            plt.plot(results_e[:,1], results_e[:,0], label="Excited State")
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.grid(True)
            plt.legend(loc='best')
            plt.show()
        if mode == 2:
            plt.clf()
            plt.plot(T_LIST, action[0:T_LENGTH + 1], label="Real Part of the Pulse")
            plt.plot(T_LIST, action[T_LENGTH + 1:], label="Imaginary Part of the Pulse")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend(loc='best')
            plt.show()