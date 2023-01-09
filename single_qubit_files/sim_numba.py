import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from numba import types
from numbalsoda import lsoda, dop853, address_as_void_pointer

T_LENGTH = 100

KAPPA = 5.35
BURT = np.sqrt(KAPPA)
CHI = 0.16*KAPPA
P1 = KAPPA/2
P2 = CHI/2
AMP = 10.*BURT
T_MAX = 0.8
T_EVAL = np.linspace(0.0, T_MAX, T_LENGTH + 1)

MAX_PHOTON = 7
MIN_PHOTON = 0.1

BLOB_WIDTH = 0.5
IDEAL_SIGMA_NUMBER = 1.96 # 2.35 for 98% fidelity readout, 1.96 for 95% fidelity readout


U0 = np.array([0., 0.])

def rhs_g(t, u, du, a, b, arr_complex):
    du[0] = -a*u[0] - b*u[1] - 12#+ arr_complex[int(t*T_LENGTH/T_MAX)]
    du[1] = -a*u[1] + b*u[0] #+ arr_complex[int(t*T_LENGTH/T_MAX + T_LENGTH + 1)]

def rhs_e(t, u, du, a, b, arr_complex):
    du[0] = -a*u[0] + b*u[1] - 12#+ arr_complex[int(t*T_LENGTH/T_MAX)]
    du[1] = -a*u[1] - b*u[0] #+ arr_complex[int(t*T_LENGTH/T_MAX + T_LENGTH + 1)]

# 'P1' is the value of P1
# 'P2' is the value of P2
# 'arr_p' is the memory address of array arr_real
# 'len_arr' is the length of array arr_real

args_dtype = types.Record.make_c_struct([
    ('a', types.float32),
    ('b', types.float32),
    ('arr_p', types.int64),
    ('len_arr', types.int64)])

# this function will create the numba function we pass to lsoda
def create_jit_rhs(rhs, args_dtype):
    jitted_rhs = nb.njit(rhs)
    @nb.cfunc(types.void(
        types.double,
        types.CPointer(types.double),
        types.CPointer(types.double),
        types.CPointer(args_dtype)))
    def wrapped(t, u, du, user_data_p):
        # unpack p and arr from user_data_p
        user_data = nb.carray(user_data_p, 1)
        a = user_data[0].a
        b = user_data[0].b
        arr_real = nb.carray(address_as_void_pointer(user_data[0].arr_p), (user_data[0].len_arr), dtype=np.float32)

        # then we call the jitted rhs function, passing in data
        jitted_rhs(t, u, du, a, b, arr_real)
    return wrapped

# create the function to be called by lsoda
rhs_g_cfunc = create_jit_rhs(rhs_g, args_dtype)
rhs_e_cfunc = create_jit_rhs(rhs_e, args_dtype)

funcptr_g = rhs_g_cfunc.address
funcptr_e = rhs_e_cfunc.address

class NumbaPulseEnv(gym.Env):
    def __init__(self):
        super(NumbaPulseEnv, self).__init__()

        lower_bound = np.zeros(2*(T_LENGTH + 1) + 1) - 1
        upper_bound = np.zeros(2*(T_LENGTH + 1) + 1) + 1

        self.action_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)
        self.observation_space = spaces.Box(-10.e5*np.ones(4), 10.e5*np.ones(4), dtype=np.float32)

    def step(self, action):
        info = {}
        pulse_part = action[:-1]
        index = int((action[-1] + 1)/2*(T_LENGTH + 1))

        pulse_part[index:T_LENGTH + 1] = 0
        pulse_part[T_LENGTH + 1 + index:] = 0

        pulse = AMP*pulse_part
        arr_pulse = np.ascontiguousarray(pulse)
        args = np.array((P1, P2, arr_pulse.ctypes.data, arr_pulse.shape[0]), dtype=args_dtype)

        usol_g, _ = lsoda(funcptr_g, U0, T_EVAL, data = args)
        usol_e, _ = lsoda(funcptr_e, U0, T_EVAL, data = args)

        distance_ratio = np.sqrt( np.abs(usol_g[:,1] - usol_e[:,1])**2 + np.abs(usol_g[:,0] - usol_e[:,0])**2 )/BLOB_WIDTH
        avg_photon = 0.5*((usol_g[:,0])**2 + (usol_g[:,1])**2 + (usol_e[:,0])**2 + (usol_e[:,1])**2)
        penalty = 0
        if max(avg_photon) > MAX_PHOTON:
            penalty += 1000
        
        delta = 0
        if avg_photon[-1] > MIN_PHOTON:
            delta = np.log(avg_photon[-1]/MIN_PHOTON)/KAPPA
        
        deviation_of_ideal = np.abs(2*IDEAL_SIGMA_NUMBER - max(distance_ratio))

        self.reward = -penalty - 10*deviation_of_ideal/(2*IDEAL_SIGMA_NUMBER) - 4*index/(T_LENGTH + 1)*T_MAX - 5*(1 - np.sum(np.abs(pulse_part))/(2*index + 1))

        self.observation = np.array([avg_photon[-1], index/(T_LENGTH), deviation_of_ideal, max(avg_photon)], dtype=np.float32)

        self.done = True

        return self.observation, self.reward, self.done, info
    
    def reset(self):
        self.done = False
        self.observation = np.array([0, 0, 0, 0], dtype=np.float32)
        return self.observation
    
    def grapher(self, action):
        beep = np.array2string(action, separator=",")
        print(f"action: {beep}")
        pulse_part = action[:-1]
        index = int((action[-1] + 1)/2*(T_LENGTH + 1))

        pulse_part[index:T_LENGTH + 1] = 0
        pulse_part[T_LENGTH + 1 + index:] = 0

        pulse = AMP*pulse_part
        arr_pulse = np.ascontiguousarray(pulse)
        args = np.array((P1, P2, arr_pulse.ctypes.data, arr_pulse.shape[0]), dtype=args_dtype)

        usol_g, _ = lsoda(funcptr_g, U0, T_EVAL, data = args)
        usol_e, _ = lsoda(funcptr_e, U0, T_EVAL, data = args)

        distance_ratio = np.sqrt( np.abs(usol_g[:,1] - usol_e[:,1])**2 + np.abs(usol_g[:,0] - usol_e[:,0])**2 )/BLOB_WIDTH
        max_index = np.where(distance_ratio == max(distance_ratio))
        graphing_index = max_index[0]
        avg_photon = 0.5*((usol_g[:,0])**2 + (usol_g[:,1])**2 + (usol_e[:,0])**2 + (usol_e[:,1])**2)
        g_photon = (usol_g[:,0])**2 + (usol_g[:,1])**2
        e_photon = (usol_e[:,0])**2 + (usol_e[:,1])**2
        penalty = 0
        if max(avg_photon) > MAX_PHOTON:
            penalty += 1000
        
        delta = 0
        if avg_photon[-1] > MIN_PHOTON:
            delta = np.log(avg_photon[-1]/MIN_PHOTON)/KAPPA
        
        deviation_of_ideal = np.abs(2*IDEAL_SIGMA_NUMBER - max(distance_ratio))

        reward = -10*(10*deviation_of_ideal/(2*IDEAL_SIGMA_NUMBER) + index*T_MAX/(T_LENGTH + 1) + delta) - penalty

        print(f"Reward: {reward}")
        print(f"Maximum Sigma Number: {max(distance_ratio)}")
        print(f"Max Photon: {max(avg_photon)}")
        print(f"Final Photon: {avg_photon[-1]}")
        print(f"Index Time: {index*T_MAX/(T_LENGTH + 1)}")
        print(f"Total Time till Base Photon: {index*T_MAX/(T_LENGTH + 1) + delta}")

        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots(1)

        #ax.plot(T_EVAL, usol_g[:,0], label='real', color='blue')
        #ax.plot(T_EVAL, usol_g[:,1], label='imag', color='orange')

        '''
        ax[1, 0].plot(usol_g[:,1], usol_g[:,0], label='ground')
        ax[0, 0].legend()
        ax[0, 0].set_xlabel('time')
        ax[1, 0].legend()
        ax[1, 0].set_xlabel('imag')
        ax[1, 0].set_ylabel('real')

        ax[0, 1].plot(T_EVAL, usol_e[:,0], label='real')
        ax[0, 1].plot(T_EVAL, usol_e[:,1], label='imag')
        ax[1, 1].plot(usol_e[:,1], usol_e[:,0], label='excited')
        ax[0, 1].legend()
        ax[0, 1].set_xlabel('time')
        ax[1, 1].legend()
        ax[1, 1].set_xlabel('imag')
        ax[1, 1].set_ylabel('real')
        '''

        #ax.plot(T_EVAL, arr_pulse[:T_LENGTH + 1], label='real part', color='blue')
        #ax.plot(T_EVAL, arr_pulse[T_LENGTH + 1:], label='imag part', color='orange')

        ax.plot(usol_g[:,1], usol_g[:,0], label='ground', color="blue")
        ax.plot(usol_e[:,1], usol_e[:,0], label='excited', color="orange")
        ax.plot(0, 0, marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black")
        ax.plot(usol_g[-1, 1], usol_g[-1, 0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        ax.plot(usol_e[-1, 1], usol_e[-1, 0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")
        ax.plot(usol_g[graphing_index, 1], usol_g[graphing_index, 0], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
        ax.plot(usol_e[graphing_index, 1], usol_e[graphing_index, 0], marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")

        bar = np.zeros_like(T_EVAL) + 0.1

        #ax.plot(T_EVAL, g_photon, color="blue", label="ground state")
        #ax.plot(T_EVAL, e_photon, color="orange", label="excited state")
        plt.axvline(x = (action[-1] + 1)/2, color="black", linestyle="dashed")

        plt.ylim(-5., 5.)
        plt.xlim(-2.2, 1.)
        plt.xlabel("Time (us)")
        plt.ylabel("Photon Count")

        plt.legend(loc="upper right")
        plt.show()