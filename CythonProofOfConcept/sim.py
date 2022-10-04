import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

DIM_RES = 10
DIM_Q = 2
A = destroy(DIM_RES)
WR = 7.062  # GHz frequency of resonator
WQ = 5.092  # GHz frequency of qubit
G = 102.9  # MHz coupling strength
K = 5.35  # MHz kappa, photon decay rate
X = 0.856  # MHz kai, dispersive shift
P1PH = 36.1  # MHz power required to maintain a population of one photon
PNORM = 4.0  # no units (number of photons)
MAX_AMP = 10  # Constant determined by the AWG in the lab

T_INITIAL = 0.0
T_FINAL = 2.0
LENGTH = 100
ACTUAL_START = (T_FINAL - T_INITIAL) / LENGTH + T_INITIAL
T_LIST = np.linspace(ACTUAL_START, T_FINAL, 50)

H_JC = (
    WR * tensor(A.dag() * A, qeye(DIM_Q))
    + tensor(qeye(DIM_RES), WQ * sigmaz() / 2)
    + X * tensor(A.dag() * A, sigmaz())
    + X / 2 * tensor(qeye(DIM_RES), sigmaz())
)
H_D1 = tensor(A.dag(), qeye(DIM_Q))
H_D2 = tensor(A, qeye(DIM_Q))

INITIAL_STATE = tensor(
    coherent(DIM_RES, 2), fock(2, 0)
)  # second input in coherent squared gives the init num of photons

NAT_DECAY = PNORM * np.exp(-K * T_LIST)

OPTS = Options(rhs_reuse=True)


class SingleQNKEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(SingleQNKEnv, self).__init__()
        self.count = 0

        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            np.array([-1, -1, 0]), np.array([1, 1, 20]), dtype=np.float32
        )

    def step(self, action):
        info = {}

        self.initial_state = tensor(
            coherent(DIM_RES, np.sqrt(PNORM)), fock(DIM_Q, 0)
        )  # second input in coherent squared gives the init num of photons
        args = {"A1": action[0], "A2": action[1], "WR": WR}

        H_first = [H_JC, [H_D1, "A1*exp(-1j*WR*t)"], [H_D2, "A1*exp(1j*WR*t)"]]
        H_second = [H_JC, [H_D1, "A2*exp(-1j*WR*t)"], [H_D2, "A2*exp(1j*WR*t)"]]


        first_time = int((action[2] + 1)*24)
        second_time = int((action[3] + 1)*24)

        if first_time == 0:
            first_time = 1
        if second_time == 0:
            second_time = 1

        if self.count == 0:
            self.past_action = np.array([0, 0], dtype=np.float32)
            self.past_sum = np.sum(NAT_DECAY)

            result_first = mesolve(
                H_first,
                INITIAL_STATE,
                T_LIST[0:first_time],
                c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
                e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
                args=args,
            )

            self.initial_state = tensor(
                coherent(DIM_RES, np.sqrt(result_first.expect[0][-1])), fock(2, 0)
            )  # second input in coherent squared gives the init num of photons

            result_second = mesolve(
                H_second,
                INITIAL_STATE,
                T_LIST[first_time:first_time + second_time],
                c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
                e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
                args=args,
                options=OPTS,
            )

        else:
            result_first = mesolve(
                H_first,
                INITIAL_STATE,
                T_LIST[0:first_time],
                c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
                e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
                args=args,
                options=OPTS,
            )

            self.initial_state = tensor(
                coherent(DIM_RES, np.sqrt(result_first.expect[0][-1])), fock(2, 0)
            )  # second input in coherent squared gives the init num of photons

            result_second = mesolve(
                H_second,
                INITIAL_STATE,
                T_LIST[first_time:first_time + second_time],
                c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
                e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
                args=args,
                options=OPTS,
            )
        
        num_photons = result_second.expect[0][-1]
        
        rest_of_decay = np.sum(num_photons*np.exp(-K*T_LIST[(first_time + second_time):(LENGTH - 1)]))

        self.reward = -np.sum(result_first.expect[0]) - np.sum(result_second.expect[0]) - rest_of_decay
        self.observation = np.array(
            [self.past_action[0], self.past_action[1], self.past_sum / 10],
            dtype=np.float32,
        )

        self.past_sum = -self.reward
        self.past_action = action

        self.count += 1
        self.done = True

        return self.observation, self.reward, self.done, info

    def reset(self):
        self.done = False

        self.observation = np.array([0, 0, np.sum(NAT_DECAY) / 10], dtype=np.float32)

        return self.observation

    def render(self):

        args = {"A1": self.past_action[0], "A2": self.past_action[1], "WR": WR}

        H_first = [H_JC, [H_D1, "A1*exp(-1j*WR*t)"], [H_D2, "A1*exp(1j*WR*t)"]]
        H_second = [H_JC, [H_D1, "A2*exp(-1j*WR*t)"], [H_D2, "A2*exp(1j*WR*t)"]]

        first_time = int((self.past_action[2] + 1)*24)
        second_time = int((self.past_action[3] + 1)*24)

        if first_time == 0:
            first_time = 1
        if second_time == 0:
            second_time = 1

        result_first = mesolve(
            H_first,
            INITIAL_STATE,
            T_LIST[0:first_time],
            c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
            e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
            args=args,
        )

        self.initial_state = tensor(
            coherent(DIM_RES, np.sqrt(result_first.expect[0][-1])), fock(2, 0)
        )  # second input in coherent squared gives the init num of photons

        result_second = mesolve(
            H_second,
            INITIAL_STATE,
            T_LIST[first_time:first_time + second_time],
            c_ops=[np.sqrt(K) * tensor(A, qeye(DIM_Q))],
            e_ops=[tensor(A.dag() * A, qeye(DIM_Q))],
            args=args,
            options=OPTS,
        )

        num_photons = result_second.expect[0][-1]

        rest_of_decay = num_photons*np.exp(-K*T_LIST[first_time + second_time:LENGTH - 1])

        fig, axes = plt.subplots(figsize=(10, 5))
        axes.plot(T_LIST, NAT_DECAY)
        axes.plot(result_first.times, result_first.expect[0])
        axes.plot(result_second.times, result_second.expect[0])
        axes.plot(T_LIST[first_time + second_time:LENGTH - 1], rest_of_decay)
        axes.set_title(
            "Photon Population in Cavity with Experimental Data", fontsize=30
        )
        axes.set_xlabel(r"$t$", fontsize=20)
        axes.set_ylabel(r"$a.dag*a$", fontsize=20)
        axes.legend(fontsize=15)
        plt.show()
