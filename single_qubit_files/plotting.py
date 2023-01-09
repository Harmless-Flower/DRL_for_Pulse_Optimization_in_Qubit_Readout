import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
from sim_numba import NumbaPulseEnv

a2c_action = np.array([ 1. ,        -1.,         -1. ,        -1.  ,        1.  ,       -1., 1.      ,   -1.        , -1.        , -1.      ,    1.     ,     1.,-1.      ,   -1.        , -1.        ,  1.      ,   -1.     ,     1., 1.       ,  -1.       ,  -1.        ,  1.      ,   -1.     ,    -1.,-1.        , -1.       ,   1.        , -1.      ,   -1.      ,   -1.,1.      ,    1.       ,  -0.336392  ,  1.      ,    1.      ,    1.,-1.       ,  -1.       ,   1.        ,  1.      ,    1.      ,    1.,1.        ,  1.       ,   0.8293851 ,  1.      ,    1.      ,   -1.,1.         , 1.       ,  -1.        , -1.      ,   -0.96721566, -1.,-0.24950564  ,1.       ,  -1.       ,  -1.      ,    0.26941156,  1.,1.      ,    1.       ,  -1.       ,   1.     ,    1.    ,     -0.44709712,0.6689135,   1.       ,  -1.       ,  -1.     ,    -1.    ,     -1.,-1.        ,  1.      ,   -1.       ,   1.     ,     1.    ,     -0.6928553,1.         ,-1.      ,    1.       ,  -1.     ,     1.    ,      1.,-1.       ,  -1.      ,   -0.35014066, -1.     ,    -1.    ,      1.,1.       ,  -1.      ,    0.20138514,  1.     ,     1.    ,     -1.,-1.       ,   1.      ,   -1.       ,  -1.     ,    -1.    ,     -1.,1.       ,  -1.       ,   1.      ,   -1.     ,    -1.     ,    -1.,-1.       ,  -1.       ,   1.      ,   -1.     ,     1.    ,      1.,-1.       ,   1.       ,   1.      ,   -1.     ,     1.     ,    -1.,-1.       ,   1.       ,  -1.      ,   -1.     ,     1.    ,     -1.,1.       ,   1.       ,  -1.      ,    1.     ,     1.    ,      1.,1.       ,  -1.       ,  -1.      ,   -1.     ,     1.    ,     -1.,-0.9483539 , -1.      ,  -0.31218666,  1.      ,   -1.     ,    1.,-1.       ,  -1.       ,  -1.       ,  -1.     ,     1.     ,    -1.,-0.81197673,  0.92803955, -1.       ,   1.     ,     1.     ,     1.,1.       ,  -1.      ,    1.       ,   1.      ,   -1.      ,   -1.,0.3731749,   1.      ,    0.57141453,  0.5887108,   1.      ,    1.,-1.       ,  -1.      ,   -1.       ,  -1.       ,   0.4175706,  -1.,-0.879663 ,  -1.      ,   -0.6067195,  -1.       ,  -1.       ,   1.,0.66683084,  0.60324395, -1.        , -1.       ,   1.       ,   1., -1.        ,  1.      ,   -1.        ,  1.       ,   0.98484707, -1., 1.        ,  1.     ,    -0.07767826,  0.07931035, -1.       ,   1., -1.        ,  1.     ,   -1.        , -1.    ,     -0.68359584], dtype=np.float32)
'''
Reward: -1.5471195301489036
Maximum Sigma Number: 3.917570736200341
Max Photon: 3.7394783724289673
Final Photon: 0.038869743883808
Index Time: 0.1485148514851485
Total Time till Base Photon: 0.1485148514851485


double oscilatting A2C Best
'''

ppo_action = np.array([-1.        ,  1.     ,     1.     ,    -1.   ,       1.    ,     -1., 1.     ,    -1.     ,     1.     ,    -1.    ,      1.     ,    -1.,1.    ,     -1.       ,   1.      ,    1.      ,   -1.       ,  -0.1117062,0.52959776, -0.39699757,  1.       ,   1.      ,    0.20550495, -0.71878016,-1.  ,       -1.        , -0.0232003 , -0.63579553 , 0.79241776,  0.0164426,1.  ,        0.9096845 , -1.       ,  -1.       ,  -0.6690966 ,  0.26798967,-1.   ,      -1.        , -0.48014277,  0.8187636,   1.       ,   1.,0.06791082, -0.44212592,  0.12411082,  0.9462005 , -1.       ,   0.21558234,1.   ,      -0.33007163,  1.      ,   -0.67611945,  1.       ,  -0.59498626,-1.   ,      -1.       ,  -1.       ,  -1.      ,    1.       ,   0.77813876,0.22991055, -0.10302337, -0.59483445, -0.4792659 ,  0.9609019,   0.33892095,-1.        , -1.        , -0.43687788,  0.7492797 , -0.10157925,  0.68947905,0.8186067 ,  0.16616935,  0.90731066,  0.8528078 ,  0.3887686 ,  0.21214911,-0.21200599, -0.17848074, -1.        ,  0.8315835 , -1.        ,  0.2364211,-1.        ,  1.        , -0.8315617,   0.45651072, -0.91223025, -0.23219155,0.45859426,  0.59842056,  1.       ,  -0.48318028,  0.84399015,  1.,1.        , -1.        , -1.       ,   1.        ,  1.      ,   -1.,-1.        , -1.        , -1.       ,  -1.        , -1.      ,   -1.,-1.        , -1.        , -1.       ,  -1.       ,  -1.       ,   1.,1.        ,  1.        , -1.       ,  -0.37672627 , 0.29709154,  1.,-1.        , -0.911713   ,-1.        ,  0.3084799 ,  0.9571123,   0.32889882,-1.        , -0.11765715 , 0.5184518 ,  1.        ,  0.6889018 ,  0.3585199,1.        , -0.68760943 , 1.        , -1.        ,  0.15723637,  0.175628,-0.2958938 , -0.72792673 , 0.6901718 , -0.7499737 , -0.5702055 ,  1.,-1.        , -1.       ,  -0.57913285,  1.       ,  -1.      ,    0.07350099,-0.5678804 ,-0.61218065, -1.        ,  0.43309096,  0.94945586, -0.6852732,-1.        , -1.        , -0.10969406, -0.7063773,  -0.6419235,  -0.47997752,0.41907972, -0.20386547,  1.       ,  -1.       ,   1.       ,   1.,0.02629274,  0.6568479 ,  0.6316981 ,  0.03623001, -0.10647291,  0.48147893,0.5351987 , -0.12699026,  0.41194496, 0.55366087 , 1.     ,    -1.,1.        , -1.       ,  -0.79427063 ,-0.09312121,  1.     ,    -0.00570703,0.96336454, -0.03085312, -0.34985903,  0.6911964 , -1.     ,    -0.11525591,0.9351214 , -1.       ,  -1.        , -0.9340818 , -0.8049302 ,  0.15736449,-1.        ,  0.4330181,   0.47222978,  0.48175678, -0.7992412 ], dtype=np.float32)
'''
Reward: -51.3352004718131
Maximum Sigma Number: 1.9464720226930454
Max Photon: 4.912774355721767
Final Photon: 0.03981925780510866
Index Time: 0.09900990099009901
Total Time till Base Photon: 0.09900990099009901

Single Oscillating PPO Best
'''

trpo_action = np.array([ 1.        , 1.        , 1.        , 1.        , 1.        , 1.        ,
  1.        ,-1.        ,-1.        ,-1.        ,-1.        ,-0.36601377,
  1.        ,-0.89295566, 1.        ,-0.09456889,-1.        , 0.00104988,
  0.03124841,-1.        ,-0.4031785 , 0.28727347,-0.79074305, 0.27006435,
  1.        ,-1.        , 1.        , 0.936998  , 0.5307789 , 0.3617764 ,
 -0.754887  , 0.5981301 , 0.98091257, 0.3667496 ,-0.16064757,-0.7529854 ,
 -1.        ,-0.2282331 , 0.34077203, 1.        ,-0.5503279 ,-1.        ,
 -0.18136153,-1.        , 0.9974794 , 1.        ,-0.3563732 ,-0.86653054,
  0.91429013, 1.        , 0.550612  , 1.        , 0.38806653,-0.8634364 ,
 -0.5197232 ,-1.        , 0.47790712, 0.8241264 ,-1.        ,-1.        ,
  0.27961951, 0.4074657 , 1.        ,-0.0552238 ,-1.        , 0.12564763,
 -1.        ,-1.        , 0.17573181,-0.50085795, 1.        , 0.18818858,
  1.        ,-0.6035902 ,-1.        ,-0.97979677,-1.        , 0.47044504,
 -0.6873621 ,-0.6512276 , 1.        , 1.        , 1.        ,-1.        ,
  0.5941422 ,-1.        ,-0.7069895 , 1.        , 1.        ,-0.9137994 ,
  0.39179   , 0.16487458, 0.39603138, 0.9570025 , 1.        ,-1.        ,
  0.40885323,-0.8568475 ,-0.8469819 , 0.7353902 ,-0.91870534,-0.4083714 ,
 -1.        , 1.        , 1.        ,-1.        , 1.        , 1.        ,
 -1.        , 1.        ,-0.82604796,-1.        ,-1.        ,-0.58624804,
 -0.09837419,-0.04842887, 0.56660104, 1.        ,-0.53017414, 1.        ,
 -0.30681437,-0.7459622 ,-1.        , 0.3556865 ,-1.        , 1.        ,
 -0.9616833 , 0.3209464 , 1.        ,-1.        , 1.        ,-1.        ,
 -1.        ,-0.07034837, 0.94254535,-1.        , 1.        ,-0.8741735 ,
  0.3489793 ,-0.82660604, 0.55975944,-0.12463811, 1.        ,-0.45867136,
 -0.523837  ,-0.5347406 , 1.        ,-0.46052897,-0.5717002 , 1.        ,
 -1.        ,-1.        , 0.03019045, 0.6804335 , 0.60019946, 0.26345223,
  0.22328667, 0.86610055,-0.45664987, 1.        , 0.02293813,-0.46345568,
  1.        , 0.38127604,-1.        ,-1.        , 1.        , 0.7017162 ,
 -0.48460174,-0.09746784, 0.26312625, 1.        ,-1.        , 0.45518544,
 -0.5557377 , 0.09132192, 1.        ,-0.36640626,-1.        , 0.2620306 ,
 -0.28253537, 0.53659004,-0.12332214, 1.        ,-1.        , 1.        ,
  1.        ,-1.        , 1.        ,-0.54560316, 0.53345025,-0.81351995,
  1.        ,-0.24831733, 0.857188  , 1.        ,-0.7119541 , 1.        ,
  0.56351215, 0.43565333,-1.        , 0.05996308,-0.8100223 ], dtype=np.float32)

'''
Reward: -78.80449956284366
Maximum Sigma Number: 0.8657943102058355
Max Photon: 2.567302884511963
Final Photon: 0.009634914336169254
Index Time: 0.0891089108910891
Total Time till Base Photon: 0.0891089108910891

TRPO, Tries to do some oscillation, but fails at doing much more than no pulse
'''

env = NumbaPulseEnv()

models_dir = "models"
model_path = f"{models_dir}/1673195597_TRPO/700000.zip"
model = TRPO.load(model_path, env=env)
obs = env.reset()
action, _ = model.predict(obs)

action = np.ones(2*101 + 1)
action[-1] = -0.99
action[101:-1] = 0

env.grapher(action)