# eval_headless_naive.py
# 
# A version of eval_headless but for benchmarking handcrafted policies that 
# need to have the actions drawn one by one (like the handcrafted naïve policy 
# or a random policy sampling the action space)

# Library Imports
from env import SimpleSim, NaivePolicy
import math
import numpy as np
import time
import random
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Load Custom Environment
from env_gym import SimpleSimGym


# ------------------------------ Hyperparameters ----------------------------- #
# Env
MAX_BUDGET = 400
MAX_TARGETS = 5
NUM_CLASSES = 10
PLAYER_FOV = 30
ACTION_FORMAT = "continuous"

# Eval
num_episodes = 10
RENDER_PLOTS = True


# -------------------------------- Environment ------------------------------- #
# Instantiate two environments: one for training and one for evaluation.
env = SimpleSimGym(
    max_budget=MAX_BUDGET,
    max_targets=MAX_TARGETS,
    num_classes=NUM_CLASSES,
    player_fov=PLAYER_FOV,
    action_format=ACTION_FORMAT,
    render_mode=None,
    # seed=808,
)
obs = env.reset()

# --------------------------- LOAD MODEL IF DESIRED -------------------------- #

# --------------------------------- RUN EVAL --------------------------------- #
num_episodes = 500 # number of episodes to eval over
obs = env.reset()

ep_rewards = []

for i in range(num_episodes):
    terminated = False
    truncated = False
    ep_reward = 0
    found = False
    j = 0

    ep_reward = 0

    naive_policy = NaivePolicy(env.game)
    
    while not (terminated or truncated):
        # For naive policy
        action = naive_policy.get_action(env.game.robot)
        # # For a random policy, simply do:
        # action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        j += 1
        ep_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    # print(f"Episode {i}, reward={ep_reward}")
    ep_rewards.append(ep_reward)

avg_ep_reward = np.average(ep_rewards)
print(f"\nAverage Ep Reward: {avg_ep_reward}")


"""
EVALUATION RESULTS FOR THE NAIVE POLICY

Environment seed: 273
Episode 0, reward=1019.8476615087156
Episode 1, reward=684.5827964589414
Episode 2, reward=1289.7194629855132
Episode 3, reward=1529.172246627259
Episode 4, reward=393.80635196267775
Episode 5, reward=247.62458302923332
Episode 6, reward=4.272338300943375
Episode 7, reward=83.86603833436038
Episode 8, reward=680.0599331253051
Episode 9, reward=1116.2040974757172
Episode 10, reward=1444.2554294046543
Episode 11, reward=196.5757294684874
Episode 12, reward=0.0
Episode 13, reward=630.6875280023919
Episode 14, reward=709.8610707831892
Episode 15, reward=742.7538490648992
Episode 16, reward=580.0454975304409
Episode 17, reward=376.65975194278434
Episode 18, reward=57.060109561286815
Episode 19, reward=957.5620888175398
Episode 20, reward=1477.544276131335
Episode 21, reward=273.7105149345643
Episode 22, reward=13.702307317938125
Episode 23, reward=59.714240124580854
Episode 24, reward=417.06194610342163
Episode 25, reward=490.6651438367006
Episode 26, reward=1339.6104519464066
Episode 27, reward=321.2965095670574
Episode 28, reward=1375.5911109314134
Episode 29, reward=701.7808870449695
Episode 30, reward=1510.051195578049
Episode 31, reward=336.2316561923985
Episode 32, reward=986.9249253667396
Episode 33, reward=1833.2137323895238
Episode 34, reward=554.5380732477792
Episode 35, reward=337.9079676773943
Episode 36, reward=1289.2963610136237
Episode 37, reward=303.4945744174769
Episode 38, reward=376.8380263622138
Episode 39, reward=15.058676401774083
Episode 40, reward=478.13326495364555
Episode 41, reward=206.40478230337632
Episode 42, reward=1312.5678670646635
Episode 43, reward=51.48018225097151
Episode 44, reward=720.5309771834053
Episode 45, reward=336.2081400532119
Episode 46, reward=1191.8476708105645
Episode 47, reward=958.0469756720181
Episode 48, reward=692.9669174325934
Episode 49, reward=1709.2482549642193
Episode 50, reward=607.3284677915634
Episode 51, reward=2028.6946492082586
Episode 52, reward=1493.3534200315514
Episode 53, reward=1385.425190248992
Episode 54, reward=1646.6085050647052
Episode 55, reward=96.61304150838434
Episode 56, reward=1032.8207292310483
Episode 57, reward=649.6851515584982
Episode 58, reward=493.3513576993111
Episode 59, reward=1235.8472757932107
Episode 60, reward=650.6007679329001
Episode 61, reward=1321.0252764397346
Episode 62, reward=72.60356520028814
Episode 63, reward=1212.660014780278
Episode 64, reward=541.9752297256022
Episode 65, reward=642.5420627036369
Episode 66, reward=1427.579615722879
Episode 67, reward=1626.8556694835054
Episode 68, reward=603.3403974525021
Episode 69, reward=135.54928164443166
Episode 70, reward=291.6867150813047
Episode 71, reward=1802.7064629272872
Episode 72, reward=1181.7292263175339
Episode 73, reward=153.16523500359813
Episode 74, reward=1620.109764060334
Episode 75, reward=457.16653595609256
Episode 76, reward=336.9855210311101
Episode 77, reward=987.2237261114866
Episode 78, reward=1121.4434685156577
Episode 79, reward=129.6563786039274
Episode 80, reward=935.2953668093852
Episode 81, reward=936.3636415717408
Episode 82, reward=1642.384144146874
Episode 83, reward=287.38289062694486
Episode 84, reward=563.0570694254133
Episode 85, reward=1236.7149933192031
Episode 86, reward=339.733595715338
Episode 87, reward=819.5404166219068
Episode 88, reward=471.9111267916758
Episode 89, reward=45.43658446130299
Episode 90, reward=403.08797701719294
Episode 91, reward=135.7963374577021
Episode 92, reward=719.9201025987085
Episode 93, reward=1450.1512644944821
Episode 94, reward=629.4860574039731
Episode 95, reward=1747.9052809592845
Episode 96, reward=2280.216067671501
Episode 97, reward=1360.9163067630839
Episode 98, reward=601.4980199873378
Episode 99, reward=111.29830628665253

Average Ep Reward: 781.2071242661768
"""

"""
EVALUATION RESULTS FOR THE RANDOM POLICY

Environment seed: 55
Episode 0, reward=25.087394799293264
Episode 1, reward=895.8529594470796
Episode 2, reward=186.6187355476714
Episode 3, reward=215.19813401380307
Episode 4, reward=197.07130657869683
Episode 5, reward=546.7985730192962
Episode 6, reward=63.40614903742045
Episode 7, reward=65.7137817830023
Episode 8, reward=165.24619105368743
Episode 9, reward=171.10841041327043
Episode 10, reward=320.69112075054
Episode 11, reward=195.3975755673339
Episode 12, reward=642.281111266699
Episode 13, reward=453.051218997126
Episode 14, reward=208.7667989136342
Episode 15, reward=443.01652609241313
Episode 16, reward=792.7172959136919
Episode 17, reward=137.29842627399694
Episode 18, reward=345.1231598152675
Episode 19, reward=172.2877881404157
Episode 20, reward=360.18763931947035
Episode 21, reward=183.47374521407457
Episode 22, reward=296.4768663062954
Episode 23, reward=654.9939115425015
Episode 24, reward=189.19827164256208
Episode 25, reward=41.31197255551563
Episode 26, reward=0.0
Episode 27, reward=17.763904631137848
Episode 28, reward=0.0
Episode 29, reward=249.71679310740942
Episode 30, reward=710.7886226290724
Episode 31, reward=341.85097887766125
Episode 32, reward=347.75798064958633
Episode 33, reward=137.86530405279592
Episode 34, reward=32.37093697653876
Episode 35, reward=118.80383028302808
Episode 36, reward=160.50682280417044
Episode 37, reward=59.800523233019696
Episode 38, reward=423.68385423886014
Episode 39, reward=323.45570670800436
Episode 40, reward=502.4120150884652
Episode 41, reward=0.0
Episode 42, reward=103.43343761076962
Episode 43, reward=163.87628531319257
Episode 44, reward=0.0
Episode 45, reward=478.43574662727036
Episode 46, reward=212.97919037098364
Episode 47, reward=454.2078856427565
Episode 48, reward=183.79501778139706
Episode 49, reward=310.00069567658676
Episode 50, reward=74.17723872900407
Episode 51, reward=5.508764763673146
Episode 52, reward=351.08433097366725
Episode 53, reward=322.92756263491486
Episode 54, reward=243.25562767468801
Episode 55, reward=77.61652051834837
Episode 56, reward=429.8858119101611
Episode 57, reward=92.55926533423704
Episode 58, reward=376.643728084929
Episode 59, reward=211.64334582404715
Episode 60, reward=796.2347528038482
Episode 61, reward=83.32017454025801
Episode 62, reward=0.8664213120937347
Episode 63, reward=2.2334437370300257
Episode 64, reward=570.2610120628088
Episode 65, reward=389.32514893055134
Episode 66, reward=582.8467552387044
Episode 67, reward=10.200169086456299
Episode 68, reward=1035.4909674147727
Episode 69, reward=229.75215769389007
Episode 70, reward=384.30294611433817
Episode 71, reward=822.71614873936
Episode 72, reward=57.83918543940499
Episode 73, reward=18.177373468875885
Episode 74, reward=0.0
Episode 75, reward=775.1938651837025
Episode 76, reward=108.97120712368665
Episode 77, reward=0.0
Episode 78, reward=115.38453023096586
Episode 79, reward=176.12318328732516
Episode 80, reward=247.261125284614
Episode 81, reward=81.65469236119716
Episode 82, reward=0.0
Episode 83, reward=131.60895684333076
Episode 84, reward=737.1170367155818
Episode 85, reward=332.8439429116028
Episode 86, reward=2072.7999694341684
Episode 87, reward=389.2582100714032
Episode 88, reward=198.85502969556384
Episode 89, reward=0.0
Episode 90, reward=0.0
Episode 91, reward=265.3172424375178
Episode 92, reward=38.19144278764725
Episode 93, reward=0.0
Episode 94, reward=456.61285428485763
Episode 95, reward=533.9292906558726
Episode 96, reward=141.32994949315918
Episode 97, reward=0.0
Episode 98, reward=0.0
Episode 99, reward=195.03659017190833

Average Ep Reward: 274.6823856831163
"""