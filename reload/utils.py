# ReLOaD
#
# evaluation.py
#
# Various visualisation utilities
#
# Alex Nichoson
# 21/07/2023

import matplotlib.pyplot as plt
import numpy as np

def show_training_graph(returns, num_iterations, eval_interval):
    """
    Use `matplotlib.pyplot` to chart how the policy improved during training.
    One iteration of `Cartpole-v0` consists of 200 time steps. The environment
    gives a reward of `+1` for each step the pole stays up, so the maximum return
    for one episode is 200. The charts shows the return increasing towards that
    maximum each time it is evaluated during training. (It may be a little
    unstable and not increase monotonically each time.)
    """

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    # plt.ylim(top=250)
    plt.savefig("training_graph.png")


def show_env_summary(env):
    """
    Prints Env Specs
    """

    print("Observation Spec:")
    print(env.time_step_spec().observation)

    print("Reward Spec:")
    print(env.time_step_spec().reward)

    print("Action Spec:")
    print(env.action_spec())

    time_step = env.reset()
    print("Time step:")
    print(time_step)

    action = np.array(1, dtype=np.int32)

    next_time_step = env.step(action)
    print("Next time step:")
    print(next_time_step)
