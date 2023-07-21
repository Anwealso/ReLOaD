# ReLOaD
#
# run_agent_simplesim.py
#
# Runs a dqn policy over multiple episodes of the simplesim env
#
# Alex Nichoson
# 19/07/2023


# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #

from reload.simplesim.gym import SimpleSimGym
import reload.eval
import reload.utils

import tensorflow as tf
import os
from tf_agents.environments import tf_py_environment
import sys
import getopt

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # ----------------------------- Command Line Args ---------------------------- #

    # Default values
    num_eval_episodes = 10  # @param {type:"integer"}
    visualize = False
    verbose = False
    STARTING_BUDGET = 400
    NUM_TARGETS = 1
    PLAYER_FOV = 60
    policy_dir = ""

    # Get command line values
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hp:", ["policy_dir=", "num_eval_episodes=", "visualize=", "verbose=", "starting_budget=", "num_targets=", "player_fov="])
    except getopt.GetoptError:
        print ('Usage: test.py -p <policydir> ...')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("Usage: test.py -p <policydir> ...")
            sys.exit()
        elif opt in ("-p", "--policy_dir"):
            policy_dir = arg
        elif opt in ("--num_eval_episodes"):
            num_eval_episodes = int(arg)
        elif opt in ("--visualize"):
            visualizee = bool(arg)
        elif opt in ("--verbose"):
            verbose = bool(arg)
        elif opt in ("--starting_budget"):
            starting_budget = int(arg)
        elif opt in ("--num_targets"):
            num_targetse = int(arg)
        elif opt in ("--player_fov"):
            player_fov = int(arg)

    # Error checking
    if policy_dir == "":
        print ('Argument policy_dir invalid or not provided')
        print ('Usage: test.py -p <policydir> ...')
        sys.exit(3)

    # -------------------------------- Environment ------------------------------- #

    # Instantiate the evaluation environment.
    eval_py_env = SimpleSimGym(
        STARTING_BUDGET, NUM_TARGETS, PLAYER_FOV, visualize=visualize
    )
    eval_env = tf_py_environment.TFPyEnvironment(
        eval_py_env,
        check_dims=True,
    )
    # View Env Specs
    if verbose:
        reload.utils.show_env_summary(eval_py_env)

    # ----------------------------- Load Saved Policy ---------------------------- #
    policy = tf.saved_model.load(policy_dir)

    # --------------------------- Evaluate Performance --------------------------- #
    avg_return = reload.eval.compute_avg_return(
        eval_env, policy, num_episodes=num_eval_episodes
    )
    print(f"Finished testing. Average return = {avg_return}.")
