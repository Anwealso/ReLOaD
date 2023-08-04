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

import reload.eval
import reload.utils

# import os
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
            print("Usage: test.py -p <policydir> ...\n\n"
                  "A script for evaluating saved tf-agents policies.\n\n"
                  "Arguments:\n"
                  "   -p <policy_dir>                           The directory that the trained policy is held in, e.g. saved_models/<modelname>/policy/\n"
                  "   --policy_dir=<policy_dir>                 The directory that the trained policy is held in, e.g. saved_models/<modelname>/policy/\n"
                  "   --num_eval_episodes=<num_eval_episodes>   Number of episodes to evaluate the policy over\n"
                  "   --visualize=<visualize>                   Whether to start with the visualisation shown (can be toggled during execution by pressing \"V\")\n"
                  "   --verbose=<verbose>                       Whether to print debug info to console\n"
                  "   --starting_budget=<starting_budget>       Starting budget for simulated episodes\n"
                  "   --num_targets=<num_targets>               Number of targets for the robot to identify in the simulator\n"
                  "   --player_fov=<player_fov>                 The FOV of the robot\n")
            sys.exit()
        elif opt in ("-p", "--policy_dir"):
            policy_dir = arg
        elif opt in ("--num_eval_episodes"):
            num_eval_episodes = int(arg)
        elif opt in ("--visualize"):
            visualize = bool(arg)
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
    import tensorflow as tf
    from tf_agents.environments import tf_py_environment
    from reload.simplesim.gym import SimpleSimGym

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
    print("\n============================================================\n"
          f"Finished testing. Average return = {avg_return}.")
