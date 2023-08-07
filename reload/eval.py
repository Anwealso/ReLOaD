# ReLOaD
#
# eval.py
#
# Holds all the helper functions necessary to evaluate rl models
#
# Alex Nichoson
# 21/07/2023


def compute_avg_return(eval_tf_env, policy, num_episodes=5):
    """
    Computes the average return of a policy per episode, given the policy, environment, and
    a number of episodes.

    The most common metric used to evaluate a policy is the average return.
    The return is the sum of rewards obtained while running a policy in an
    environment for an episode. Several episodes are run, creating an average
    return.

    See also the metrics module for standard implementations of different metrics.
    https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    Running this computation on the `random_policy` shows a baseline performance
    in an environment.
    """

    print("Evaluating average return...", end="")
    total_return = 0.0
    for i in range(num_episodes):
        print(f" {i}", end="", flush=True)
        time_step = eval_tf_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = eval_tf_env.step(action_step.action)
            episode_return += time_step.reward

        total_return += episode_return
        # print(f"total_return: {total_return}")
    


    avg_return = total_return / num_episodes
    print(f"\nFinished evaluating. Avg Return: {avg_return}\n")
    # print(f"avg_return.numpy()[0]: {avg_return.numpy()[0]}")
    # quit()
    return avg_return.numpy()[0]


def run_episodes_and_create_video(policy, eval_tf_env, eval_py_env):
    """
    Runs a number of example episodes with an agent and saves a video of it to file.
    """
    # num_episodes = 3
    # frames = []
    # for _ in range(num_episodes):
    #     time_step = eval_tf_env.reset()
    #     frames.append(eval_py_env.render())
    #     while not time_step.is_last():
    #         action_step = policy.action(time_step)
    #         time_step = eval_tf_env.step(action_step.action)
    #         frames.append(eval_py_env.render())

    # gif_file = io.BytesIO()
    # imageio.mimsave(gif_file, frames, format='gif', fps=60)
    pass

