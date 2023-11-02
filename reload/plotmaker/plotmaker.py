import matplotlib.pyplot as plt
import numpy as np

# Create figure and customise formatting
font_family = "Helvetica"
plt.ion()
plt.rc("font", family=font_family)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes


# --------------------------------- LOAD DATA -------------------------------- #
# Series 1
sac_results = np.loadtxt('run-SAC_0-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_sac = sac_results[:, 2]
step_sac = sac_results[:, 1]
print(np.max(reward_sac))

# Series 2
ppo_results = np.loadtxt('run-PPO_0-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_ppo = ppo_results[:, 2]
step_ppo = ppo_results[:, 1]
print(np.max(reward_ppo))

# Series 3
dqn_results = np.loadtxt('run-DQN_0-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_dqn = dqn_results[:, 2]
step_dqn = dqn_results[:, 1]
print(np.max(reward_dqn))
quit()

# Series 4
max_reward = np.full_like(step_sac, 2000)


# --------------------------------- PLOT DATA -------------------------------- #
# Make plots
series1 = ax.plot(
    step_sac,
    reward_sac
)
series2 = ax.plot(
    step_ppo,
    reward_ppo
)
series3 = ax.plot(
    step_dqn,
    reward_dqn
)
series4 = ax.plot(
    step_sac,
    max_reward
)

plt.legend(['SAC', 'PPO', 'DQN', 'Avg. Max. Available'], loc ="lower right") 

# Add labels and a title.
ax.set_xlabel(
    "Training Iterations (Millions)", color="#333333"
)
ax.set_ylabel(
    "Episode Average Reward", color="#333333"
)
ax.set_title(
    f"Model Average Reward over Training", color="#333333"
)


# -------------------------------- FORMATTING -------------------------------- #
# Axis formatting.
ax.set_ylim(0, 2100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#DDDDDD")
ax.tick_params(bottom=False, left=False)
# ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#EEEEEE")
ax.xaxis.grid(False)

# fig.tight_layout()


# ----------------------------------- SAVE ----------------------------------- #
plt.savefig("plot.png")
