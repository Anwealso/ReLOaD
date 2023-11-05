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
sac_results = np.loadtxt('run-SAC_2-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_sac = sac_results[:, 2]
step = sac_results[:, 1]

# Series 2
naive_reward = np.full_like(step, 906.98)

# Series 3
random_reward = np.full_like(step, 580.21)

# Series 4
max_reward = np.full_like(step, 1656.6)


# --------------------------------- PLOT DATA -------------------------------- #
# Make plots
series1 = ax.plot(
    step,
    reward_sac
)
series2 = ax.plot(
    step,
    naive_reward
)
series3 = ax.plot(
    step,
    random_reward
)
series4 = ax.plot(
    step,
    max_reward
)

plt.legend(['ReLOaD (SAC)', 'Hand-crafted Policy', 'Random Policy', 'Avg. Max. Available'], loc ="lower right") 

# Add labels and a title.
ax.set_xlabel(
    "Training Iterations (Millions)", color="#333333"
)
ax.set_ylabel(
    "Episode Average Reward", color="#333333"
)
ax.set_title(
    f"Model Average Reward over Training vs Baselines", color="#333333"
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
plt.savefig("fig_5.1.png")