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
relative_results = np.loadtxt('run-SAC_2-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_relative = relative_results[:, 2]
step = relative_results[:, 1]

# Series 2
absolute_results = np.loadtxt('run-SAC_5-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
reward_absolute = absolute_results[:, 2]
step = absolute_results[:, 1]


# Series 4
max_reward = np.full_like(step, 1656.6)


# --------------------------------- PLOT DATA -------------------------------- #
# Make plots
series1 = ax.plot(
    step,
    reward_relative
)
series2 = ax.plot(
    step,
    reward_absolute
)
series4 = ax.plot(
    step,
    max_reward,
    c="C3"
)


plt.legend(['Relative Positions', 'Absolute Positions', 'Avg. Max. Available'], loc ="upper right") 

# Add labels and a title.
ax.set_xlabel(
    "Training Iterations (Millions)", color="#333333"
)
ax.set_ylabel(
    "Episode Average Reward", color="#333333"
)
ax.set_title(
    f"Comparison of Usage of Relative vs Absolute Target Positions in State Space", color="#333333"
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
plt.savefig("fig_5.9.pdf")