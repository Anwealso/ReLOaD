"""
This figure shows the training reward of our model
"""

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
run1_results = np.loadtxt('run-SAC_2-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
run1_reward = run1_results[:, 2]
step1 = run1_results[:, 1]

# Series 2
run2_results = np.loadtxt('run-SAC_3-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
run2_reward = run2_results[:, 2]
step2 = run2_results[:, 1]

# Series 3
run3_results = np.loadtxt('run-SAC_4-tag-rollout_ep_rew_mean.csv', delimiter=',', skiprows=1)
run3_reward = run3_results[:, 2]
step3 = run3_results[:, 1]

# Series 4
max_reward = np.full_like(step3, 1656.6)


# --------------------------------- PLOT DATA -------------------------------- #
# Make plots
series1 = ax.plot(
    step1,
    run1_reward,
    linewidth=1,
    color=(30/255,119/255,180/255,1)
)
series2 = ax.plot(
    step2,
    run2_reward,
    linewidth=1,
    color=(30/255,60/255,180/255,1)
)
series3 = ax.plot(
    step3,
    run3_reward,
    linewidth=1,
    color=(30/255,180/255,153/255,1)

)
series4 = ax.plot(
    step3,
    max_reward,
    c="C3"
)

plt.legend(['Run 1', 'Run 2', 'Run 3', 'Avg. Max. Available'], loc ="lower right") 

# Add labels and a title.
ax.set_xlabel(
    "Training Iterations (Millions)", color="#333333"
)
ax.set_ylabel(
    "Episode Average Reward", color="#333333"
)
ax.set_title(
    f"ReLOaD Average Reward over Training", color="#333333"
)


# -------------------------------- FORMATTING -------------------------------- #
# Axis formatting.
ax.set_ylim(0, 2100)
ax.set_xlim(0, 6_000_000)
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
plt.savefig("fig_5.1.pdf")