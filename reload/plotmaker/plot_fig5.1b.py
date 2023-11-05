"""
Presents the percentage of the episode that was spent observing each target
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
# Series 4
x = [0, 1, 2, 3, 4]
max_reward = np.full_like(x, 1656.6)


# --------------------------------- PLOT DATA -------------------------------- #

# Make plots
series1 = ax.errorbar(
    1,
    849.21,
    643.68,
    marker='o',
    markersize=8,
    capsize=4,
)
series2 = ax.errorbar(
    2,
    906.98,
    637.06,
    marker='o',
    markersize=8,
    capsize=4,
)
series3 = ax.errorbar(
    3,
    580.21,
    568.24,
    marker='o',
    markersize=8,
    capsize=4,
)
series4 = ax.plot(
    x,
    max_reward,
    c="C3"
)

# series5 = ax.plot(
#     4,
#     2000,
#     c="k"
# )

plt.legend(['Avg. Max. Available', 'ReLOaD (SAC)', 'Hand-crafted Policy', 'Random Policy'], loc ="upper right") 


# Add labels and a title.
ax.set_xlabel(
    ""
)
ax.set_ylabel(
    "Episode Average Reward", color="#333333"
)
ax.set_title(
    f"Average Eval Reward of ReLOaD vs Baselines", color="#333333"
)


# -------------------------------- FORMATTING -------------------------------- #
# Axis formatting.
ax.set_ylim(0, 2100)
ax.set_xlim(0, 4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#DDDDDD")
ax.tick_params(bottom=False, left=False)
# ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#EEEEEE")
ax.xaxis.grid(False)
plt.xticks([])

# fig.tight_layout()


# ----------------------------------- SAVE ----------------------------------- #
plt.savefig("fig_5.1b.pdf")