import matplotlib.pyplot as plt
import numpy as np


array = np.loadtxt('sac_training_reward3.csv', delimiter=',', skiprows=1)

reward = array[:, 2]
step = array[:, 1]
max_reward = np.full_like(array[:, 1], 2000)

# # print(array)
# print()
# print(reward[0])
# print(step[0])
# print(max_reward[0])
# quit()

font_family = "Helvetica"


x = step
# y = reward

# Create figure and customise formatting
plt.ion()
plt.rc("font", family=font_family)

fig = plt.figure()
# fig.suptitle(
#     f"Target Confidence Distributions", color="#333333"
# )
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

# Make plots
series1 = ax.plot(
    step,
    reward
)
series2 = ax.plot(
    step,
    max_reward
)

plt.legend(['SAC Training Reward', 'Max. Available Reward'], loc ="lower right") 
# plt.legend(['SAC Training Reward', 'Max. Available Reward']) 

# ax.set_xticklabels(x, rotation=90, ha="right")

# Axis formatting.
ax.set_ylim(0, 2100)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.spines["bottom"].set_color("#DDDDDD")
ax.tick_params(bottom=False, left=False)
# ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#EEEEEE")
ax.xaxis.grid(False)

# Add labels and a title.
ax.set_xlabel(
    "Training Iterations (Millions)", color="#333333"
)
ax.set_ylabel(
    "Average Reward", color="#333333"
)
ax.set_title(
    f"Episode Average Reward over Training", color="#333333"
)


# fig.tight_layout()

plt.savefig("training_reward.png")
