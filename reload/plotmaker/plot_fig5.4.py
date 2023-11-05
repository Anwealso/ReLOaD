import matplotlib.pyplot as plt
import numpy as np

# Create figure and customise formatting
font_family = "Helvetica"
plt.ion()
plt.rc("font", family=font_family)
fig = plt.figure()

rows = 1
cols = 2
fig, ax = plt.subplots(rows, cols)
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

# --------------------------------- LOAD DATA -------------------------------- #
# Series 1
sac_results = np.loadtxt('performance_across_numtargets_sac.csv', delimiter=',', skiprows=1)

x = sac_results[:, 0]
reward_sac = sac_results[:, 1]
std_dev_sac = sac_results[:, 2]

# Series 2
hc_results = np.loadtxt('performance_across_numtargets_handcrafted.csv', delimiter=',', skiprows=1)

# num_targets = sac_results[:, 0]
reward_hc = hc_results[:, 1]
std_dev_hc = hc_results[:, 2]



# --------------------------------- PLOT DATA -------------------------------- #

# Make plots
series1 = ax[0].errorbar(
    x,
    reward_sac,
    std_dev_sac,
    marker='o',
    capsize=5,
)
ax[0].legend(['ReLOaD'], loc ="upper right") 

series2 = ax[1].errorbar(
    x,
    reward_hc,
    std_dev_hc,
    marker='o',
    capsize=5,
    color="orange",
    ecolor="orange",
)
ax[1].legend(['Hand-crafted Policy'], loc ="upper right") 


fig.suptitle(
    f"Effect of Number of Targets on Performance", color="#333333"
)

for axis in ax:

    # Add labels and a title.
    axis.set_xlabel(
        "Number of Targets", color="#333333"
    )
    axis.set_ylabel(
        "Episode Average Reward", color="#333333"
    )
    # axis.set_title(
    #     f"Model Average Reward over Training vs Baselines", color="#333333"
    # )


    # -------------------------------- FORMATTING -------------------------------- #

    # Axis formatting.
    axis.set_ylim(0, 2100)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#DDDDDD")
    axis.tick_params(bottom=False, left=False)
    # axis.set_axisbelow(True)
    axis.yaxis.grid(True, color="#EEEEEE")
    axis.xaxis.grid(False)

    fig.tight_layout()


# ----------------------------------- SAVE ----------------------------------- #
plt.savefig("fig_5.4.png")