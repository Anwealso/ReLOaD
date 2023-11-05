"""
Presents the fraction of observations that each target was a part of
"""

import matplotlib.pyplot as plt
import numpy as np

# Create figure and customise formatting
font_family = "Helvetica"
size = "22"
plt.ion()
plt.rc("font", family=font_family, size=size)


rows = 1
cols = 2
fig, ax = plt.subplots(rows, cols, figsize=(20,10))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes

# --------------------------------- LOAD DATA -------------------------------- #
# Series 1
sac_results = np.loadtxt('obs_distribution_sac.csv', delimiter=',', skiprows=1)

num_targets_sac = sac_results[:, 0]
target_obs_fraction_of_total_obs_sac = sac_results[:, 5]

fraction_of_total_obs_by_num_targets_sac = [[], [], [], [], []]
for i in range(0, np.shape(target_obs_fraction_of_total_obs_sac)[0]):
    # Append this obs_fraction to the list of obs_fractions fo this number of targets
    fraction_of_total_obs_by_num_targets_sac[int(num_targets_sac[i])-1].append(target_obs_fraction_of_total_obs_sac[i])



# Series 2
hc_results = np.loadtxt('obs_distribution_hc.csv', delimiter=',', skiprows=1)

num_targets_hc = hc_results[:, 0]
target_obs_fraction_of_total_obs_hc = hc_results[:, 5]

fraction_of_total_obs_by_num_targets_hc = [[], [], [], [], []]
for i in range(0, np.shape(target_obs_fraction_of_total_obs_hc)[0]):
    # Append this obs_fraction to the list of obs_fractions fo this number of targets
    fraction_of_total_obs_by_num_targets_hc[int(num_targets_hc[i])-1].append(target_obs_fraction_of_total_obs_hc[i])



# --------------------------------- PLOT DATA -------------------------------- #


# Make plots
series1 = ax[0].boxplot(
    fraction_of_total_obs_by_num_targets_sac,
    sym=""
)
for i in range(5):
    y = fraction_of_total_obs_by_num_targets_sac[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    ax[0].plot(x, y, '.', alpha=0.2, c='C0')
# series1 = ax[0].boxplot(
#     num_targets_sac,
#     obs_fraction_sac,
#     # marker='.',
# )
# ax[0].legend(['ReLOaD'], loc ="upper right") 


series2 = ax[1].boxplot(
    fraction_of_total_obs_by_num_targets_hc,
    sym="",
)
for i in range(5):
    y = fraction_of_total_obs_by_num_targets_hc[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    ax[1].plot(x, y, '.', alpha=0.2, c="orange")
# series2 = ax[1].boxplot(
#     num_targets_hc,
#     obs_fraction_hc,
#     # marker='.',
#     color="orange",
# )
# ax[1].legend(['Hand-crafted Policy'], loc ="upper right") 


fig.suptitle(
    f"Distribution of Observations Amongst Targets", color="#333333"
)

ax[0].set_title(
    f"ReLOaD", color="#333333"
)
ax[1].set_title(
    f"Hand-crafted Policy", color="#333333"
)

for axis in ax:
    # Add labels and a title.
    axis.set_xlabel(
        "Number of Targets", color="#333333"
    )
    axis.set_ylabel(
        "Fraction of Observations that a Target was a Part of", color="#333333"
    )


    # -------------------------------- FORMATTING -------------------------------- #

    # Axis formatting.
    # axis.set_ylim(0, 1)
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
plt.savefig("fig_5.5_v3.pdf")