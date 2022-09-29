import matplotlib.pyplot as plt
import numpy as np

# Plot Configuration
plt.rcParams["figure.figsize"] = (3.25, 6)
plt.rcParams["font.size"] = 10

perovskite = np.loadtxt("perovskiteVolume/tpot/generations.dat")
bandgap = np.loadtxt("bandgap2DMaterials/tpot/generations.dat")
exfoliation = np.loadtxt("exfoliation2DMaterials/tpot/generations.dat")

fig, [ax1, ax2, ax3] = plt.subplots(
    nrows=3,
    ncols=1,
    sharex=True,
)
fig.subplots_adjust(
    top=0.99, bottom=0.07, left=0.20, right=0.95, hspace=0.10
)

ax1.set_xlim([0, 125.0])
ax1.set_xticks(range(0,126,25))
ax1.set_xticks(range(0,126,5), minor=True)

ax1.set_ylim([3.5, 5.0])
ax2.set_ylim([0.4, 0.44])
ax3.set_ylim([0.39, 0.43])

ax3.set_xlabel("Generation")

ax1.set_ylabel("RMSE (Å$^3$)")
ax2.set_ylabel("CV RMSE (eV)")
ax3.set_ylabel("CV RMSE (J / m$^2$)")

ax1.plot(perovskite[:, 0], -1*perovskite[:, 1], 'k-', lw=1.50)
ax2.plot(bandgap[:, 0], -1*bandgap[:, 1], 'k-', lw=1.50)
ax3.plot(exfoliation[:, 0], -1*exfoliation[:, 1], 'k-', lw=1.50)

for aa, ax in enumerate([ax1, ax2, ax3]):
    ax.tick_params(direction="in", which="both", right=True, top=True)

ax1.text(120, 3.5 + 0.85 * 1.5, "Perovskite Volume", ha="right")
ax2.text(120, 0.4 + 0.85 * 0.04, "Bandgap", ha="right")
ax3.text(120, 0.39 + 0.85 * 0.04, "Exfoliation Energy", ha="right")

fig.savefig("convergence.pdf")
