import torch
import matplotlib.pyplot as plt

# Load the learning rate trajectories for each initial point
lr_traj1 = torch.load("lr_traj0.pt")
lr_traj2 = torch.load("lr_traj1.pt")
lr_traj3 = torch.load("lr_traj2.pt")

# Update the default font size
plt.rcParams.update({'font.size': 15})

# # Plot the learning rate trajectories with different line styles and markers
plt.semilogy(lr_traj1.T, linestyle=":", color = 'b', label=r"$\theta$$_{\mathrm{init1}}$", linewidth=2)
plt.semilogy(lr_traj2.T, linestyle="--", color = 'r', label=r"$\theta$$_{\mathrm{init2}}$", linewidth=2)
plt.semilogy(lr_traj3.T, linestyle="-.", color = 'g', label=r"$\theta$$_{\mathrm{init3}}$", linewidth=2)


# Set tick direction to inward
plt.tick_params(axis='both', which='both', direction='in')
# Set dash style for grid
plt.grid(True, linestyle='-')

# Add labels and title to the plot
# plt.xlabel("Vòng lặp")
# plt.ylabel("Tỉ lệ học ($\\alpha$)")
plt.xlabel("Iteration")
plt.ylabel("Learning rate")

# plt.title("Tỉ lệ học đối với các điểm khởi tại khác nhau")

# Add legend to the plot
plt.legend()

name = f"./figures/lr_acagrad"
plt.tight_layout()
plt.savefig(f"{name}.pdf", bbox_inches='tight', dpi=1000)
plt.show()

