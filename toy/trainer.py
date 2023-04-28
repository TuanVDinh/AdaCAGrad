from settings import *
from numpy import linalg as LA
import os

### Create a folder named 'figures' to save figures if it does not exist.
folder_name = "figures"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
###

### Define the problem ###
F = F_loss

def run_all():
    all_traj = {}

    # the initial positions
    inits = theta_inits

    for i, init in enumerate(inits):
        lr_traj = []
        # for m in tqdm(["sgd", "mgd", "pcgrad", "cagrad", "acagrad"]):
        for m in tqdm(["acagrad"]):
            all_traj[m] = None
            traj = []
            solver = maps[m]
            x = init.clone()
            x.requires_grad = True

            # n_iter = 100000
            n_iter = num_iter
            opt = torch.optim.Adam([x], lr=alpha)
            lr_decreased = False  # Flag to track whether the learning rate has decreased

            for it in range(n_iter):
                traj.append(x.detach().numpy().copy())
                f_prev, grads = F(x, True)
                if m == "acagrad":
                    g, w = solver(grads, c=0.5)
                    lr_traj.append(opt.param_groups[0]['lr'])
                elif m == "cagrad":
                    g = solver(grads, c=0.5)
                else:
                    g = solver(grads)
                opt.zero_grad()
                x.grad = g
                opt.step()
                f_curr, _ = F(x, True)
                imp_fact = - sigma * opt.param_groups[0]['lr'] * LA.norm(g)**2
                if m == "acagrad":
                    if ((w * f_curr[0] + (1 - w) * f_curr[1]) <=
                            (w * f_prev[0] + (1 - w) * f_prev[1])
                            + imp_fact):
                        opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']
                        lr_decreased = False  # Reset the flag if the learning rate remains the same
                    else:
                        if not lr_decreased:  # Check if the learning rate has already decreased
                            opt.param_groups[0]['lr'] = opt.param_groups[0]['lr'] * kappa
                            lr_decreased = True  # Set the flag if the learning rate decreases
                else:
                    opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']

            # all_traj[m] = torch.tensor(traj)
            all_traj[m] = torch.from_numpy(np.array(traj))
            torch.save(all_traj, f"theta{i}.pt")

        lr_traj = torch.tensor(lr_traj)
        torch.save(lr_traj, f"lr_traj{i}.pt")  # changed filename to include index


if __name__ == "__main__":
    run_all()