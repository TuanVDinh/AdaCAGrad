from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import settings


def plot_results(F):
    plot3d(F)
    plot_contour(F, 1, name="./figures/contour_task_1")
    plot_contour(F, 2, name="./figures/contour_task_2")
    t1 = torch.load(f"theta0.pt")
    t2 = torch.load(f"theta1.pt")
    t3 = torch.load(f"theta2.pt")

    length = t1["acagrad"].shape[0]

    # for method in ["sgd", "mgd", "pcgrad", "cagrad", "acagrad"]:
    for method in ["acagrad"]:
        ranges = list(range(10, length, 1000))
        ranges.append(length - 1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0,  # task == 0 meeas plot for both tasks
                         traj=[t1[method][:t], t2[method][:t], t3[method][:t]],
                         plotbar=(method == "acagrad"),
                         name=f"./figures/contour_{method}_{t}")

################################################################################
#
# Plot Utils
#
################################################################################

def plot3d(F, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(1).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.cividis)
    print(Ys.mean(1).min(), Ys.mean(1).max())

    ax.set_zticks([-2, 0, 8, 16])
    ax.set_zlim(-20, 10)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    ax.view_init(25)
    plt.xlabel(r'$\theta_1$', fontsize=15)
    plt.ylabel(r'$\theta_2$', fontsize=15)
    plt.tight_layout()
    # plt.savefig(f"3d-obj.pdf", bbox_inches='tight', dpi=1000)
    plt.savefig(f"./figures/3d-obj.png", bbox_inches='tight', dpi=1000)

def plot_contour(F, task=1, traj=None, xl=12, plotbar=False, name="tmp"):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    cmap = cm.get_cmap('cividis')

    # yy = 0.75 # for new_loss_func
    # yy = -8.3552

    init_pos1, init_pos2, init_pos3 = settings.init_pos1, settings.init_pos2, settings.init_pos3
    pareto_bar1, pareto_bar2 = settings.pareto_bar1, settings.pareto_bar2
    average = settings.average

    if task == 0:
        Yv = Ys.mean(1)


        plt.plot(init_pos1[0], init_pos1[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot(init_pos2[0], init_pos2[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot(init_pos3[0], init_pos3[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot(pareto_bar1, pareto_bar2, linewidth=8.0, zorder=0, color='pink')
        plt.plot(average[0], average[1], marker='*', markersize=15, zorder=5, color='k')
        # plt.annotate("$L_0^*$", xy=(0, 2),xytext=(0, 3), fontsize=20)
        # plt.annotate("$L_0^*$", xy=(2, 2),xytext=(0, 3), fontsize=20)

        # plt.plot(-2.73, 0.6, marker='o', markersize=10, zorder=5, color='k')
        # plt.plot(3.5, -1.14, marker='o', markersize=10, zorder=5, color='k')
        # plt.plot(1.14, 4.37, marker='o', markersize=10, zorder=5, color='k')
        # plt.plot([-2., 3.], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        # plt.plot(1.33, yy, marker='*', markersize=15, zorder=5, color='k')

    elif task == 1:
        Yv = Ys[:,0]
        # plt.plot(-2, yy, marker='*', markersize=15, zorder=5, color='k')
        plt.plot(pareto_bar1[1], pareto_bar2[0], marker='*', markersize=15, zorder=5, color='k')
        # plt.annotate("$L_0^*$", xy=(2, 2),xytext=(1, -6), fontsize=20)
    else:
        Yv = Ys[:,1]
        # plt.plot(3, 0, marker='*', markersize=15, zorder=5, color='k')
        plt.plot(pareto_bar1[0], pareto_bar2[0], marker='*', markersize=15, zorder=5, color='k')
        # plt.annotate("$L_0^*$", xy=(2, 2),xytext=(1, -6), fontsize=20)

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.cividis, linewidths=4.0)

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            #color_list[:,2] = 1-np.linspace(0, 1, l)
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        # cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar = fig.colorbar(c)
        cbar.ax.tick_params(labelsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.xlabel(r'$\theta_1$', fontsize=15)
    plt.ylabel(r'$\theta_2$', fontsize=15)
    plt.tight_layout()
    # plt.savefig(f"{name}.pdf", bbox_inches='tight', dpi=100)
    plt.savefig(f"{name}.png", bbox_inches='tight', dpi=100)
    plt.close()

def smooth(x, n=20):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 3:
            y.append(x[i])
        else:
            y.append(v.mean())
    return y

################################################################################
#
# Multi-Objective Optimization Solver
#
################################################################################

def mean_grad(grads):
    return grads.mean(1)

def pcgrad(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1-g12/g11)*g1+(1-g12/g22)*g2)/2
    else:
        return (g1+g2)/2

def mgd(grads):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    if g12 < min(g11, g22):
        x = (g22-g12) / (g11+g22-2*g12 + 1e-8)
    elif g11 < g22:
        x = 1
    else:
        x = 0

    g_mgd = x * g1 + (1-x) * g2 # mgd gradient g_mgd
    return g_mgd

def cagrad(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)

def acagrad(grads, c=0.5):
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1 + c), x


maps = {
    "sgd": mean_grad,
    "cagrad": cagrad,
    "acagrad": acagrad,
    "mgd": mgd,
    "pcgrad": pcgrad,
}

