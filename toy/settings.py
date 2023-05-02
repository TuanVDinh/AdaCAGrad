from utils import*
import problem


def get_F_loss(choice):
    if choice == "1":
        F_loss = F_l[0]()
    elif choice == "2":
        F_loss = F_l[1]()
    else:
        raise ValueError("Invalid choice: {}".format(choice))
    return F_loss


def get_w_inits(choice):
    if choice == "1":
        init_pos1 = [-8.5, 7.5]
        init_pos2 = [-8.5, -5.]
        init_pos3 = [9., 9.]
        w_inits = [torch.Tensor(init_pos1), torch.Tensor(init_pos2), torch.Tensor(init_pos3), ]
    elif choice == "2":
        init_pos1 = [-8.5, 7.5]
        init_pos2 = [-8.5, -5.]
        init_pos3 = [9., 9.]
        w_inits = [torch.Tensor(init_pos1), torch.Tensor(init_pos2), torch.Tensor(init_pos3), ]
    else:
        raise ValueError("Invalid choice: {}".format(choice))
    return w_inits


def get_init_pos(choice):
    if choice == "1":
        init_pos1 = [-8.5, 7.5]
        init_pos2 = [-8.5, -5.]
        init_pos3 = [9., 9.]
    elif choice == "2":
        init_pos1 = [-8.5, 7.5]
        init_pos2 = [-8.5, -5.]
        init_pos3 = [9., 9.]
    else:
        raise ValueError("Invalid choice: {}".format(choice))
    return init_pos1, init_pos2, init_pos3


def get_pareto_bars(choice):
    if choice == "1":
        pareto_bar1 = [-7., 7.]
        pareto_bar2 = [-8.3552, -8.3552]
    elif choice == "2":
        pareto_bar1 = [-2., 3.]
        pareto_bar2 = [0.75, 0.75]
    else:
        raise ValueError("Invalid choice: {}".format(choice))
    return pareto_bar1, pareto_bar2


def get_average(choice):
    if choice == "1":
        average = [0, -8.3552]
    elif choice == "2":
        average = [1.33, 0.75]
    else:
        raise ValueError("Invalid choice: {}".format(choice))
    return average

F_l = [problem.Toy, problem.new_loss_func]

##########################
kappa = 0.99
sigma = 0.5
alpha = 0.001
num_iter = 100000

# kappa = 0.7
# sigma = 0.5
# alpha = 0.8
# num_iter = 500

choice = "1"

F_loss = get_F_loss(choice)
w_inits = get_w_inits(choice)
init_pos1, init_pos2, init_pos3 = get_init_pos(choice)
pareto_bar1, pareto_bar2 = get_pareto_bars(choice)
average = get_average(choice)




