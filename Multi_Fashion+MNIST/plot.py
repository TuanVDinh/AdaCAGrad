import wandb
import numpy as np
#run = wandb.init(project="CAGrad_multi_fashion_and_mnist",name = "CAGrad")
api = wandb.Api()
run = api.run("tuantran23012000/CAGrad_multi_fashion_and_mnist/w1j3uafr")
his = run.scan_history()
losses = [row["Learning rate"] for row in his]
losses = [0.05] + losses
np.save('LR.npy', np.array(losses))    # .npy extension is added if not given
d = np.load('LR.npy')
print(d)