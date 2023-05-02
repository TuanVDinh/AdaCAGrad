# from tqdm import tqdm
import matplotlib as plt
from utils import plot_results
from settings import F_loss
plt.rc('font', family='serif')
# ### Define the problem ###
F = F_loss

plot_results(F_loss)
