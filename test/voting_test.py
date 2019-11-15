import sys, os
sys.path.append(os.path.abspath('../scripts/'))
from common import *
from utils import *
from data_loader import *
from random_transformer import *
file = open('can_1000.pickle', 'rb')
autoencoder_mixture = pickle.load(file)
dataset_paths = '../data/can_1000/can_1000_0/dataset/obs_'

sim_ds = ATGDataset(dataset_paths, image_size=IMAGE_SIZE)

sim_data_loader = DataLoader(sim_ds, batch_size=1, shuffle=False)

for index, image  in enumerate(sim_data_loader):

    recon_ll = get_recon_likelihood_with_all_ae(image, autoencoder_mixture)

    c_aspect_node_blief = belief_from_recon_ll(recon_ll)
    plt.bar(np.arange(len(c_aspect_node_blief)), c_aspect_node_blief)
    plt.show()
