from common import *
from utils import *

file = open('../results/cup_random.pickle', 'rb')
result = pickle.load(file)[0]
autoencoder_mixture = result['autoencoder_mixture']
atg_mat = result['atg']
for goal_image_path in (glob.glob('../data/cup_random/cup_random_0/aspect_nodes/*')):
    print(goal_image_path)
    goal_image = transforms.functional.resize(Image.open(goal_image_path), (IMAGE_SIZE, IMAGE_SIZE))
    goal_image = to_var(transforms.functional.to_tensor(goal_image)).unsqueeze(0)

    recon_loss, recon_loss_norm = get_reconstruction_loss_with_all_ae(goal_image,
                                                     autoencoder_mixture,
                                                     loss_fn = torch.nn.functional.mse_loss)
    c_aspect_node_blief = belief_from_recon_loss(recon_loss)
    plt.bar(np.arange(len(c_aspect_node_blief)), c_aspect_node_blief)
    plt.show()
