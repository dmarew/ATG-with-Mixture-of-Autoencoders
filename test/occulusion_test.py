import sys, os
sys.path.append(os.path.abspath('../scripts/'))
from common import *
from utils import *
from data_loader import *
from random_transformer import *
file = open('cup.pickle', 'rb')
autoencoder_mixture = pickle.load(file)
image = Image.open('test_image.jpg')
image = pil_to_tensor(image).unsqueeze(0)
belief = belief_for_observation(image, autoencoder_mixture)
print(belief)
plt.bar(np.arange(len(belief)), belief)
plt.title('before occulusion')
image = Image.open('test_image_occ_4.jpg')
image = pil_to_tensor(image).unsqueeze(0)
belief = belief_for_observation(image, autoencoder_mixture)
recon_ll = get_recon_likelihood_with_all_ae(image, autoencoder_mixture)
print(100*recon_ll)
print(belief)
plt.figure()
plt.bar(np.arange(len(belief)), belief)
plt.title('after occulusion')
plt.show()
