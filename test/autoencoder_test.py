import sys, os
sys.path.append(os.path.abspath('../scripts/'))
from common import *
from utils import *
from data_loader import *
from random_transformer import *

dataset_paths = '../data/can_1000/can_1000_0/dataset/obs_'
#dataset_paths = '../data/real_aspects/Aspect-Raw'
sim_ds = ATGDataset(dataset_paths, image_size=IMAGE_SIZE)

sim_data_loader = DataLoader(sim_ds, batch_size=1, shuffle=False)
indices = [3, 6, 7, 18, 4]
autoencoder_mixture = {}
recon_ll = np.array([])

for index, image  in enumerate(sim_data_loader):
    if index > 0:
        recon_ll = get_recon_likelihood_with_all_ae(image, autoencoder_mixture)

        c_aspect_node_blief = belief_from_recon_ll(recon_ll)
        #plt.bar(np.arange(len(c_aspect_node_blief)), c_aspect_node_blief)
        #plt.show()
    if index == 0 or np.max(recon_ll) < 0.3:
        gen_images = generate_random_versions_of_image(image.squeeze(0), random_transformer, n_versions=N_VERSIONS)
        autoencoder_mixture[index] = {}
        autoencoder_mixture[index]['autoencoder'] = init_autoencoder()
        optimizer = optim.Adam(autoencoder_mixture[index]['autoencoder'].parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        ds = AutoEncoderDataset(gen_images, aspect_image=None)
        data_loader = DataLoader(ds, batch_size=4, shuffle=True)
        train_autoencoder(autoencoder_mixture[index]['autoencoder'],
                          optimizer,
                          criterion,
                          data_loader,
                          number_of_epochs=20,
                          name='autoencoder_' + str(index), verbose=VERBOSE)
        recon_images = autoencoder_mixture[index]['autoencoder'](to_var(gen_images)).cpu().data
        autoencoder_mixture[index]['stat'] = stat_of_mse_loss(recon_images, gen_images)

        rand_image = random_image('../data/can_1000/can_1000_0/dataset/')
        rand_image_recon = autoencoder_mixture[index]['autoencoder'](rand_image)

        test_image = to_var(gen_images[0]).unsqueeze(0)
        image_recon = autoencoder_mixture[index]['autoencoder'](test_image)

        plt.figure()
        imshow(make_grid(torch.stack([rand_image, rand_image_recon]).squeeze(1).cpu().data))
        plt.figure()
        imshow(make_grid(torch.stack([test_image.cpu().data, image_recon.cpu().data]).squeeze(1)), True)
        if index ==0:
            c_aspect_node = 0
        else:
            c_aspect_node = len(recon_ll)

    else:
        c_aspect_node = np.argmax(recon_ll)
        print('current observation matched to aspect node: ', c_aspect_node)
    #obs_save_dir_full = '../data/box_online_testing_1/box_online_testing_1_0/dataset/by_aspect/'+ str(c_aspect_node) + '/'
    obs_save_dir_full = '../data/can_1000/can_1000_0/dataset/by_aspect/'+ str(c_aspect_node) + '/'

    if not os.path.exists(obs_save_dir_full):
        os.makedirs(obs_save_dir_full)
    writing_path = 'obs_' + str(index) + '.jpg'
    write_loc = os.path.join(obs_save_dir_full, writing_path)

    cv2.imwrite(write_loc , tensor_to_cv(image.squeeze(0)))

with open('can_1000.pickle', 'wb') as handle:
    pickle.dump(autoencoder_mixture, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
    imshow(make_grid(gen_images), False)
    plt.figure()
    recon_images = autoencoder(to_var(gen_images)).cpu().data
    recon_loss = torch.nn.functional.mse_loss(recon_images, gen_images)
    imshow(make_grid(recon_images), False)
    rand_image = random_image('../data/can_data/can_data_0/dataset/')
    rand_image_recon = autoencoder(rand_image)
    rand_loss = torch.nn.functional.mse_loss(rand_image.cpu().data, rand_image_recon.cpu().data)

    image = image_with_index(dataset_paths, indices[index])
    image_recon = autoencoder(image)
    image_loss = torch.nn.functional.mse_loss(image.cpu().data, image_recon.cpu().data[0])

    print(recon_loss, rand_loss,  image_loss, 100*(rand_loss/recon_loss - 1), 100*(image_loss/recon_loss - 1))
    print(stat_of_mse_loss(recon_images, gen_images))
    print(image.shape)
    print(stat_of_mse_loss(image.cpu().data, image_recon.cpu().data))
    print(stat_of_mse_loss(rand_image.cpu().data, rand_image_recon[0].cpu().data))


    plt.figure()
    imshow(make_grid(torch.stack([rand_image, rand_image_recon]).squeeze(1).cpu().data))
    plt.figure()
    imshow(make_grid(torch.stack([image.cpu().data, image_recon.cpu().data]).squeeze(1)), True)
    '''
