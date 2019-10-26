from common import *

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()

	return Variable(x, volatile=volatile)

def imshow(img, display=False):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('autoencoder_output.png')
    if display:
        plt.show()
def generate_random_versions_of_image(image, transformer, n_versions=10):
    output = []
    for i in range(n_versions):
        output.append(transformer(image))

    return torch.stack(output)
def get_reconstruction_loss_with_all_ae(image, autoencoder_mixture, loss_fn):
    recon_loss_mix = []
    recon_loss_mix_normalized = []

    for aspect, aspect_param in autoencoder_mixture.items():
        image = to_var(image)
        recon_image = aspect_param['autoencoder'](image)
        recon_loss  = loss_fn(recon_image, image).data.sum()
        recon_loss_mix.append(recon_loss)
        recon_loss_mix_normalized.append(abs(recon_loss - aspect_param['recon_error']))
    return np.array(recon_loss_mix), np.array(recon_loss_mix_normalized)
def get_mixure_output(autoencoder_mixture, images, n_clusters=10):
    output = []
    for cluster in range(n_clusters):
        output.append(autoencoder_mixture[cluster]['autoencoder'](images))
    return output

def belief_for_observation(image, autoencoder_mixture, loss_fn):
    belief = 1./get_reconstruction_loss_with_all_ae(image, autoencoder_mixture, loss_fn)[0]
    belief /= belief.sum()
    return belief
def train_autoencoder(autoencoder, optimizer, criterion, data_loader, number_of_epochs=1, name='main', verbose=False):
    print('Training %s ...'%(name))
    for epoch in range(number_of_epochs):

        running_loss = 0.0
        autoencoder.train()
        for batch_index, (in_images, aspect_image) in enumerate(data_loader):

            in_images = to_var(in_images)
            out_images = autoencoder(in_images)

            loss = criterion(out_images, aspect_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy()
            if batch_index % 100==0 and verbose:
                print('epoch %d loss: %.5f' % (epoch, running_loss/((batch_index + 1))))
            if batch_index != 0 and batch_index % 1000 == 0:
                break
    print('Done training %s'%(name))
