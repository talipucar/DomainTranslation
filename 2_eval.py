"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Evaluation of Domain Translation using Autoencoders.
"""
import os
from os.path import dirname, abspath
import imageio

import torch as th
import torch.utils.data

from src.model import AEModel
from src.model_dt import DTModel
from utils.load_data import Loader
from sklearn.preprocessing import StandardScaler
from utils.arguments import print_config_summary
from utils.arguments import get_arguments, get_config
from utils.utils import set_dirs

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import mlflow
torch.manual_seed(1)


def eval(data_loader, config):
    """
    :param IterableDataset data_loader: Pytorch data loader.
    :param dict config: Dictionary containing options.
    :return: None
    """
    # Print which dataset we are using
    print(f"{config['dataset']} is being used to test performance.")
    # Instantiate image Autoencoder model
    model_img = AEModel(config)
    # Load contrastive encoder
    model_img.load_models()
    # Instantiate image Autoencoder model
    model_rna = DTModel(config)
    # Load contrastive encoder
    model_rna.load_models()
    # Get Autoencoders for both modalities
    ae_img, ae_rna = model_img.autoencoder, model_rna.autoencoder_rna
    # Move the models to the device
    ae_img.to(config["device"])
    ae_rna.to(config["device"])
    # Set models to evaluation mode
    ae_img.eval()
    ae_rna.eval()
    # Evaluate Autoencoders
    with th.no_grad():
        evalulate_models(data_loader, ae_img, ae_rna,  config, plot_suffix="")



def evalulate_models(data_loader, ae_img, ae_rna, config, plot_suffix = "_Test"):
    # Load data. train=False is used to turn off transformations for training set.
    img_loader, rna_loader = data_loader

    output = map_source2target(ae_source=ae_rna, ae_target=ae_img, data_loader=rna_loader, config=config)
    inputs_rna, y_rna, h_rna, rna2img_recon, h_rna2img2lat, _ = output

    output = map_source2target(ae_source=ae_img, ae_target=ae_rna, data_loader=img_loader, config=config)
    inputs_img, y_img, h_img, img2rna_recon, h_img2rna2lat, image_recon = output


    h_img_features = np.concatenate([h_img, h_rna2img2lat])
    h_img_labels = np.concatenate([y_img, y_rna + 2 ])

    
    h_rna_features = np.concatenate([h_rna, h_img2rna2lat])
    h_rna_labels = np.concatenate([y_rna+2, y_img ])

    
    visualise_clusters(h_rna_features, h_rna_labels, plt_name="bothDomains_inRNALatentSpace" + plot_suffix)
    visualise_clusters(h_img_features, h_img_labels, plt_name="bothDomains_inImgLatentSpace" + plot_suffix)

    visualise_clusters(h_rna, y_rna, plt_name="rna_inLatentSpace" + plot_suffix)
    visualise_clusters(h_img, y_img, plt_name="img_inLatentSpace" + plot_suffix)

    visualise_clusters(h_rna2img2lat, y_rna, plt_name="translation_rna2img2latent_withRNALabels" + plot_suffix)
    visualise_clusters(h_img2rna2lat, y_img, plt_name="translation_img2rna2latent_withImgLabels" + plot_suffix)

    generate_image(inputs_img[0], name="image", suffix="input")
    generate_image(image_recon[0], name="image", suffix="recon")
    generate_image(rna2img_recon[0], name="rna2img", suffix="recon")


    print(20*"*"+"Classification test in latent space"+20*"*")
    linear_model_eval(h_img, y_img, h_rna2img2lat, y_rna, 
    description="Trained on latent of Image, tested on latent from: RNA encoder->Image decoder->Image encoder -> latent")
    
    linear_model_eval(h_rna, y_rna, h_img2rna2lat, y_img, 
    description="Trained on latent of RNA, tested on latent from: Image encoder->RNA decoder->RNA encoder -> latent")

    print(20*"*"+"Classification test in data space"+20*"*")
    linear_model_eval(inputs_rna, y_rna, img2rna_recon, y_img, 
    description="Trained on input RNA, tested on translations from: Image encoder->RNA decoder")

    # Reshape input images to be 2D so that they can be trained using Logistic regression
    inputs_img = inputs_img.reshape(y_img.shape[0], -1)
    # Reshape images translated from RNA domain to be 2D so that they can be evaluated using Logistic regression
    rna2img_recon = rna2img_recon.reshape(y_rna.shape[0], -1)
    # Train Logistic regression on original images, evaluate it on images translated from RNA data.
    linear_model_eval(inputs_img, y_img, rna2img_recon, y_rna, description="Trained on Image, tested on rna2img")


def map_source2target(ae_source, ae_target, data_loader, config):
    latent_l, labels_l, s2t2z_l, s2t_l, inputs_l, s_outputs_l = [], [], [], [], [], []
    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader

    for idx, samples in enumerate(train_loader):
        inputs = samples['tensor'].to(config['device'])
        labels = samples['binary_label']

        source_recon, source_latents, _, _ = ae_source(inputs)
        s2t_recon = ae_target.decoder(source_latents)
        _, source2target2lat, _, _ = ae_target(s2t_recon)

        s_outputs_l.append(source_recon.cpu().detach().numpy())
        latent_l.append(source_latents.cpu().detach().numpy())
        labels_l.append(labels)
        inputs_l.append(inputs.cpu().detach().numpy())
        s2t2z_l.append(source2target2lat.cpu().detach().numpy())
        s2t_l.append(s2t_recon.cpu().detach().numpy())


    latents = np.concatenate(latent_l)
    labels = np.concatenate(labels_l)
    s2t2z = np.concatenate(s2t2z_l)
    s2t = np.concatenate(s2t_l)
    inputs = np.concatenate(inputs_l)
    outputs = np.concatenate(s_outputs_l)

    return [inputs, labels, latents, s2t, s2t2z, outputs]


def linear_model_eval(X_train, y_train, X_test, y_test, use_scaler=False, description="Baseline: PCA + Logistic Reg."):
    # If true, scale data using scikit-learn scaler
    X_train,  X_test = scale_data(X_train, X_test) if use_scaler else X_train, X_test
    # Initialize Logistic regression
    clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
    # Fit model to the data
    clf.fit(X_train, y_train)
    # Summary of performance
    print(10*">"+description)
    print("Train score:", clf.score(X_train, y_train))
    print("Test score:", clf.score(X_test, y_test))



def generate_image(inputs, name="image", suffix="input"):
    img_dir = os.path.join("./results/evaluation/", "reconstructions")
    os.makedirs(img_dir, exist_ok=True)
    imageio.imwrite(os.path.join(img_dir, name + "_" + suffix + ".jpg"), np.uint8(inputs.reshape(64, 64) * 255))


def scale_data(Xtrain, Xtest):
    # Initialize scaler
    scaler = StandardScaler()
    # Fit and transform representations from training set
    Xtrain = scaler.fit_transform(Xtrain)
    # Transform representations from test set
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest


def visualise_clusters(embeddings, labels, plt_name = "test"):
    color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00',
                  "#006A5E", "#00BDA9", "#008DF9", "#FF66FD", "#00EFF9", "#A40122", "#FFC33B", "#00FB1D"]
    palette = {}
    for i in range(17):
        palette[str(i)] = color_list[i]

    y = labels.reshape(-1, )
    y = list(map(str, y.tolist()))
    img_n = 2

    fig, axs = plt.subplots(1, img_n, figsize=(10, 3.5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.2)

    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',
        left=False,  # both major and minor ticks are affected
        right=False,
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
    axs = axs.ravel() if img_n > 1 else [axs, axs]

    # Lower feature dimension to same as the one of the output dimension of contranstive encoder to be a fair comparison
    pca = PCA(n_components=2)
    # Fit training data and transform
    embeddings_pca = pca.fit_transform(embeddings)
    axs[0].title.set_text('Embeddings from PCA')
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=axs[0], palette=palette, hue=y, s=10)
    
    embeddings_tsne = tsne(embeddings)
    axs[1].title.set_text('Embeddings from t-SNE')
    sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=axs[1], palette=palette, hue=y, s=10)
    # plt.setp(axs[i].get_legend().get_texts(), fontsize='4')
    root_path = dirname(abspath(__file__))
    fig_path = os.path.join(root_path, "results", "evaluation", "clusters", plt_name + ".png")
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.savefig(fig_path, bbox_inches="tight")
    plt.clf()


def tsne(latent):
    mds = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return mds.fit_transform(latent)


def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for imaging dataset.
    img_loader = Loader(config, dataset_name="NucleiDataset")
    # Get data loader for RNA dataset.
    rna_loader = Loader(config, dataset_name="RNADataset")
    # Start training
    eval([img_loader, rna_loader], config)


if __name__ == "__main__":
    # Get parser / command line arguments
    args = get_arguments()
    # Get configuration file
    config = get_config(args)
    # Summarize config and arguments on the screen as a sanity check
    print_config_summary(config, args)
    # --If True, start of MLFlow for experiment tracking:
    if config["mlflow"]:
        # Experiment name
        mlflow.set_experiment(experiment_name=config["model_mode"]+"_"+str(args.experiment))
        # Start a new mlflow run
        with mlflow.start_run():
            # Run the main
            main(config)
    else:
        # Run the main
        main(config)
