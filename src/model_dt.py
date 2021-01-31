"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Class to train an Autoencoder using the RNA sequence data and to align the latent representations
of two modalities (Chromatin Images and RNA-seq data).

The pre-trained CNN-based AE is loaded, and a fully-connected AE is initialized together with a discriminator, or
classifier. The fully-connected AE is trained using either adversarial, or cross-entropy loss, during which its latent
space is aligned with that of CNN-based AE, which was pre-trained on the chromatin images.

Two choices to align the latent representations:
    I) A discriminator that can be used to compare samples from the latent layers of two Autoencoders, and align
    corresponding clusters of two domain in latent space. It is a four layer fully-connected neural network with
    128 hidden dimensions in each layer, which can be change under "utils>model_utils.py"

    II) A classifier that can be trained on samples from the latent layers of two Autoencoders, and align corresponding
    clusters of two domain in latent space. It is a four layer fully-connected neural network with 128 hidden dimensions
    in each layer, which can be change under "utils>model_utils.py"

"""

import os
import gc
from tqdm import tqdm
import pandas as pd
import itertools
from itertools import cycle

from utils.utils import set_seed, set_dirs
from utils.loss_functions import get_th_vae_loss, getMSEloss, get_generator_loss, get_discriminator_loss
from utils.model_plot import save_loss_plot
from utils.model_utils import Autoencoder, Discriminator, Classifier
from src.model import AEModel

import torch as th
import torch.nn.functional as F

th.autograd.set_detect_anomaly(True)


class DTModel:
    """
    Model: Consists of two Autoencoders together with either Discriminator, or Classifier.
    Loss function: Reconstruction loss of untrained Autoencoder + either Adversarial or Cross-entropy loss.
    ------------------------------------------------------------
    Architecture:  Encoder -> Decoder
                             -> Discriminator, or Classifier
                   Encoder -> Decoder
    ------------------------------------------------------------
    Autoencoders can be configured as
                        - Autoencoder (ae),
                        - Variational autoencoder (vae),
                        - Beta-VAE (bvae),
                        - Adversarial autoencoder (aae).
    ------------------------------------------------------------
    By default, the joint training is done by using adversarial loss. To change it to cross-entropy loss, change
    "joint_training_mode: aae" to "joint_training_mode: sl" in "./config/ae.yaml".
    """

    def __init__(self, options):
        """
        :param dict options: Configuration dictionary.
        """
        # Load pre-trained image autoencoder model
        model_img = AEModel(options)
        # Load weights
        model_img.load_models()
        # Extract autoencoder from the model
        self.autoencoder = model_img.autoencoder
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # Set paths for results and Initialize some arrays to collect data during training
        self._set_paths()
        # Set directories i.e. create ones that are missing.
        set_dirs(self.options)
        # ------Network---------
        # Instantiate networks
        print("Building RNA models for Data Translation...")
        # Turn off convolution to get fully-connected AE model
        self.options["convolution"] = False
        # Set RNA Autoencoder i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder_rna()
        # Set AEE i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        if self.options["joint_training_mode"] == "aae":
            self.set_aae()
            self.options["supervised"] = False
        # Instantiate and set up Classifier if "supervised" i.e. loss, optimizer, device (GPU, or CPU)
        self.set_classifier_dt() if self.options["supervised"] else None
        # Set scheduler (its use is optional)
        self._set_scheduler()
        # Print out model architecture
        self.print_model_summary()

    def set_autoencoder_rna(self):
        # Instantiate the model for fully-connected Autoencoder used for RNA-seq
        self.autoencoder_rna = Autoencoder(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"autoencoder_rna": self.autoencoder_rna})
        # Assign autoencoder to a device
        self.autoencoder_rna.to(self.device)
        # Reconstruction loss
        self.recon_loss = getMSEloss
        # Set optimizer for autoencoder
        self.optimizer_rna = self._adam([self.autoencoder_rna.parameters()], lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": [], "kl_loss": []})

    def set_classifier_dt(self):
        # Instantiate Classifier
        self.classifier_dt = Classifier(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"classifier": self.classifier_dt})
        # Assign classifier to a device
        self.classifier_dt.to(self.device)
        # If data is imbalanced, use weights in cross-entropy loss. For a 9-to-1 ratio of binary class, use 4.5, 0.5
        self.ce_weights = th.FloatTensor([1.0, 1.0]).to(self.device)
        # Cross-entropy loss
        self.xloss = th.nn.CrossEntropyLoss(weight=self.ce_weights, reduction='mean')
        # Set optimizer for classifier
        self.optimizer_cl = self._adam([self.classifier_dt.parameters(), self.autoencoder_rna.encoder.parameters()],
                                       lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"class_train_acc": [], "class_test_acc": []})

    def set_aae(self):
        # Instantiate Classifier
        self.discriminator = Discriminator(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"discriminator": self.discriminator})
        # Assign classifier to a device
        self.discriminator.to(self.device)
        # Generator loss
        self.gen_loss = get_generator_loss
        # Discriminator loss
        self.disc_loss = get_discriminator_loss
        # Set optimizer for generator
        self.optimizer_gen = self._adam([self.autoencoder_rna.encoder.parameters()], lr=1e-3)
        # Set optimizer for discriminator
        self.optimizer_disc = self._adam([self.discriminator.parameters()], lr=1e-5)
        # Add items to summary to be used for reporting later
        self.summary.update({"disc_train_acc": [], "disc_test_acc": []})

    def set_parallelism(self, model):
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        if th.cuda.is_available() and th.cuda.device_count() > 1:
            print(th.cuda.device_count(), " GPUs will be used!")
            model = th.nn.DataParallel(model)
        return model

    def fit(self, data_loader):
        """
        :param list args: List of training and test datasets.
        :return: None

        Fits model to the data
        """
        # Get data loaders for imaging and RNA sequence datasets
        img_loader, rna_loader = data_loader
        # Training and Validation set for imaging dataset
        train_loader_img, validation_loader_img = img_loader.train_loader, img_loader.test_loader
        # Training and Validation set for RNA dataset
        train_loader_rna, validation_loader_rna = rna_loader.train_loader, rna_loader.test_loader
        # Placeholders for record batch losses
        self.loss = {"rloss_b": [], "kl_loss": [], "closs_b": [], "rloss_e": [], "closs_e": [], "vloss_e": [],
                     "aae_loss": []}
        # Placeholder for classifier accuracy
        self.acc = {"acc_e": [], "acc_b": []}
        # Turn on training mode for each model.
        self.set_mode(mode="training")
        # Compute batch size
        bs = self.options["batch_size"]
        # Compute total number of batches per epoch
        self.total_batches = len(train_loader_img)
        # Start joint training of Autoencoder, and/or classifier
        for epoch in range(self.options["epochs"]):
            # zip() both data loaders, and cycle the one with smaller dataset to go through all samples of longer one.
            zipped_data_loaders = zip(train_loader_img, cycle(train_loader_rna))
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(zipped_data_loaders), total=self.total_batches, leave=True)
            # Go through batches
            for i, (img_dict, rna_dict) in self.train_tqdm:
                # Process the batch i.e. turning it into a tensor
                Ximg, Xrna = img_dict['tensor'].to(self.device), rna_dict['tensor'].to(self.device)
                # Get labels
                labels_img, labels_rna = img_dict["binary_label"].to(self.device), rna_dict["binary_label"].to(
                    self.device)
                # Forward pass on Img Autoencoder
                with th.no_grad():
                    _, latent_img, _, _ = self.autoencoder(Ximg)
                # Forward pass on RNA Autoencoder
                recon_rna, latent_rna, mean_rna, logvar_rna = self.autoencoder_rna(Xrna)
                # Compute reconstruction loss
                recon_rna_loss = self.recon_loss(recon_rna, Xrna)
                # Add KL loss to compute total loss if we are using variational methods
                total_loss, kl_loss = get_th_vae_loss(recon_rna_loss, mean_rna, logvar_rna, self.options)
                # Record reconstruction loss
                self.loss["rloss_b"].append(recon_rna_loss.item())
                # Record KL loss if we are using variational inference
                self.loss["kl_loss"] += [kl_loss.item()] if self.options["model_mode"] in ["vae", "bvae"] else []
                # Update Autoencoder params
                self._update_model(total_loss, self.optimizer_rna, retain_graph=True)
                # Update generator (Encoder) and discriminator if we are using Adversarial AE
                if self.options["joint_training_mode"] == "aae":
                    disc_loss, gen_loss = self.update_generator_discriminator(
                        [latent_img, Xrna, labels_img, labels_rna])
                    self.loss["aae_loss"].append([disc_loss, gen_loss])
                    del disc_loss, gen_loss
                # Update Classifier if it is being used
                if self.options["supervised"]:
                    # Forward pass on Autoencoder
                    _, latent_rna, _, _ = self.autoencoder_rna(Xrna)
                    # Forward pass on Classifier
                    preds_img, preds_rna = self.classifier_dt(latent_img), self.classifier_dt(latent_rna)
                    # Compute loss
                    xloss = 0.5 * self.xloss(preds_img, labels_img) + 0.5 * self.xloss(preds_rna, labels_rna)
                    # Update Classifier params
                    self._update_model(xloss, self.optimizer_cl, retain_graph=False)
                    # Compute mean accuracy for imaging
                    acc = 0.5 * (th.argmax(preds_img, dim=1) == labels_img).float().mean()
                    # Compute mean accuracy for RNA
                    acc += 0.5 * (th.argmax(preds_rna, dim=1) == labels_rna).float().mean()
                    # Record accuracy
                    self.acc["acc_b"].append(acc)
                    # Get accuracy per epoch
                    self.acc["acc_e"].append(sum(self.acc["acc_b"][-self.total_batches:-1]) / self.total_batches)
                    # Record cross-entropy loss
                    self.loss["closs_b"].append(xloss.item())
                    # Get cross-entropy loss for training per epoch
                    self.loss["closs_e"].append(sum(self.loss["closs_b"][-self.total_batches:-1]) / self.total_batches)
                    # Clean-up for efficient memory use.
                    del xloss, preds_img, preds_rna, labels_img, labels_rna, acc
                # Update log message using epoch and batch numbers
                self.update_log(epoch, i)
                # Clean-up for efficient memory use.
                del recon_rna_loss, total_loss, kl_loss, recon_rna, latent_rna, mean_rna, logvar_rna
                gc.collect()
            # Get reconstruction loss for training per epoch
            self.loss["rloss_e"].append(sum(self.loss["rloss_b"][-self.total_batches:-1]) / self.total_batches)
            # Validate every nth epoch. n=1 by default
            _ = self.validate(validation_loader_rna) if epoch % self.options["nth_epoch"] == 0 else None
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")
        return self.loss

    def validate(self, validation_loader):
        with th.no_grad():
            # Initialize validation loss
            vloss = 0
            # Turn on evaluatin mode
            self.set_mode(mode="evaluation")
            # Compute total number of batches per epoch
            total_batches = len(validation_loader)
            # Print  validation message
            print(f"Computing validation loss. #Samples:{len(validation_loader.dataset)},#Batches:{total_batches}")
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            val_tqdm = tqdm(enumerate(validation_loader), total=total_batches, leave=True)
            # Go through batches
            for i, Xbatch in val_tqdm:
                # Move batch to the device
                Xbatch = Xbatch['tensor'].to(self.device).float()
                recon, latent, _, _ = self.autoencoder_rna(Xbatch)
                # Record validation loss
                val_loss = getMSEloss(recon, Xbatch)
                # Get validation loss
                vloss = vloss + val_loss.item()
                # If supervised, it means that the classifier is also used, so plot ROC curve
                if self.options["supervised"]:
                    predictions = self.classifier_dt(latent)
                    predictions = predictions.argmax(1)
                    del predictions
                # Clean up to avoid memory issues
                del val_loss, recon, latent
                gc.collect()
            # Turn on training mode
            self.set_mode(mode="training")
            # Compute mean validation loss
            vloss = vloss / total_batches
            # Record the loss
            self.loss["vloss_e"].append(vloss)
            # Return mean validation loss
        return vloss

    def update_generator_discriminator(self, data, retain_graph=True):
        # Get the data
        latent_img, Xrna, labels_img, labels_rna = data
        # Sample fake samples
        latent_real = latent_img
        # Normalize the noise if samples from posterior (i.e. latent variable) is also normalized.
        latent_real = F.normalize(latent_real, p=2, dim=1) if self.options["normalize"] else latent_real
        # ----  Start of Discriminator update: Autoencoder in evaluation mode
        self.autoencoder_rna.eval()
        # Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder_rna(Xrna)
        # Concatenate labels of image data (repeated 10 times) to its corresponding embedding (i.e. conditional)
        latent_real = th.cat((latent_real, labels_img.float().view(-1, 1).expand(-1, 10)), dim=1)
        # Concatenate labels of RNA data (repeated 10 times) to its corresponding embedding (i.e. conditional)
        latent_fake = th.cat((latent_fake, labels_rna.float().view(-1, 1).expand(-1, 10)), dim=1)
        # Get predictions for real samples
        pred_fake = self.discriminator(latent_fake.detach())
        # Get predictions for fake samples
        pred_real = self.discriminator(latent_real)
        # Compute discriminator loss
        disc_loss = self.disc_loss(pred_real, pred_fake)
        # Reset optimizer
        self.optimizer_disc.zero_grad()
        # Backward pass
        disc_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_disc.step()
        # ---- Start of Generator update: Autoencoder in train mode
        self.autoencoder_rna.encoder.train()
        # Discriminator in evaluation mode
        self.discriminator.eval()
        #  Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder_rna(Xrna)
        # Concatenate labels of RNA data (repeated 10 times) to its corresponding embedding (i.e. conditional)
        latent_fake = th.cat((latent_fake, labels_rna.float().view(-1, 1).expand(-1, 10)), dim=1)
        # Get predictions for real samples
        pred_fake = self.discriminator(latent_fake)
        # Compute discriminator loss
        gen_loss = self.gen_loss(pred_fake)
        # Reset optimizer
        self.optimizer_gen.zero_grad()
        # Backward pass
        gen_loss.backward(retain_graph=retain_graph)
        # Update parameters of discriminator
        self.optimizer_gen.step()
        # Turn training mode back on
        self.autoencoder_rna.train()
        self.discriminator.train()
        # Return losses
        return disc_loss.item(), gen_loss.item()

    def update_log(self, epoch, batch):
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch - 1}], Batch:[{batch}], Recon. loss:{self.loss['rloss_b'][-1]:.4f}"
            if self.options["supervised"]:
                description += f", CE loss:{self.loss['closs_b'][-1]:.4f}, Accuracy:{self.acc['acc_b'][-1]:.4f}"
            if self.options["joint_training_mode"] == "aae":
                description += f", Disc loss:{self.loss['aae_loss'][-1][0]:.4f}, Gen:{self.loss['aae_loss'][-1][1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch - 1}] training loss:{self.loss['rloss_e'][-1]:.4f}, val loss:{self.loss['vloss_e'][-1]:.4f}"
            if self.options["supervised"]:
                description += f" , CE loss:{self.loss['closs_e'][-1]:.4f}, Accuracy:{self.acc['acc_e'][-1]:.4f}"
            if self.options["joint_training_mode"] == "aae":
                description += f", Disc loss:{self.loss['aae_loss'][-1][0]:.4f}, Gen:{self.loss['aae_loss'][-1][1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        if mode == "training":
            self.autoencoder_rna.train()
            self.classifier_dt.train() if self.options["supervised"] else None
            self.discriminator.train() if self.options["joint_training_mode"] == "aae" else None
        else:
            self.autoencoder_rna.eval()
            self.classifier_dt.eval() if self.options["supervised"] else None
            self.discriminator.eval() if self.options["joint_training_mode"] == "aae" else None

    def save_weights(self):
        """
        :return: None
        Used to save weights of Autoencoder, and (if options['supervision'] == 'supervised) Classifier
        """
        for model_name in self.model_dict:
            th.save(self.model_dict[model_name], self._model_path + "/" + model_name + ".pt")
        print("Done with saving models.")

    def load_models(self):
        """
        :return: None
        Used to load weights saved at the end of the training.
        """
        print(self.model_dict)
        for model_name in self.model_dict:
            model = th.load(self._model_path + "/" + model_name + ".pt", map_location=self.device)
            setattr(self, model_name, model.eval())
            print(f"--{model_name} is loaded")
        print("Done with loading models.")

    def print_model_summary(self):
        """
        :return: None
        Sanity check to see if the models are constructed correctly.
        """
        # Summary of the model
        description = f"{40 * '-'}Summarize models (Autoencoder for RNA-seq and Discriminator/Classifier for Joint Training):{40 * '-'}\n"
        description += f"{34 * '='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34 * '='}\n"
        description += f"{self.autoencoder_rna}\n"
        # Summary of Classifier if it is being used
        if self.options["supervised"]:
            description += f"{30 * '='} Classifier {30 * '='}\n"
            description += f"{self.classifier_dt}\n"
            # Summary of Discriminator if the model is based on Adversarial AE
        if self.options["joint_training_mode"] == "aae":
            description += f"{30 * '='} Discriminator {30 * '='}\n"
            description += f"{self.discriminator}\n"
            # Print model architecture
        print(description)

    def _update_model(self, loss, optimizer, retain_graph=True):
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def _set_scheduler(self):
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_rna, step_size=2, gamma=0.99)

    def _set_paths(self):
        """ Sets paths to bse used for saving results at the end of the training"""
        # Top results directory
        self._results_path = self.options["paths"]["results"]
        # Directory to save model
        self._model_path = os.path.join(self._results_path, "training", self.options["model_mode"], "model")
        # Directory to save plots as png files
        self._plots_path = os.path.join(self._results_path, "training", self.options["model_mode"], "plots")
        # Directory to save losses as csv file
        self._loss_path = os.path.join(self._results_path, "training", self.options["model_mode"], "loss")

    def _adam(self, params, lr=1e-4):
        return th.optim.Adam(itertools.chain(*params), lr=lr, betas=(0.9, 0.999))

    def _tensor(self, data):
        return th.from_numpy(data).to(self.device).float()