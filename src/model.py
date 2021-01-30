"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Class to train an Autoencoder using the chromatin images.
"""

import os
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
import itertools
from sklearn.utils import shuffle
from utils.utils import set_seed
from utils.loss_functions import  getKLloss, get_th_vae_loss, getMSEloss, getCEloss, get_generator_loss, get_discriminator_loss
from utils.model_plot import save_loss_plot
from utils.model_utils import Autoencoder, Discriminator, Classifier
import torch as th
import torch.nn.functional as F
th.autograd.set_detect_anomaly(True)


class AEModel:
    """
    Model:
    Loss function:
    """

    def __init__(self, options):
        """
        :param dict options: Configuration dictionary.
        """
        # Get config
        self.options = options
        # Define which device to use: GPU, or CPU
        self.device = options["device"]
        # Create empty lists and dictionary
        self.model_dict, self.summary = {}, {}
        # Set random seed
        set_seed(self.options)
        # ------Network---------
        # Instantiate networks
        print("Building models...")
        # Set Autoencoder i.e. setting loss, optimizer, and device assignment (GPU, or CPU)
        self.set_autoencoder()
        # Instantiate and set up Classifier if "supervised" i.e. loss, optimizer, device (GPU, or CPU)
        self.set_classifier() if self.options["supervised"] else None
        # Set scheduler (its use is optional)
        self.set_scheduler()
        # Set paths for results and Initialize some arrays to collect data during training
        self.set_paths()
        # Print out model architecture
        self.get_model_summary()

    def set_autoencoder(self):
        # Instantiate the model
        self.autoencoder = Autoencoder(self.options)
        # If we are using GPU, and if there are multiple GPUs, parallelize training
        # self.autoencoder = self.set_parallelism(self.autoencoder)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"autoencoder": self.autoencoder})
        # Assign autoencoder to a device
        self.autoencoder.to(self.device)
        # Reconstruction loss
        self.recon_loss = getMSEloss
        # Set optimizer for autoencoder
        self.optimizer_ae = self._adam([self.autoencoder.parameters()], lr=self.options["learning_rate"])
        # Add items to summary to be used for reporting later
        self.summary.update({"recon_loss": [], "kl_loss": []})

    def set_classifier(self):
        # Instantiate Classifier
        self.classifier = Classifier(self.options)
        # Add the model and its name to a list to save, and load in the future
        self.model_dict.update({"classifier": self.classifier})
        # Assign classifier to a device
        self.classifier.to(self.device)
        # If data is imbalanced, use weights in cross-entropy loss. For a 9-to-1 ratio of binary class, use 4.5, 0.5
        self.ce_weights = th.FloatTensor([1.0, 1.0]).to(self.device)
        # Cross-entropy loss
        self.xloss = th.nn.CrossEntropyLoss(weight=self.ce_weights, reduction='mean')
        # Set optimizer for classifier
        self.optimizer_cl = self._adam([self.classifier.parameters(), self.autoencoder.encoder.parameters()], lr=self.options["learning_rate"])
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
        self.optimizer_gen = self._adam([self.autoencoder.encoder.parameters()], lr=1e-3)
        # Set optimizer for discriminator
        self.optimizer_disc = self._adam([self.discriminator.parameters()], lr=1e-4)
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
        # Training dataset
        train_loader = data_loader.train_loader
        # Validation dataset
        validation_loader = data_loader.test_loader
        # Placeholders for record batch losses
        self.loss = {"rloss_b": [], "kl_loss": [], "closs_b": [], "rloss_e": [], "closs_e": [], "vloss_e": [], "aae_loss": []}
        # Placeholder for classifier accuracy
        self.acc = {"acc_e":[], "acc_b":[]}
        # Turn on training mode for each model.
        self.set_mode(mode="training")
        # Compute batch size
        bs = self.options["batch_size"]
        # Compute total number of batches per epoch
        self.total_batches = len(train_loader)
        # Start joint training of Autoencoder, and/or classifier
        for epoch in range(self.options["epochs"]):
            # Attach progress bar to data_loader to check it during training. "leave=True" gives a new line per epoch
            self.train_tqdm = tqdm(enumerate(train_loader), total=self.total_batches, leave=True)
            # Go through batches
            for i, data_dict in self.train_tqdm:
                # Process the batch i.e. turning it into a tensor
                Xbatch = data_dict['tensor'].to(self.device)
                # Forward pass on Autoencoder
                recon, latent, mean, logvar = self.autoencoder(Xbatch)
                # Compute reconstruction loss
                recon_loss = self.recon_loss(recon, Xbatch)
                # Add KL loss to compute total loss if we are using variational methods
                total_loss, kl_loss = get_th_vae_loss(recon_loss, mean, logvar, self.options)
                # Record reconstruction loss
                self.loss["rloss_b"].append(recon_loss.item())
                # Record KL loss if we are using variational inference
                self.loss["kl_loss"] += [kl_loss.item()] if self.options["model_mode"] in ["vae", "bvae", "mvae"] else []
                # Update Autoencoder params
                self.update_model(total_loss, self.optimizer_ae, retain_graph=True)
                # Update generator (Encoder) and discriminator if we are using Adversarial AE
                if self.options["model_mode"] == "aae":
                    disc_loss, gen_loss = self.update_generator_discriminator(Xbatch)
                    self.loss["aae_loss"].append([disc_loss, gen_loss])
                # Update Classifier if it is being used
                if self.options["supervised"]:
                    # Get labels
                    labels = data_dict["binary_label"].to(self.device)
                    # Forward pass on Autoencoder
                    _, latent, _, _ = self.autoencoder(Xbatch)
                    # Forward pass on Classifier
                    preds = self.classifier(latent)
                    # Compute loss
                    xloss = self.xloss(preds, labels)
                    # Record cross-entropy loss
                    self.loss["closs_b"].append(xloss.item())
                    # Record accuracy
                    self.acc["acc_b"].append((th.argmax(preds, dim=1) == labels).float().mean())
                    # Update Classifier params
                    self.update_model(xloss, self.optimizer_cl, retain_graph=False)
                    # Clean-up for efficient memory use.
                    del xloss, preds
                # Update log message using epoch and batch numbers
                self.update_log(epoch, i)
                # Clean-up for efficient memory use.
                del recon_loss, total_loss, kl_loss, recon, latent, mean, logvar
                gc.collect()
            # Get reconstruction loss for training per epoch
            self.loss["rloss_e"].append(sum(self.loss["rloss_b"][-self.total_batches:-1])/self.total_batches)
            # Get cross-entropy loss for training per epoch
            self.loss["closs_e"].append(sum(self.loss["closs_b"][-self.total_batches:-1])/self.total_batches)
            # Get accuracy per epoch
            self.acc["acc_e"].append(sum(self.acc["acc_b"][-self.total_batches:-1])/self.total_batches)
            # Validate every nth epoch. n=1 by default
            _ = self.validate(validation_loader) if epoch % self.options["nth_epoch"] == 0 else None
        # Save plot of training and validation losses
        save_loss_plot(self.loss, self._plots_path)
        # Convert loss dictionary to a dataframe
        loss_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.loss.items()]))
        # Save loss dataframe as csv file for later use
        loss_df.to_csv(self._loss_path + "/losses.csv")
        return self.loss

    def tune(self, data_loader):
        """
        :return: None
        Continues training of previously pre-trained model
        """
        self.load_models()
        self.fit(data_loader)
        self.save_weights()
        print("Done with tuning the model.")

    def update_log(self, epoch, batch):
        # For the first epoch, add losses for batches since we still don't have loss for the epoch
        if epoch < 1:
            description = f"Epoch:[{epoch-1}], Batch:[{batch}], Recon. loss:{self.loss['rloss_b'][-1]:.4f}"
            if self.options["supervised"]:
                description += f", CE loss:{self.loss['closs_b'][-1]:.4f}, Accuracy:{self.acc['acc_b'][-1]:.4f}"
        # For sub-sequent epochs, display only epoch losses.
        else:
            description = f"Epoch:[{epoch-1}] training loss:{self.loss['rloss_e'][-1]:.4f}, val loss:{self.loss['vloss_e'][-1]:.4f}"
            if self.options["supervised"]:
                description += f" , CE loss:{self.loss['closs_e'][-1]:.4f}, Accuracy:{self.acc['acc_e'][-1]:.4f}"
        # Update the displayed message
        self.train_tqdm.set_description(description)

    def set_mode(self, mode="training"):
        if mode == "training":
            self.autoencoder.train()
            self.classifier.train() if self.options["supervised"] else None
        else:
            self.autoencoder.eval()
            self.classifier.eval() if self.options["supervised"] else None

    def sample_noise(self):
        bs = self.options["batch_size"]
        dim = self.options["dims"][-1]
        a = np.random.randint(0, dim, (bs,))
        a = shuffle(a)
        b = np.zeros((a.size, dim))
        b[np.arange(a.size), a] = 1

        eps = np.random.normal(0, 0.1, (bs, dim))

        data2 = b + eps
        data2 = th.from_numpy(data2).float()
        return data2  # th.normal(0, 1, size=(self.options["batch_size"], self.options["dims"][-1]))

    def process_batch(self, Xbatch):
        # Convert the batch to tensor and move it to where the model is
        Xbatch = Xbatch['tensor']
        # Return batches
        return Xbatch

    def get_validation_batch(self, data_loader):
        # Validation dataset
        validation_loader = data_loader.test_loader
        # Use only the first batch of validation set to save from computation
        ((xi, xj), _) = next(iter(validation_loader))
        # Concatenate xi, and xj, and turn it into a tensor
        Xval = self.process_batch(xi, xj)
        # Return
        return Xval

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
                recon, latent, _, _ = self.autoencoder(Xbatch)
                # Record validation loss
                val_loss = getMSEloss(recon, Xbatch)
                # Get validation loss
                vloss = vloss + val_loss.item()
                # If supervised, it means that the classifier is also used, so plot ROC curve
                if self.options["supervised"]:
                    predictions = self.classifier(latent)
                    predictions = predictions.argmax(1)
                    del predictions
                # Clean up to avoid memory issues
                del val_loss, recon, latent
                gc.collect()
            # Turn on training mode
            self.set_mode(mode="training")
            # Compute mean validation loss
            vloss = vloss/total_batches
            # Record the loss
            self.loss["vloss_e"].append(vloss)
            # Return mean validation loss
        return vloss

    def update_generator_discriminator(self, Xbatch, retain_graph=True):
        # Sample fake samples
        latent_real = self.sample_noise()
        # Normalize the noise if samples from posterior (i.e. latent variable) is also normalized.
        latent_real = F.normalize(latent_real, p=2, dim=1) if self.options["normalize"] else latent_real
        #----  Start of Discriminator update: Autoencoder in evaluation mode
        self.autoencoder.eval()
        # Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder(Xbatch)
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
        #---- Start of Generator update: Autoencoder in train mode
        self.autoencoder.encoder.train()
        # Discriminator in evaluation mode
        self.discriminator.eval()
        #  Forward pass on Autoencoder
        _, latent_fake, _, _ = self.autoencoder(Xbatch)
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
        self.autoencoder.train()
        self.discriminator.train()
        # Return losses
        return disc_loss.item(), gen_loss.item()

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

    def get_model_summary(self):
        """
        :return: None
        Sanity check to see if the models are constructed correctly.
        """
        # Summary of the model
        description  = f"{40*'-'}Summarize models:{40*'-'}\n"
        description += f"{34*'='}{self.options['model_mode'].upper().replace('_', ' ')} Model{34*'='}\n"
        description += f"{self.autoencoder}\n"
        # Summary of Classifier if it is being used
        if self.options["supervised"]:
            description += f"{30*'='} Classifier {30*'='}\n"
            description += f"{self.classifier}\n"
        # Print model architecture
        print(description)

    def update_model(self, loss, optimizer, retain_graph=True):
        # Reset optimizer
        optimizer.zero_grad()
        # Backward propagation to compute gradients
        loss.backward(retain_graph=retain_graph)
        # Update weights
        optimizer.step()

    def set_scheduler(self):
        # Set scheduler (Its use will be optional)
        self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer_ae, step_size=2, gamma=0.99)

    def set_paths(self):
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
