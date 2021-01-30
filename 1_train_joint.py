"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: 2nd training step of Domain Translation using Autoencoders.
Training for autoencoders for domain translation across different modalities.

The pre-trained CNN-based AE is loaded, and a fully-connected AE is initialized together with a discriminator, or
classifier. The fully-connected AE is trained with either adversarial, or cross-entropy loss, during which its latent
space is aligned with that of CNN-based AE. Whehter to use adversarial setting, or supervised setting can be defined
by changing "joint_training_mode: aae" option in "/config/ae.yaml".

joint_training_mode: aae  # Use aae for adversarial training, 'sl' for supervised training i.e. cross-entropy loss.
"""

import time
import mlflow
from src.model_dt import DTModel
from utils.load_data import Loader
from utils.arguments import get_arguments, get_config, print_config_summary
from utils.utils import set_dirs

def train(config, data_loader, save_weights=True):
    """
    :param dict config: Dictionary containing options.
    :param IterableDataset data_loader: Pytorch data loader.
    :param bool save_weights: Saves model if True.
    :return:

    Utility function for training and saving the model.
    """
    # Instantiate model
    model = DTModel(config)
    # Start the clock to measure the training time
    start = time.process_time()
    # Fit the model to the data
    model.fit(data_loader)
    # Total time spent on training
    training_time = time.process_time() - start
    # Report the training time
    print(f"Training time:  {training_time//60} minutes, {training_time%60} seconds")
    # Save the model for future use
    _ = model.save_weights() if save_weights else None
    print("Done with training...")
    # Track results
    if config["mlflow"]:
        # Log config with mlflow
        mlflow.log_artifacts("./config", "config")
        # Log model and results with mlflow
        mlflow.log_artifacts("./results/training/" + config["model_mode"], "training_results")
        # log model
        mlflow.pytorch.log_model(model, "models")

def main(config):
    # Ser directories (or create if they don't exist)
    set_dirs(config)
    # Get data loader for imaging dataset.
    img_loader = Loader(config, dataset_name="NucleiDataset")
    # Get data loader for RNA dataset.
    rna_loader = Loader(config, dataset_name="RNADataset")
    # Start training and save model weights at the end
    train(config, [img_loader, rna_loader], save_weights=True)

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

