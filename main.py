from utilities import utils, webdataset_writer
from inference.infere_uncertainties import infer, save_features, uncertainty_vs_loss
from train.train_uncertainties import train
if __name__ == '__main__':
    #Load configurations
    configs = utils.load_configs("configs")
    
    utils.init_wandb(configs)
    if configs["task"] == "inference":
        infer(configs)
    elif configs["task"] == "train_uncertainties":
        train(configs)
    elif configs["task"]== "prediction_uncertainty":
        uncertainty_vs_loss(configs)
    elif configs['task'] == "save_features":
        save_features(configs)
    elif configs['task'] == "wds_write_parallel":
        webdataset_writer.wds_write_parallel(configs)
    else:
        raise NotImplementedError(f"Task {configs['task']} is not implemented.")
   