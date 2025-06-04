import torch.optim

from cifar_mps.data.dataset import get_dataloaders 
from cifar_mps.config import parse_args
from cifar_mps.models import get_models
from cifar_mps.training_utils.train_n_val import train_n_val
from cifar_mps.training_utils.schedulers import CosineWarmupScheduler 

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")

    args,train_config, exp_config = parse_args()

    if exp_config.verbose:
        print(args)
        print(train_config)
        print(exp_config)

    # Get data
    train_loader,test_loader = get_dataloaders(train_config)
    
    total_iters = len(train_loader)*train_config.epochs

    # Create Model and optimizer and scheduler 
    model = get_models(train_config.model_name)
    model = model.to(device)

    if train_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params=model.parameters,lr=train_config.max_lr,weight_decay=train_config.wd)
    elif train_config.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(),lr=train_config.max_lr,weight_decay=train_config.wd,nesterov=True,momentum=0.9)
    else:
        print("Optimizer not in the accepted, defaulting to sgd")
        train_config.optimizer = "sgd"
        optimizer = torch.optim.SGD(params=model.parameters(),lr=train_config.max_lr,weight_decay=train_config.wd,nesterov=True,momentum=0.9)
    
    if train_config.scheduler == "cosine":
        scheduler = CosineWarmupScheduler(optimizer, train_config.max_lr,total_iters,train_config.lr_warmup,train_config.max_lr/100)
    else:
        scheduler = None
    
    train_n_val(model,optimizer,scheduler,train_loader,test_loader,train_config,exp_config,device)