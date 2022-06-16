import wandb
import timm
import argparse
import torchvision
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

WANDB_PROJECT = 'paddy-ft'
WANDB_ENTITY = 'bilalcodehub'

dafault_split="fastai.vision.learner.default_split"

config_defaults = SimpleNamespace(
    batch_size=32,
    epochs=1,
    num_experiments=1,
    learning_rate=2e-3,
    img_size=224,
    resize_method="crop",
    model_name="resnet34",
    pool="concat",
    seed=42,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    split_func="default",
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--img_size', type=int, default=config_defaults.img_size)
    parser.add_argument('--resize_method', type=str, default=config_defaults.resize_method)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--split_func', type=str, default=config_defaults.split_func)
    parser.add_argument('--pool', type=str, default=config_defaults.pool)
    parser.add_argument('--seed', type=int, default=config_defaults.seed)
    parser.add_argument('--wandb_project', type=str, default=WANDB_PROJECT)
    parser.add_argument('--wandb_entity', type=str, default=WANDB_ENTITY)
    return parser.parse_args()

def get_gpu_mem(device=0):
    gpu_mem = torch.cuda.memory_stats_as_nested_dict(device=device)
    return (gpu_mem["reserved_bytes"]["small_pool"]["peak"] + gpu_mem["reserved_bytes"]["large_pool"]["peak"])*1024**-3

def get_dataset(batch_size, img_size, seed, method="crop"):
    path = Path.home()/'.fastai/data/paddy'
    files = get_image_files(path/'train_images')
    dls = ImageDataLoaders.from_folder(path, files, valid_pct=0.2,
                                       seed=seed, bs=batch_size, item_tfms=Resize(img_size, method=method))
    return dls, [error_rate, accuracy]

def train(config=config_defaults):
    with wandb.init(project=config.wandb_project, group='timm', entity=config.wandb_entity, config=config):
        config=wandb.config
        dls, metrics = get_dataset(config.batch_size, config.img_size, config.seed, config.resize_method)
        learn = vision_learner(dls, config.model_name, metrics=metrics, concat_pool=(config.pool=="pool"),
                               splitter=default_split if config.split_func=="default" else None,
                               cbs=WandbCallback(log=None, log_preds=False)).to_fp16()
        
        ti=time.perf_counter()
        learn.fine_tune(config.epochs, config.learning_rate)
        wandb.summary["GPU_mem"] = get_gpu_mem(learn.dls.device)
        wandb.summary["model_family"] = config.model_name.split('_')[0]
        wandb.summary["fit_time"] = time.perf_counter() - ti

if __name__=='__main__':
    args=parse_args()
    train(config=args)