import timm
import wandb, argparse, pandas
from types import SimpleNamespace
from fine_tune import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--sweep_count', type=int, default=None)
    parser.add_argument('--sweep_method', type=str, default="grid")
    return parser.parse_args()

models = [
            'convnext_base',
            'convnext_small',
            'convnext_tiny',
            'convnext_tiny_hnf',
            'efficientnet_b0',
            'efficientnet_es',
            'efficientnet_es_pruned',
            'efficientnet_lite0',
            'levit_128',
            'levit_128s',
            'levit_192',
            'levit_256',
            'levit_384',
            'regnetx_002',
            'regnetx_004',
            'regnetx_006',
            'regnetx_008',
            'regnetx_016',
            'regnetx_032',
            'regnetx_040',
            'regnetx_064',
            'regnetx_080',
            'regnetx_120',
            'regnetx_160',
            'regnety_002',
            'regnety_004',
            'regnety_006',
            'regnety_008',
            'regnety_016',
            'regnety_120',
            'resnet101',
            'resnet152',
            'resnet18',
            'resnet18d',
            'resnet26',
            'resnet26d',
            'resnet34',
            'resnet34d',
            'resnet50',
            'resnet50d',
            'resnet50_gn',
            'resnetblur50',
            'resnetrs50',
            'resnetv2_101',
            'resnetv2_50',
            'resnetv2_50x1_bit_distilled',
            'vit_base_patch16_224',
            'vit_base_patch16_224_miil',
            'vit_base_patch16_224_sam',
            'vit_base_patch32_224',
            'vit_base_patch32_224_sam',
            'vit_small_patch16_224',
            'vit_small_patch32_224',
            'vit_small_r26_s32_224',
            'vit_tiny_patch16_224',
            'vit_tiny_r_s16_p8_224',
            'swin_base_patch4_window7_224',
            'swin_base_patch4_window7_224_in22k',
            'swin_large_patch4_window7_224',
            'swin_large_patch4_window7_224_in22k',
            'swin_s3_base_224',
            'swin_s3_small_224',
            'swin_s3_tiny_224',
            'swin_small_patch4_window7_224',
            'swin_tiny_patch4_window7_224',
            'swinv2_cr_small_224',
            'swinv2_cr_small_ns_224',
            'swinv2_cr_tiny_ns_224',
            'convnext_base_in22k',
            'convnext_large_in22k',
            'convnext_small_in22k',
            'convnext_tiny_in22k',
            'efficientnetv2_rw_m',
            'efficientnetv2_rw_s',
            'efficientnetv2_rw_t',
            'mobilevit_s',
            'mobilevit_xs',
            'mobilevit_xxs',
            'regnetz_040',
            'regnetz_040h',
            'regnetz_b16',
            'regnetz_c16',
            'regnetz_d32',
            'regnetz_d8',
            'regnetz_e8'
        ]

def do_sweep():
    args = parse_args()
    sweep_configs = {
        "method": args.sweep_method,
        "parameters": {
            "model_name": {"values": models},
            "concat_pool": {"values": [True, False]},
            "resize_method":{"values":["crop", "squish"]},
            "num_experiments": {"values": [1,2]},  #just to log the number of the exp
            "learning_rate": {"values": [0.001, 0.008]},
            
        },
    }
    if args.sweep_id is None:
        sweep_id = wandb.sweep(
            sweep_configs,
            project="paddy-ft",
        )
    else:
        print(f"Attaching runs to sweep: {args.sweep_id}")
        sweep_id = "paddy-ft"+"/"+args.sweep_id
    wandb.agent(sweep_id, function=train, count=args.sweep_count)


if __name__ == "__main__":
    do_sweep()