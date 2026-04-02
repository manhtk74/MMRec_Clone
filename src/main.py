# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import wandb
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb_key = user_secrets.get_secret("WANDB_KEY")
    wandb.login(key=wandb_key)
    print("Authenticate WandB via Kaggle Secrets.")
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    print("Authenticate WandB via .env")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--text_encoder', '-te', type=str, default='ST-all-MiniLM-L6-V2', help='text encoder name')
    parser.add_argument('--visual_encoder', '-ve', type=str, default='ViT', help='visual encoder name')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    # Init wandb
    wandb.init(
        project="MMRec", 
        name=f"{args.model}-{args.dataset}-{args.text_encoder}-{args.visual_encoder}",
        config={
            "model": args.model,
            "dataset": args.dataset,
            "text_encoder": args.text_encoder,
            "visual_encoder": args.visual_encoder,
        }
    )

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
    
    wandb.finish()


