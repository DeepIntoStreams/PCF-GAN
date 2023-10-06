import os

from os import path as pt
import argparse
from src.evaluations.evaluate import full_evaluation
from src.utils import get_experiment_dir, save_obj, load_config
import torch
from torch import nn


def main(config):
    """
    Main function for training and evaluating a synthetic data generator.

    Args:
        config (object): Configuration object containing the experiment settings.

    Returns:
        tuple: A tuple containing the discriminative score, predictive score, and signature MMD.
    """
    # print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(config)
    if config.device == "cuda" and torch.cuda.is_available():
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)
    config.update({'pretrained':False},allow_val_change=True)
    config.update({'train':False},allow_val_change=True)
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    get_experiment_dir(config)
    from src.datasets.dataloader import get_dataset

    train_dl, test_dl = get_dataset(config, num_workers=4)
    from src.models import get_trainer

    trainer, generator = get_trainer(config, train_dl, test_dl)
    save_obj(config, pt.join(config.exp_dir, "config.pkl"))

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)
        save_obj(
            trainer.G.state_dict(), pt.join(config.exp_dir, "generator_state_dict.pt")
        )

        save_obj(
            trainer.averaged_G.module.state_dict(),
            pt.join(config.exp_dir, "ave_generator_state_dict.pt"),
        )

        if config.gan_algo == "RPathChar_GAN":
            save_obj(
                trainer.char_func.state_dict(),
                pt.join(config.exp_dir, "M_net_state_dict.pt"),
            )
            save_obj(
                trainer.D.state_dict(),
                pt.join(config.exp_dir, "discriminator_state_dict.pt"),
            )
            save_obj(
                trainer.losses_history, pt.join(config.exp_dir, "losses_history.pkl")
            )
        elif config.gan_algo == "TimeGAN":
            save_obj(
                trainer.recovery.state_dict(),
                pt.join(config.exp_dir, "recovery_state_dict.pt"),
            )
            save_obj(
                trainer.supervisor.state_dict(),
                pt.join(config.exp_dir, "supervisor_state_dict.pt"),
            )
            save_obj(
                trainer.embedder.state_dict(),
                pt.join(config.exp_dir, "embedder_state_dict.pt"),
            )

    elif config.pretrained:
        # config = load_obj(pt.join(
        #   config.exp_dir, 'config.pkl'))
        trainer.G.load_state_dict(
            torch.load(pt.join(config.exp_dir, "ave_generator_state_dict.pt")),
            strict=True,
        )

        if config.gan_algo == "RPathChar_GAN":
            trainer.D.load_state_dict(
                torch.load(pt.join(config.exp_dir, "discriminator_state_dict.pt")),
                strict=True,
            )
            trainer.char_func.load_state_dict(
                torch.load(pt.join(config.exp_dir, "M_net_state_dict.pt")), strict=True
            )
        trainer.fit(config.device)
        save_obj(
            trainer.G.state_dict(), pt.join(config.exp_dir, "generator_state_dict.pt")
        )

        save_obj(
            trainer.averaged_G.module.state_dict(),
            pt.join(config.exp_dir, "ave_generator_state_dict.pt"),
        )
        if config.gan_algo == "RPathChar_GAN":
            save_obj(
                trainer.D.state_dict(),
                pt.join(config.exp_dir, "discriminator_state_dict.pt"),
            )
            save_obj(
                trainer.char_func.state_dict(),
                pt.join(config.exp_dir, "M_net_state_dict.pt"),
            )
        pass

    from src.models import GENERATORS

    if config.gan_algo == "TimeGAN":
        generator = generator.to(device="cpu")
        generator.load_state_dict(
            torch.load(pt.join(config.exp_dir, "ave_generator_state_dict.pt")),
            strict=True,
        )
        supervisor = trainer.supervisor.to(device="cpu")
        supervisor.load_state_dict(
            torch.load(pt.join(config.exp_dir, "supervisor_state_dict.pt")), strict=True
        )
        recovery = trainer.recovery.to(device="cpu")
        recovery.load_state_dict(
            torch.load(pt.join(config.exp_dir, "recovery_state_dict.pt")), strict=True
        )
        recovery = nn.Sequential(supervisor, recovery)
        generator.eval()

        d_score, p_score, sig_mmd = full_evaluation(
            generator, train_dl, test_dl, config, recovery=recovery
        )
    else:
        generator = generator.to(device="cpu")
        generator.load_state_dict(
            torch.load(pt.join(config.exp_dir, "ave_generator_state_dict.pt")),
            strict=True,
        )
        generator.eval()
        d_score, p_score, sig_mmd = full_evaluation(
            generator, train_dl, test_dl, config
        )

    return d_score, p_score, sig_mmd


if __name__ == "__main__":
    import logging
    import argparse
    from src.utils import load_config
    from os import path as pt
    import numpy as np
    from src.evaluations.evaluate import evaluate_reconstruction

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gan_algo",
        type=str,
        default="RCGAN",
        help="choose from TimeGAN,RCGAN,PCFGAN,COTGAN,all",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rough",
        help="choose from rough, stock, air_quality,eeg,all",
    )
    args = parser.parse_args()
    dataset = ["rough", "stock", "air_quality", "eeg"]
    algos = ["PCFGAN", "TimeGAN", "RCGAN", "COTGAN"]
    config_dir = pt.join("configs/", args.gan_algo, args.dataset + ".yaml")
    logging.basicConfig(
        filename="numerical_results/{}_result".format(args.dataset),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    if args.dataset == "all" and args.gan_algo == "all":
        for data in dataset:
            for algo in algos:
                logging.info("Running on {} with {}".format(data, algo))
                config_dir = pt.join("configs/", algo, data + ".yaml")
                main(load_config(config_dir))
                #if algo in ["PCFGAN", "TimeGAN"]:
                 #   logging.info(
                  #      "Running reconstruction evaluation on {} with {}".format(
                   #         data, algo
                    #    )
                    #)
                    #config = load_config(config_dir)
                    #evaluate_reconstruction(config)

    elif args.dataset == "all" and args.gan_algo != "all":
        for data in dataset:
            logging.info("Running on {} with {}".format(data, args.gan_algo))
            config_dir = pt.join("configs/", args.gan_algo, data + ".yaml")
            main(load_config(config_dir))
            if args.gan_algo in ["PCFGAN", "TimeGAN"]:
                logging.info(
                    "Running reconstruction evaluation on {} with {}".format(
                        data, args.gan_algo
                    )
                )
                config = load_config(config_dir)
                evaluate_reconstruction(config)

    elif args.gan_algo == "all" and args.dataset != "all":
        for algo in algos:
            logging.info("Running on {} with {}".format(args.dataset, algo))
            config_dir = pt.join("configs/", algo, args.dataset + ".yaml")
            main(load_config(config_dir))
            if algo in ["PCFGAN", "TimeGAN"]:
                logging.info(
                    "Running reconstruction evaluation on {} with {}".format(
                        args.dataset, algo
                    )
                )
                config = load_config(config_dir)
                evaluate_reconstruction(config)

    else:
        logging.info("Running on {} with {}".format(args.dataset, args.gan_algo))
        main(load_config(config_dir))
        if args.gan_algo in ["PCFGAN", "TimeGAN"]:
            logging.info(
                "Running reconstruction evaluation on {} with {}".format(
                    args.dataset, args.gan_algo
                )
            )
            config = load_config(config_dir)
            evaluate_reconstruction(config)
