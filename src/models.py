from src.baselines.RCGAN import RCGANTrainer
from src.baselines.CotGAN import COTGANTrainer
from src.baselines.TimeGAN import TIMEGANTrainer
from src.PCFGAN.PCFGAN import PCFGANTrainer, RPCFGANTrainer
from src.networks.discriminators import LSTMDiscriminator
from src.networks.generators import LSTMGenerator
import torch
from src.evaluations.test_metrics import get_standard_test_metrics
from src.utils import loader_to_tensor
from torch import nn

GENERATORS = {"LSTM": LSTMGenerator}


def get_generator(generator_type, input_dim, output_dim, **kwargs):
    return GENERATORS[generator_type](
        input_dim=input_dim, output_dim=output_dim, **kwargs
    )


DISCRIMINATORS = {"LSTM": LSTMDiscriminator}


def get_discriminator(discriminator_type, input_dim, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, **kwargs)


def get_trainer(config, train_dl, test_dl):
    model_name = "%s" % (config.gan_algo)

    x_real_train = loader_to_tensor(train_dl).to(config.device)

    x_real_test = loader_to_tensor(test_dl).to(config.device)
    D_out_dim = config.D_out_dim
    return_seq = True
    # print(model_name)
    if config.gan_algo == "RCGAN":
        D_out_dim = 1
    else:
        D_out_dim = config.D_out_dim
    if config.dataset == "OU":
        activation = nn.Identity()
    else:
        activation = nn.Tanh()

    generator = GENERATORS[config.generator](
        input_dim=config.G_input_dim,
        hidden_dim=config.G_hidden_dim,
        output_dim=config.input_dim,
        n_layers=config.G_num_layers,
        noise_scale=config.noise_scale,
        BM=config.BM,
        activation=activation,
    )
    discriminator = DISCRIMINATORS[config.discriminator](
        input_dim=config.input_dim,
        hidden_dim=config.D_hidden_dim,
        out_dim=D_out_dim,
        n_layers=config.D_num_layers,
        return_seq=return_seq,
    )
    discriminator1 = DISCRIMINATORS[config.discriminator](
        input_dim=config.input_dim,
        hidden_dim=config.D_hidden_dim,
        out_dim=D_out_dim,
        n_layers=config.D_num_layers,
        return_seq=return_seq,
    )

    # Compute test metrics for train and test set
    test_metrics_train = get_standard_test_metrics(x_real_train)
    test_metrics_test = get_standard_test_metrics(x_real_test)

    trainer = {
        "PathChar_GAN": PCFGANTrainer(
            G=generator,
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            train_dl=train_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        ),
        "RPathChar_GAN": RPCFGANTrainer(
            G=generator,
            D=discriminator,
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            train_dl=train_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        ),
        "RCGAN": RCGANTrainer(
            G=generator,
            D=discriminator,
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            train_dl=train_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        ),
        "TimeGAN": TIMEGANTrainer(
            G=generator,
            gamma=1,
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            train_dl=train_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        ),
        "COTGAN": COTGANTrainer(
            G=generator,
            D_h=discriminator,
            D_m=discriminator1,
            sinkhorn_eps=1,
            sinkhorn_l=10,
            test_metrics_train=test_metrics_train,
            test_metrics_test=test_metrics_test,
            train_dl=train_dl,
            batch_size=config.batch_size,
            n_gradient_steps=config.steps,
            config=config,
        ),
    }[model_name]

    torch.backends.cudnn.benchmark = True

    return trainer, generator
