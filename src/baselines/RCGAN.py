import functools

import torch
from torch import autograd

from src.baselines.base import BaseTrainer
from tqdm import tqdm
from src.utils import sample_indices, AddTime
from torch.nn.functional import one_hot
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils


class RCGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(RCGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR

        self.train_dl = train_dl
        self.reg_param = 0
        self.losses_history
        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer,
            gamma=config.gamma)

        self.D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer,
            gamma=config.gamma)

    def fit(self, device):
        self.G.to(device)
        self.D.to(device)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            with torch.no_grad():
                x_real_batch = next(iter(self.train_dl))[0].to(device)
                x_fake = self.G(batch_size=self.batch_size,
                                n_lags=self.config.n_lags, device=device)

            D_loss_real, D_loss_fake = self.D_trainstep(
                x_fake, x_real_batch)
            if i == 0:
                self.losses_history['D_loss_fake'].append(D_loss_fake)
                self.losses_history['D_loss_real'].append(D_loss_real)
                self.losses_history['D_loss'].append(D_loss_fake + D_loss_real)
        G_loss = self.G_trainstep(x_real_batch, device, step)
        if step % 500 == 0:
            self.D_lr_scheduler.step()
            self.G_lr_scheduler.step()
            # self.M_lr_scheduler.step()
            for param_group in self.D_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

    def G_trainstep(self, x_real, device, step):

        x_fake = self.G(batch_size=self.batch_size,
                        n_lags=self.config.n_lags, device=device)
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        self.D.train()
        G_loss = self.compute_loss(d_fake, 1.)
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), self.config.grad_clip)
        self.losses_history['G_loss'].append(G_loss)
        self.G_optimizer.step()
        if step % 2000 == 0:
            self.evaluate(x_fake, x_real, step, self.config)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1.)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0.)

        # Compute regularizer on fake / real
        dloss = dloss_fake + dloss_real

        dloss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.D.parameters(), self.config.grad_clip)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss_real.item(), dloss_fake.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = torch.nn.BCELoss()(torch.nn.Sigmoid()(d_out), targets)
        return loss


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
