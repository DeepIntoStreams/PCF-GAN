import functools

import torch
from torch import autograd

from src.baselines.base import BaseTrainer
from tqdm import tqdm
from src.utils import sample_indices, AddTime
from torch.nn.functional import one_hot
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils
from torch import nn


class RCFGANTrainer(BaseTrainer):
    def __init__(self, D, G, train_dl, config,
                 **kwargs):
        super(RCFGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9), weight_decay=config.weight_decay),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D = D
        self.D_optimizer = torch.optim.Adam(
            D.parameters(), lr=config.lr_D, betas=(0, 0.9), weight_decay=config.weight_decay)  # Using TTUR

        self.train_dl = train_dl
        self.conditional = self.config.conditional
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

            if self.conditional:
                data = next(iter(self.train_dl))
                x = data[0].to(device)
                condition = one_hot(
                    data[1], self.config.num_classes).unsqueeze(1).repeat(1, data[0].shape[1], 1).to(device)
                x_real_batch = torch.cat(
                    [x, condition], dim=2)
            else:
                condition = None
                x_real_batch = next(iter(self.train_dl))[0].to(device)
                if self.config.dataset == 'MNIST':
                    # x = data[0]
                    x_real_batch = x_real_batch.squeeze(1).permute(0, 2, 1)
            with torch.no_grad():
                z = (self.config.noise_scale*torch.randn(self.config.batch_size, self.config.n_lags,
                                                         self.config.G_input_dim)).to(device)

            # z[:, 0, :] *= 0
                if self.config.BM:
                    # z[:, 0] = (0.5*torch.randn(self.config.batch_size,
                    #                          self.config.G_input_dim)).to(device)
                    z = z.cumsum(1)
                else:
                    pass
                x_fake = self.G(batch_size=self.batch_size,
                                n_lags=self.config.n_lags, condition=condition, device=device)

            D_loss, enc_loss = self.D_trainstep(x_fake, x_real_batch, z)
            if i == 0:
                self.losses_history['D_loss'].append(D_loss)
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
        if self.conditional:
            condition = one_hot(torch.randint(
                0, self.config.num_classes, (self.batch_size,)),
                self.config.num_classes).float().unsqueeze(1).repeat(1, self.config.n_lags, 1).to(device)

        else:
            condition = None
        x_fake = self.G(batch_size=self.batch_size,
                        n_lags=self.config.n_lags, condition=condition, device=device)

        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.D.train()
        G_loss = CFLossFunc()(x=self.D(x_fake), target=self.D(x_real))
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), self.config.grad_clip)
        self.losses_history['G_loss'].append(G_loss)
        self.G_optimizer.step()
        self.evaluate(x_fake, x_real, step, self.config)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real, z):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()
        enc_loss = self.config.Lambda1 *\
            nn.MSELoss()(self.D(x_fake), z)
        # On real data
        x_real.requires_grad_()
        dloss = CFLossFunc()(x=self.D(x_fake), target=z) - \
            CFLossFunc()(x=self.D(x_fake), target=z) + enc_loss

        # On fake data

        dloss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.D.parameters(), self.config.grad_clip)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return dloss.item(), enc_loss.item()


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        loss_type: a specification of choosing types of CF loss, we use the original one in this version
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1
        threshold: this is mainly used to reduce the effect of CF values around
                    some zero-around t, we do not use this technique in this paper by setting it to 1, you can refer to
                    https://link.springer.com/chapter/10.1007/978-3-030-30487-4_27 for more details
    """

    def __init__(self, alpha=0.5, beta=0.5, threshold=1):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, target):

        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)  # flatten the time series
            target = target.reshape(target.shape[0], -1)
        elif x.dim() == 2:
            pass

        t = torch.randn(64, x.shape[1], device=x.device)
        #print(t.shape, x.shape)
        #t = t.reshape(t.shape[0], -1)
        t_x = torch.mm(t, x.t())
        t_x_real = calculate_real(t_x)
        t_x_imag = calculate_imag(t_x)
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target = torch.mm(t, target.t())
        t_target_real = calculate_real(t_target)
        t_target_imag = calculate_imag(t_target)
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(t_target_norm, t_x_norm) -
                        torch.mul(t_x_real, t_target_real) -
                        torch.mul(t_x_imag, t_target_imag))

        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability

        loss = torch.mean(torch.sqrt(
            self.alpha * loss_amp + self.beta * loss_pha))
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
