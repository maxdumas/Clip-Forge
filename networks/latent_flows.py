####### Code build on top of  https://github.com/ikostrikov/pytorch-flows

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl

EPS = 1e-6


class BatchNormFlow(nn.Module):
    """An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var + EPS)).sum(
                -1, keepdim=True
            )
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var + EPS)).sum(
                -1, keepdim=True
            )


class CouplingLayer(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden,
        mask,
        num_cond_inputs=None,
        s_act="tanh",
        t_act="relu",
    ):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_hidden),
            s_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_hidden),
            t_act_func(),
            nn.Linear(num_hidden, num_inputs),
        )

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == "direct":
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_prob(self, pred):
        u, log_jacob = pred
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi + EPS)).sum(
            -1, keepdim=True
        )
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        if cond_inputs is not None:
            cond_inputs = cond_inputs
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples


class LatentFlows(pl.LightningModule):
    def __init__(
        self,
        num_inputs: int,
        num_cond_inputs: int,
        input_type: str,
        output_type: str,
        lr=0.00003,
        noise="add",
        flow_type="realnvp",
        num_blocks=5,
        num_hidden=1024,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_inputs = num_inputs
        self.lr = lr
        self.noise = noise
        self.input_type = input_type
        self.output_type = output_type

        modules = []
        if flow_type == "realnvp":  ### Checkered Masking
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.float()

            for _ in range(num_blocks):
                modules += [
                    CouplingLayer(
                        num_inputs,
                        num_hidden,
                        mask,
                        num_cond_inputs,
                        s_act="tanh",
                        t_act="relu",
                    ),
                    BatchNormFlow(num_inputs),
                ]
                mask = 1 - mask
            self.generator = FlowSequential(*modules)

        elif flow_type == "realnvp_half":  # Dimension Masking
            mask = (torch.arange(0, num_inputs) < (num_inputs / 2)).type(torch.uint8)
            mask = mask.float()
            for _ in range(num_blocks):
                modules += [
                    CouplingLayer(
                        num_inputs,
                        num_hidden,
                        mask,
                        num_cond_inputs,
                        s_act="tanh",
                        t_act="relu",
                    ),
                    BatchNormFlow(num_inputs),
                ]
                mask = 1 - mask
            self.generator = FlowSequential(*modules)

    def forward(self, inputs, cond_inputs=None, mode="direct", logdets=None):
        return self.generator.forward(inputs, cond_inputs, mode, logdets)

    def log_prob(self, pred):
        return self.generator.log_prob(pred)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        return self.generator.sample(num_samples, noise, cond_inputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, data, batch_idx):
        # Load appropriate input data from the training set
        train_embs, train_cond_embs = data[0]

        # Add noise to improve robustness
        if self.noise == "add":
            train_embs = train_embs + 0.1 * torch.randn(
                train_embs.size(0), self.num_inputs
            )

        # Run prediction
        pred = self.forward(train_embs, train_cond_embs)

        # Compute loss
        loss = -self.log_prob(pred).mean()

        self.log("Loss/train", loss)

        return loss

    def validation_step(self, data, data_idx):
        assert len(data) == 1, "More than one datum came to validation step."
        # TODO: Why is this wrapped in an array with one element?
        train_embs, train_cond_embs = data[0]

        # Run prediction
        pred = self.forward(train_embs, train_cond_embs)

        # Compute loss
        loss = -self.log_prob(pred).mean()

        self.log("Loss/val", loss)

        return pred
