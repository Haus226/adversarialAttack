from torch import nn, optim
import torch
from abc import ABC, abstractmethod
from activation import *
from misc import *
import math

# TODO: 
# Weight decay, dcouple?
# Adanorm ?
# Adam debias ?
# Gradient Centralization ?

class Attack(ABC):
    def __init__(self, 
                alpha, 
                eps, 
                adjustment=None,
                n_iter=10,
                device="cuda",
                activation:str="sign",
                dim=None,
                tim=None,
                admix=None,
                vt=None,
                ) -> None:
        if adjustment is None:
            adjustment = [1, 1, 1]
        self.ad_alpha = alpha * torch.tensor(adjustment, device=device).view(1, 3, 1, 1)
        self.ad_eps = eps * torch.tensor(adjustment, device=device).view(1, 3, 1, 1)
        self.n_iter = n_iter
        self.device = device
        self.perturbation = None
        self.dim = DI(**dim) if dim is not None else Dummy()
        self.tim = TI(**tim) if tim is not None else Dummy()
        self.admix = Dummy()
        if activation not in ACTIVATION:
            raise NotImplementedError(f"Please implement {activation} function in activation.py")
        self.activation_name = activation
        self.activation = ACTIVATION[activation]
        self.vt = None
        if admix is not None:
            self.admix = Admix(**admix)
            self.m = self.admix.m
            self.n = self.admix.num_samples
            self.gamma = self.admix.gamma
        if vt is not None:
            self.vt = VT(vt["num_samples"], (self.ad_eps * vt["bound"]).view(1, 3, 1, 1).to(device), 
                        admix=self.admix, dim=self.dim, enhanced=vt["enhanced"])
            self.N = self.vt.num_samples
        self.grad = None

    @abstractmethod
    def init_components(self):
        pass
    
    def nesterov(self, adv, images):
        return adv

    def forward(self, model, images, labels, target_labels=None):
        
        early_stop = True if images.size(0) else False        
        labels_ = labels.to(self.device) if isinstance(self.admix, Dummy) else labels.repeat(self.m * self.n).to(self.device)
        images = images.to(self.device)
        if target_labels is not None:
            target_labels_ = target_labels.to(self.device) if isinstance(self.admix, Dummy) else target_labels.to(self.device).repeat(self.m * self.n).to(self.device)
        else:
            target_labels_ = None
        loss = nn.CrossEntropyLoss()
        self.perturbation = torch.zeros_like(images).to(self.device)

        self.init_components(images)
        adv = images.detach().clone().to(self.device)
        for idx in range(self.n_iter):
            adv.requires_grad = True

            # Initialize NAG
            adv_nes = self.nesterov(adv, images)
            outputs = model(adv_nes)

            l = -loss(outputs, target_labels_) if target_labels is not None else loss(outputs, labels_)
            if early_stop:
                if (target_labels is not None and outputs.max(1)[1] == target_labels_) or (target_labels is None and outputs.max(1)[1] != labels_):
                    return adv_nes.detach(), outputs.max(1)[1], l.detach().item(), self.perturbation.detach(), idx + 1
            noise = torch.autograd.grad(
                l, adv, retain_graph=False, create_graph=False
            )[0]

            self.grad = noise
            self.grad = self.grad / torch.mean(torch.abs(self.grad), dim=(1, 2, 3), keepdim=True)
            adv = self.update(adv, idx)
            adv = self.clip(adv, images).detach()
        return adv.detach(), model(adv).max(1)[1], l.detach().item(), self.perturbation.detach(), self.n_iter 

    @abstractmethod
    def update(self, adv, idx):
        pass

    def clip(self, adv, images):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        delta = torch.clamp(adv - images, min=-self.ad_eps, max=self.ad_eps)
        self.perturbation = delta
        adv_ = images + delta
        adv_ = adv_ * std + mean        
        adv_ = torch.clamp(adv_, 0, 1)
        adv_ = (adv_ - mean) / std
        return adv_

class Momentum(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda", 
                beta=1.0,
                activation="sign",
                dim:DI=None, tim:TI=None, admix:Admix=None, vt:VT=None
                ):
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.beta = beta
        self.momentum = None

    def init_components(self, x):
        self.momentum = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        g = self.beta * self.momentum + self.grad
        self.momentum = g
        return adv + self.ad_alpha * self.activation(g)
    
class Nesterov(Momentum):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda", 
                beta=1.0,
                activation="sign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, beta, activation, dim, tim, admix, vt)

    def nesterov(self, adv, images):
        adv_nes = adv + self.beta * self.ad_alpha * self.activation(self.momentum)
        return self.clip(adv_nes, images)
    
class AdaGrad(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda", 
                delta = 1e-8,
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.delta = delta
        self.squared_grad = None

    def init_components(self, x):
        self.squared_grad = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.squared_grad = self.squared_grad + self.grad ** 2
        g = self.grad / (torch.sqrt(self.squared_grad) + self.delta)
        return adv + self.ad_alpha * self.activation(g)
    
class AdaDelta(AdaGrad):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta=0.9, delta=1e-6,
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, delta, activation, dim, tim, admix, vt)
        self.beta = beta
        self.squared_x = None

    def init_components(self, x):
        self.squared_grad = torch.zeros_like(x).to(self.device)
        self.squared_x = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.squared_grad = self.beta * self.squared_grad + (1 - self.beta) * self.grad ** 2
        delta_x = torch.sqrt((self.squared_x + self.delta) / (self.squared_grad + self.delta)) * self.grad
        self.squared_x = self.beta * self.squared_x + (1 - self.beta) * delta_x ** 2
        return adv + self.activation(delta_x)
    
class RMSprop(AdaGrad):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta=0.99, delta=1e-8, 
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, delta, activation, dim, tim, admix, vt)
        self.beta = beta

    def update(self, adv, idx):
        self.squared_grad = self.beta * self.squared_grad + (1 - self.beta) * self.grad ** 2
        g = self.grad / (torch.sqrt(self.squared_grad) + self.delta)
        return adv + self.ad_alpha * self.activation(g)
    
class Adam(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta_1=0.9, beta_2=0.999, delta=1e-8,
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta

    def init_components(self, x):
        self.momentum_1 = torch.zeros_like(x).to(self.device)
        self.momentum_2 = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.momentum_1 = self.beta_1 * self.momentum_1 + (1 - self.beta_1) * self.grad
        self.momentum_2 = self.beta_2 * self.momentum_2 + (1 - self.beta_2) * self.grad ** 2
        b_momentum_1 = self.momentum_1 / (1 - self.beta_1 ** (idx + 1))
        b_momentum_2 = self.momentum_2 / (1 - self.beta_2 ** (idx + 1))
        g = (b_momentum_1 / (torch.sqrt(b_momentum_2) + self.delta))
        return adv + self.ad_alpha * self.activation(g)

class AdaBelief(Adam):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta_1=0.9, beta_2=0.999, delta=1e-8,
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, beta_1, beta_2, delta, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta

    def update(self, adv, idx):
        self.momentum_1 = self.beta_1 * self.momentum_1 + (1 - self.beta_1) * self.grad
        self.momentum_2 = self.beta_2 * self.momentum_2 + (1 - self.beta_2) * (self.grad - self.momentum_1) ** 2 + self.delta
        b_momentum_1 = self.momentum_1 / (1 - self.beta_1 ** (idx + 1))
        b_momentum_2 = self.momentum_2 / (1 - self.beta_2 ** (idx + 1))
        g = (b_momentum_1 / (torch.sqrt(b_momentum_2) + self.delta))
        adv = adv + self.ad_alpha * self.activation(g)
        return adv

class NAdam(Adam):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta_1=0.9, beta_2=0.999, delta=1e-8,
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, beta_1, beta_2, delta, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta

    def update(self, adv, idx):
        self.momentum_1 = self.beta_1 * self.momentum_1 + (1 - self.beta_1) * self.grad
        self.momentum_2 = self.beta_2 * self.momentum_2 + (1 - self.beta_2) * self.grad ** 2
        b_momentum_1 = self.momentum_1 / (1 - self.beta_1 ** (idx + 1))
        b_momentum_2 = self.momentum_2 / (1 - self.beta_2 ** (idx + 1))
        g = (self.beta_1 * b_momentum_1 + (1 - self.beta_1) * self.grad / (1 - self.beta_1 ** (idx + 1))) / (torch.sqrt(b_momentum_2) + self.delta)
        adv = adv + self.ad_alpha * self.activation(g)
        return adv
    
class Adan(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta_1=0.02, beta_2=0.08, beta_3=0.01, delta=1e-8, 
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.delta = delta
        self.m_k, self.v_k, self.n_k = None, None, None
        self.g_previous = None
    
    def init_components(self, x):
        self.m_k = torch.zeros_like(x).to(self.device)
        self.v_k = torch.zeros_like(x).to(self.device)
        self.n_k = torch.zeros_like(x).to(self.device)
        self.g_previous = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        bias_correction1 = 1.0 - math.pow(self.beta_1, idx + 1)
        bias_correction2 = 1.0 - math.pow(self.beta_2, idx + 1)
        bias_correction3_sq = math.sqrt(1.0 - math.pow(self.beta_3, idx + 1))
        self.m_k = (1 - self.beta_1) * self.m_k + self.beta_1 * self.grad
        self.v_k = (1 - self.beta_2) * self.v_k + self.beta_2 * (self.grad - self.g_previous)
        self.n_k = (1 - self.beta_3) * self.n_k + self.beta_3 * (self.grad + (1 - self.beta_2) * (self.grad - self.g_previous)) ** 2
        self.g_previous = self.grad.clone()
        de_norm = self.n_k.sqrt().div_(bias_correction3_sq).add_(self.delta)
        g1 = self.m_k / de_norm / bias_correction1
        g2 = self.v_k * (1 - self.beta_2) / de_norm / bias_correction2

        return adv + self.ad_alpha * self.activation(g1 + g2)
    
class Adai(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
            beta_1=0.1, beta_2=0.99, dampening=1.0, delta=1e-3, 
            activation="softsign",
            dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.dampening = dampening
        self.delta = delta
        self.exp_avg, self.exp_avg_sq, self.beta1_prod = None, None, None
        self.param_size = None

    def init_components(self, images):
        self.exp_avg = torch.zeros_like(images)
        self.exp_avg_sq = torch.zeros_like(images)
        self.beta1_prod = torch.ones_like(images)
        self.param_size = images.numel()

    def update(self, adv, idx):
        exp_avg_sq_hat_sum = 0.0
        self.exp_avg_sq.mul_(self.beta_2).addcmul_(self.grad, self.grad, value=1.0 - self.beta_2)
        bias_correction2 = 1 - self.beta_2 ** (idx + 1)
        exp_avg_sq_hat_sum += self.exp_avg_sq.sum() / bias_correction2
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / self.param_size

        exp_avg_sq_hat = self.exp_avg_sq / bias_correction2

        beta1 = (
            1.0
            - (exp_avg_sq_hat / exp_avg_sq_hat_mean).pow_(1.0 / (3.0 - 2.0 * self.dampening)).mul_(self.beta_1)
        ).clamp_(0.0, 1.0 - self.delta)
        beta3 = (1.0 - beta1).pow_(self.dampening)

        self.beta1_prod.mul_(beta1)

        self.exp_avg.mul_(beta1).addcmul_(beta3, self.grad)
        exp_avg_hat = self.exp_avg.div(1.0 - self.beta1_prod).mul_(math.pow(self.beta_1, 1. - self.dampening))

        return adv + self.ad_alpha * self.activation(exp_avg_hat)

class Yogi(Attack):
    def __init__(self, alpha=1, eps=5, adjustment=None, n_iter=10, device="cuda",
                beta_1=0.9, beta_2=0.999, initial_accumulator=1e-6, delta=1e-3, 
                activation="softsign",
                dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, activation, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta
        self.initial_accumulator = initial_accumulator
        self.exp_avg, self.exp_avg_sq = None, None

    def init_components(self, images):
        self.exp_avg = torch.full_like(images, fill_value=self.initial_accumulator)
        self.exp_avg_sq = torch.full_like(images, fill_value=self.initial_accumulator)


    def update(self, adv, idx):
        bias_correction2_sq = math.sqrt(1.0 - math.pow(self.beta_2, idx + 1))
        grad_sq = self.grad * self.grad
        self.exp_avg.mul_(self.beta_1).add_(self.grad, alpha=1.0 - self.beta_1)
        self.exp_avg_sq.addcmul_((self.exp_avg_sq - grad_sq).sign_(), grad_sq, value=-(1.0 - self.beta_2))
        de_nom = self.exp_avg_sq.sqrt().div_(bias_correction2_sq).add_(self.delta)
        return adv + self.ad_alpha * self.activation(self.exp_avg / de_nom)
