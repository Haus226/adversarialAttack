import random
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class Dummy:
    '''
    Creating a dummy class to maintain consistency in function or method calls 
    when certain classes are None, useful in simplifying code logic
    '''
    def __init__(self):
        pass
    
    def forward(self, x):
        return x
    
class DI:
    def __init__(self, low, high, p:float=0.5) -> None:
        self.low = low
        self.high = high
        self.p = p

    def forward(self, x):
        # Random resize
        rnd = torch.randint(low=self.low, high=self.high, size=(1, ))
        rescaled = F.interpolate(x, size=(rnd, rnd), mode='nearest')
        
        # Calculate padding
        h_rem = self.high - rnd
        w_rem = self.high - rnd
        pad_top = random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top
        pad_left = random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        
        # Apply padding
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        # Ensure the padded tensor has the correct shape
        padded = padded.view(x.size(0), self.high, self.high, 3).permute(0, 3, 1, 2)
        return padded if random.random() < self.p else x
    
class TI:
    def __init__(self, 
                kernel_size, 
                nsig,
                device="cuda"
                ) -> None:
        import scipy.stats as st
        kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernel_size))
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        self.kernel = torch.from_numpy(stack_kernel).float().to(device)
        
    def forward(self, grad, device="cuda"):
        return F.conv2d(grad, self.kernel, stride=1, padding="same", groups=3)

class Admix:
    '''
    Paper : Admix: Enhancing the Transferability of Adversarial Attacks
    '''
    def __init__(self, 
                gamma=[1 / (2 ** i) for i in range(5)], 
                eta=0.2, 
                num_samples=3
                ) -> None:
        self.eta = eta
        self.gamma = gamma
        self.num_samples = num_samples
        if not isinstance(self.gamma, list):
                self.m = 1
        else:
            self.m = len(self.gamma)
    
    def forward(self, x):
        batch_size = x.size(0)
        idxs = torch.arange(batch_size)
        x_admix = []
        for _ in range(self.num_samples):
            shuffled_idxs = idxs[torch.randperm(batch_size)]
            slave = x[shuffled_idxs]
            x_admix.append(x + self.eta * slave)
        x_admix = torch.cat(x_admix, dim=0)
        return torch.cat([x_admix * gamma for gamma in self.gamma], dim=0)

class VT:
    '''
    Paper : "Enhancing the Transferability of Adversarial Attacks through Variance Tuning"
    '''
    def __init__(self, 
                num_samples,
                 bound,
                admix=None, 
                dim=None,
                enhanced=True
                ) -> None:
        self.num_samples = num_samples
        self.bound = bound
        self.dim = dim
        self.admix = admix

        # Paper : "Boosting Adversarial Transferability through Enhanced Momentum"
        self.g = 1
        self.enhanced = enhanced
        if not isinstance(admix, Dummy):
            self.m = self.admix.m
            self.n = self.admix.num_samples
            self.gamma = self.admix.gamma

    def forward(self, x, model, 
                labels, 
                target_labels=None,
                device="cuda"
               ):
        noise = torch.zeros_like(x).detach().to(device)
        for _ in range(self.num_samples):
            x_neighbor = x + torch.randn_like(x).to(device) * self.bound.to(device) * self.g
            x_neighbor = self.admix.forward(x_neighbor)
            x_neighbor.retain_grad()
            loss = nn.CrossEntropyLoss()
            outputs = model(self.dim.forward(x_neighbor))
            if target_labels is not None:
                l = -loss(outputs, target_labels)
            else:
                l = loss(outputs, labels)
            grad = torch.autograd.grad(
                    l, x_neighbor, retain_graph=False, create_graph=False
                )[0]
            if isinstance(self.admix, Admix):
                grad = torch.sum(grad, dim=0) / (self.m * self.n)
#             if not isinstance(self.admix, Dummy):
#                 grad = torch.chunk(grad, self.m)
#                 weights = self.gamma
#                 weighted_noise = [g * w for g, w in zip(grad, weights)]
#                 grad = torch.mean(torch.stack(weighted_noise), dim=0)
#                 grad = torch.sum(torch.stack(torch.chunk(torch.Tensor(grad), self.n), dim=0), dim=0)
            noise += grad
            self.g = noise / self.num_samples if self.enhanced else 1
        return noise

class Attack(ABC):
    def __init__(self, 
                alpha, 
                eps, 
                adjustment=None,
                n_iter=10,
                device="cuda",
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
        self.dim = DI(**dim) if dim is not None else Dummy()
        self.tim = TI(**tim) if tim is not None else Dummy()
        self.admix = Dummy()
        self.vt = None
        if admix is not None:
            self.admix = Admix(**admix)
            self.m = self.admix.m
            self.n = self.admix.num_samples
            self.gamma = self.admix.gamma
        if vt is not None:
            self.vt = VT(vt["num_samples"], (self.ad_eps * vt["bound"]).view(1, 3, 1, 1).to(device), admix=self.admix, dim=self.dim, enhanced=vt["enhanced"])
            self.N = self.vt.num_samples
        self.grad = None

    @abstractmethod
    def init_components(self):
        pass

    @abstractmethod
    def get_hp(self):
        pass
    
    def nesterov(self, adv, images):
        '''
        For applying nesterov accelerated gradient
        '''
        return adv

    def forward(self, model, images, labels, target_labels=None):
        
        if images.size(0) == 1:
            early_stop = True
        else:
            early_stop = False
        
        # Ensure the images have gradients enabled
        labels_ = labels.to(self.device) if isinstance(self.admix, Dummy) else labels.repeat(self.m * self.n).to(self.device)
        images = images.to(self.device)
        if target_labels is not None:
            target_labels_ = target_labels.to(self.device) if isinstance(self.admix, Dummy) else target_labels.to(self.device).repeat(self.m * self.n).to(self.device)
        else:
            target_labels_ = None
        loss = nn.CrossEntropyLoss()
        v = torch.zeros_like(images).to(self.device)

        # Initialize the componenets for different subclasses
        self.init_components(images)

        adv = images.detach().clone().to(self.device)
        for idx in range(self.n_iter):
            # Initialize NAG
            adv_nes = self.nesterov(adv, images)
            adv_nes.requires_grad = True
            
            adv_ = self.admix.forward(adv_nes)
            adv__ = self.dim.forward(adv_)
            outputs = model(adv__)
            if early_stop:
                if (target_labels is not None and outputs.max(1)[1] == target_labels_) or (target_labels is None and outputs.max(1)[1] != labels_):
                    return adv_nes.detach(), outputs.max(1)[1], idx + 1
            l = loss(outputs, target_labels_) if target_labels is not None else loss(outputs, labels_)
            noise = torch.autograd.grad(
                l, adv_, retain_graph=False, create_graph=False
            )[0]
            if isinstance(self.admix, Admix):
                noise = torch.sum(noise, dim=0) / (self.m * self.n)
            
#             if not isinstance(self.admix, Dummy):
#                 noise = torch.chunk(noise, self.m)
#                 weights = self.gamma
#                 weighted_noise = [g * w for g, w in zip(noise, weights)]
#                 noise = torch.mean(torch.stack(weighted_noise), dim=0)
#                 noise = torch.sum(torch.stack(torch.chunk(torch.Tensor(noise), self.n), dim=0), dim=0)
            self.grad = noise + v
            self.grad = self.tim.forward(self.grad)
            self.grad = self.grad / torch.mean(torch.abs(self.grad), dim=(1, 2, 3), keepdim=True)
            if self.vt is not None:
                # Why pass adv instead of adv_ or adv__?
                # According to the formula, since the gradient used to update is from
                # adv_, shouldn't use the adv_ as center of the neighborhood?
                # Maybe that is why we perform admix and dim in vt
                v_grad = self.vt.forward(adv_nes, model, labels_, target_labels_, self.device)
                v = v_grad / self.N - noise
            adv = self.update(adv, idx)
            adv = self.clip(adv, images)
        return adv.detach(), model(adv).max(1)[1], self.n_iter 

    @abstractmethod
    def update(self, adv, idx):
        pass

    def clip(self, adv, images):
        delta = torch.clamp(adv - images, min=-self.ad_eps, max=self.ad_eps)
        return (images + delta).detach()

class MI_FGSM(Attack):
    def __init__(self, alpha=1, beta=1.0, eps=5, adjustment=None, n_iter=10, device="cuda",
                dim:DI=None, tim:TI=None, admix:Admix=None, vt:VT=None
                ):
        super().__init__(alpha, eps, adjustment, n_iter, device, dim, tim, admix, vt)
        self.beta = beta
        self.momentum = None

    def get_hp(self):
        return {"beta":self.beta}

    def init_components(self, x):
        self.momentum = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.grad = self.beta * self.momentum + self.grad
        self.momentum = self.grad
        return adv + self.ad_alpha * self.grad.sign()
    
class NI_FGSM(MI_FGSM):
    def __init__(self, alpha=1, beta=1.0, eps=5, adjustment=None, n_iter=10, device="cuda", dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, beta, eps, adjustment, n_iter, device, dim, tim, admix, vt)

    def nesterov(self, adv, images):
        adv_nes = adv + self.beta * self.ad_alpha * self.momentum.sign()
        return self.clip(adv_nes, images)
    
class AGI_FGSM(Attack):
    def __init__(self, alpha=1, delta=1e-8, eps=5, adjustment=None, n_iter=10, device="cuda", dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, dim, tim, admix, vt)
        self.delta = delta
        self.momentum = None

    def get_hp(self):
        return {"delta":self.delta}

    def init_components(self, x):
        self.momentum = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.momentum = self.momentum + self.grad ** 2
        return adv + self.ad_alpha * (self.grad / (torch.sqrt(self.momentum) + self.delta)).sign()
    
class RMSI_FGSM(Attack):
    def __init__(self, alpha=1, beta=0.99, delta=1e-8, eps=5, adjustment=None, n_iter=10, device="cuda", dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, dim, tim, admix, vt)
        self.beta = beta
        self.delta = delta
        self.momentum = None

    def get_hp(self):
        return {"beta":self.beta, "delta":self.delta}

    def init_components(self, x):
        self.momentum = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.momentum = self.beta * self.momentum + (1 - self.beta) * self.grad ** 2
        return adv + self.ad_alpha * (self.grad / (torch.sqrt(self.momentum) + self.delta)).sign()
    
class AI_FGSM(Attack):
    def __init__(self, alpha=1, beta_1=0.9, beta_2=0.999, delta=1e-8, eps=5, adjustment=None, n_iter=10, device="cuda", dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, eps, adjustment, n_iter, device, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta

    def get_hp(self):
        return {"beta_1":self.beta_1, "beta_2":self.beta_2, "delta":1e-8}

    def init_components(self, x):
        self.momentum_1 = torch.zeros_like(x).to(self.device)
        self.momentum_2 = torch.zeros_like(x).to(self.device)

    def update(self, adv, idx):
        self.momentum_1 = self.beta_1 * self.momentum_1 + (1 - self.beta_1) * self.grad
        self.momentum_2 = self.beta_2 * self.momentum_2 + (1 - self.beta_2) * self.grad ** 2
        b_momentum_1 = self.momentum_1 / (1 - self.beta_1 ** (idx + 1))
        b_momentum_2 = self.momentum_2 / (1 - self.beta_2 ** (idx + 1))
        return adv + self.ad_alpha * (b_momentum_1 / (torch.sqrt(b_momentum_2) + self.delta)).sign()

class NAI_FGSM(AI_FGSM):
    def __init__(self, alpha=1, beta_1=0.9, beta_2=0.999, delta=1e-8, eps=5, adjustment=None, n_iter=10, device="cuda", dim=None, tim=None, admix=None, vt=None) -> None:
        super().__init__(alpha, beta_1, beta_2, delta, eps, adjustment, n_iter, device, dim, tim, admix, vt)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.delta = delta

    def update(self, adv, idx):
        self.momentum_1 = self.beta_1 * self.momentum_1 + (1 - self.beta_1) * self.grad
        self.momentum_2 = self.beta_2 * self.momentum_2 + (1 - self.beta_2) * self.grad ** 2
        b_momentum_1 = self.momentum_1 / (1 - self.beta_1 ** (idx + 1))
        b_momentum_2 = self.momentum_2 / (1 - self.beta_2 ** (idx + 1))
        adv = adv + self.ad_alpha * ((self.beta_1 * b_momentum_1 + (1 - self.beta_1) * self.grad / (1 - self.beta_1 ** (idx + 1))) / (torch.sqrt(b_momentum_2) + self.delta)).sign()        
        return adv