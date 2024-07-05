import random
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

class Dump:
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
                nsig
                ) -> None:
        import scipy.stats as st
        kern1d = st.norm.pdf(np.linspace(-nsig, nsig, kernel_size))
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        self.kernel = torch.from_numpy(stack_kernel).float()
        
    def forward(self, grad, device="cuda"):
        return F.conv2d(grad, self.kernel.to(device), stride=1, padding="same", groups=3)

class Admix:
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
    
    def forward(self, x, device="cuda"):
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
    def __init__(self, 
                bound, 
                num_samples,
                admix=None, 
                dim=None
                ) -> None:
        self.num_samples = num_samples
        self.bound = bound
        self.dim = dim
        self.admix = admix
        if not isinstance(admix, Dump):
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
            x_neighbor = x + torch.randn_like(x).to(device) * self.bound.to(device)
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
            if not isinstance(self.admix, Dump):
                grad = torch.chunk(grad, self.m)
                weights = self.gamma
                weighted_noise = [g * w for g, w in zip(grad, weights)]
                grad = torch.mean(torch.stack(weighted_noise), dim=0)
                grad = torch.sum(torch.stack(torch.chunk(torch.Tensor(grad), self.n), dim=0), dim=0)
            noise += grad
        return noise

class Attack(ABC):
    def __init__(self, 
                alpha, 
                eps, 
                adjustment=None,
                dim=None,
                tim=None,
                admix=None,
                vt=None,
                n_iter=10,
                device="cuda"
                ) -> None:
        if adjustment is None:
            adjustment = [1, 1, 1]
        self.ad_alpha = alpha * torch.tensor(adjustment, device=device).view(1, 3, 1, 1)
        self.ad_eps = eps * torch.tensor(adjustment, device=device).view(1, 3, 1, 1)
        self.n_iter = n_iter
        self.device = device
        self.dim = DI(**dim) if dim is not None else Dump()
        self.tim = TI(**tim) if tim is not None else Dump()
        self.admix = Dump()
        self.vt = None
        if admix is not None:
            self.admix = Admix(**admix)
            self.m = self.admix.m
            self.n = self.admix.num_samples
            self.gamma = self.admix.gamma
        if vt is not None:
            self.vt = VT((self.ad_eps * vt["bound"]).view(1, 3, 1, 1).to(device), vt["num_samples"], admix=self.admix, dim=self.dim)
            self.N = self.vt.num_samples

    @abstractmethod
    def forward(self):
        pass

    def clip(self, adv, images):
        delta = torch.clamp(adv - images, min=-self.ad_eps, max=self.ad_eps)
        adv_ = images + delta
        return adv_.detach()

class MI_FGSM(Attack):
    def __init__(self, alpha, beta, eps, adjustment, n_iter, device="cuda",
                dim:DI=None, tim:TI=None, admix:Admix=None, vt:VT=None
                ):
        super().__init__(alpha, eps, adjustment, dim, tim, admix, vt, n_iter, device)
        self.beta = beta

    def forward(self, model, images, labels, target_labels=None):
    
        if images.size(0) == 1:
            early_stop = True
        # Ensure the images have gradients enabled
        labels_ = labels.to(self.device) if isinstance(self.admix, Dump) else labels.repeat(self.m * self.n).to(self.device)
        images = images.to(self.device)
        if target_labels is not None:
            target_labels_ = target_labels.to(self.device) if isinstance(self.admix, Dump) else target_labels.to(self.device).repeat(self.m * self.n).to(self.device)
        else:
            target_labels_ = None
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).to(self.device)
        v = torch.zeros_like(images).to(self.device)

        adv = images.detach().clone().to(self.device)
        for _ in range(self.n_iter):
            adv.requires_grad = True
            adv_ = self.admix.forward(adv)
            adv__ = self.dim.forward(adv_)
            outputs = model(adv__)
            print(labels_, outputs.max(1)[1])
            if target_labels is not None:
                l = -loss(outputs, target_labels_)
            else:
                l = loss(outputs, labels_)

            noise = torch.autograd.grad(
                l, adv_, retain_graph=False, create_graph=False
            )[0]

            if not isinstance(self.admix, Dump):
                noise = torch.chunk(noise, self.m)
                weights = self.gamma
                weighted_noise = [g * w for g, w in zip(noise, weights)]
                noise = torch.mean(torch.stack(weighted_noise), dim=0)
                noise = torch.sum(torch.stack(torch.chunk(torch.Tensor(noise), self.n), dim=0), dim=0)
            grad = noise + v
            
            grad = self.tim.forward(grad)
            
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = self.beta * momentum + grad
            momentum = grad
            if self.vt is not None:
                v_grad = self.vt.forward(adv, model, labels_, target_labels_, self.device)
                v = v_grad / self.N - noise
            
            adv = adv + self.ad_alpha * grad.sign()
            adv = self.clip(adv, images)
        return adv.detach(), model(adv).max(1)[1], self.n_iter  
