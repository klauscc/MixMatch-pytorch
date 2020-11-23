import numpy as np
import torch
import math
import torch.distributions as dist
import torch.nn.functional as F


def generate_gaussian_means(num_classes, data_shape, seed):
    D = np.prod(data_shape)
    means = torch.zeros((num_classes, D))
    cov_std = torch.ones((num_classes))

    torch.manual_seed(seed)
    for i in range(num_classes):
        means[i] = torch.randn(D)

    return means.detach()


def compute_bits_per_dim(x, y, model, prior):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)    # run model forward

    logpz = prior.log_prob(z, y)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()    # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim, z


def compute_nll1(inputs_x, targets_x, inputs_u, targets_u, cnf, prior):
    zx = cnf(inputs_x)
    logits_x = prior.class_logits(zx)
    nll_x = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * targets_x, dim=1))

    zu = cnf(inputs_u)
    logits_u = prior.class_logits(zu)
    nll_u = -torch.mean(torch.sum(F.log_softmax(logits_u, dim=1) * targets_u, dim=1))
    return nll_x, nll_u, zx, zu


def compute_nll(inputs_x, targets_x, inputs_u, targets_u, cnf, prior):
    # zx  = cnf(inputs_x)
    # delta_logp = cnf.logdet()
    # logpz = prior.log_prob(zx, targets_x)
    # logpx = logpz + delta_logp

    # zu = cnf(inputs_u)
    # delta_logp = cnf.logdet()
    # logpz_u = prior.log_prob(zu, targets_u)
    # logpu = logpz_u + delta_logp

    nll_x, zx = compute_bits_per_dim(inputs_x, targets_x, cnf, prior)
    nll_u, zu = compute_bits_per_dim(inputs_u, targets_u, cnf, prior)
    return nll_x, nll_u, zx, zu


class SSLGaussMixture(torch.distributions.Distribution):

    def __init__(self, means, inv_cov_stds=None, device=None):
        self.n_components, self.d = means.shape
        self.means = means

        if inv_cov_stds is None:
            self.inv_cov_stds = math.log(math.exp(1.0) - 1.0) * torch.ones((len(means)), device=device)
        else:
            self.inv_cov_stds = inv_cov_stds

        self.weights = torch.ones((len(means)), device=device)
        self.device = device

    @property
    def gaussians(self):
        gaussians = [
            dist.MultivariateNormal(mean,
                                    F.softplus(inv_std)**2 * torch.eye(self.d).to(self.device))
            for mean, inv_std in zip(self.means, self.inv_cov_stds)
        ]
        return gaussians

    def parameters(self):
        return [self.means, self.inv_cov_std, self.weights]

    def sample(self, sample_shape, gaussian_id=None):
        if gaussian_id is not None:
            g = self.gaussians[gaussian_id]
            samples = g.sample(sample_shape)
        else:
            n_samples = sample_shape[0]
            idx = np.random.choice(self.n_components, size=(n_samples, 1), p=F.softmax(self.weights))
            all_samples = [g.sample(sample_shape) for g in self.gaussians]
            samples = all_samples[0]
            for i in range(self.n_components):
                mask = np.where(idx == i)
                samples[mask] = all_samples[i][mask]
        return samples

    def log_prob(self, x, y=None, label_weight=1.):
        if x.ndim != 2:
            x = torch.flatten(x, 1)
        if y is not None and y.ndim == 2:
            y = torch.argmax(y, dim=1)

        all_log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(F.softmax(self.weights)), dim=1)
        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == -1)
            log_probs[mask] += mixture_log_probs[mask]
            for i in range(self.n_components):
                #Pavel: add class weights here?
                mask = (y == i)
                log_probs[mask] += all_log_probs[:, i][mask] * label_weight
            return log_probs
        else:
            return mixture_log_probs

    def class_logits(self, x):
        if x.ndim != 2:
            x = torch.flatten(x, 1)
        log_probs = torch.cat([g.log_prob(x)[:, None] for g in self.gaussians], dim=1)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)
