import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import copy
import numpy as np

from mdlt.models import networks
from mdlt.utils.misc import count_samples_per_class, random_pairs_of_minibatches, ParamDict


ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'Fish',
    'ReSamp',
    'ReWeight',
    'SqrtReWeight',
    'CBLoss',
    'Focal',
    'LDAM',
    'BSoftmax',
    'CRT',
    'BoDA'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, env_feats=None):
        """
        Perform one update step, given a list of (x, y) tuples for all envs.
        Admits an optional dict of features from each training domains.
        """
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(ERM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


class ReSamp(ERM):
    """Naive resample, with no changes to ERM, but enable balanced sampling in hparams"""


class ReWeight(ERM):
    """Naive inverse re-weighting"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(ReWeight, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            per_cls_weights = 1 / np.array(count_samples_per_class(labels, num_classes))
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x), y, weight=self.weights_per_env[env].to(device))
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class SqrtReWeight(ReWeight):
    """Square-root inverse re-weighting"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(SqrtReWeight, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            per_cls_weights = 1 / np.sqrt(np.array(count_samples_per_class(labels, num_classes)))
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)


class CBLoss(ReWeight):
    """Class-balanced loss, https://arxiv.org/pdf/1901.05555.pdf"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CBLoss, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.weights_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            effective_num = 1. - np.power(self.hparams["beta"], count_samples_per_class(labels, num_classes))
            effective_num = np.array(effective_num)
            effective_num[effective_num == 1] = np.inf
            per_cls_weights = (1. - self.hparams["beta"]) / effective_num
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * num_classes
            self.weights_per_env[i] = torch.FloatTensor(per_cls_weights)


class Focal(ERM):
    """Focal loss, https://arxiv.org/abs/1708.02002"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Focal, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    @staticmethod
    def focal_loss(input_values, gamma):
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values
        return loss.mean()

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = self.focal_loss(F.cross_entropy(self.predict(all_x), all_y, reduction='none'), self.hparams["gamma"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class LDAM(ERM):
    """LDAM loss, https://arxiv.org/abs/1906.07413"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(LDAM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.m_list = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            m_list = 1. / np.sqrt(np.sqrt(np.array(count_samples_per_class(labels, num_classes))))
            m_list = m_list * (self.hparams["max_m"] / np.max(m_list))
            self.m_list[i] = torch.FloatTensor(m_list)

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            index = torch.zeros_like(x, dtype=torch.uint8)
            index.scatter_(1, y.data.view(-1, 1), 1)
            index_float = index.type(torch.FloatTensor)
            batch_m = torch.matmul(self.m_list[env][None, :].to(device), index_float.transpose(0, 1).to(device))
            batch_m = batch_m.view((-1, 1))
            x_m = x - batch_m
            output = torch.where(index, x_m, x)
            loss += F.cross_entropy(self.hparams["scale"] * output, y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class BSoftmax(ERM):
    """Balanced softmax, https://arxiv.org/abs/2007.10740"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(BSoftmax, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.n_samples_per_env = {}
        for i, env in enumerate(sorted(env_labels)):
            labels = env_labels[env]
            n_samples_per_cls = np.array(count_samples_per_class(labels, num_classes))
            n_samples_per_cls[n_samples_per_cls == np.inf] = 1
            self.n_samples_per_env[i] = torch.FloatTensor(n_samples_per_cls)

    def update(self, minibatches, env_feats=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            x = self.predict(x)
            spc = self.n_samples_per_env[env].type_as(x)
            spc = spc.unsqueeze(0).expand(x.shape[0], -1)
            x = x + spc.log()
            loss += F.cross_entropy(input=x, target=y)
        loss /= len(minibatches)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second earning stage"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CRT, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        self.optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )


class BoDA(ERM):
    """BoDA: balanced domain-class distribution alignment"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(BoDA, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.train_feats = None
        self.train_labels = None
        self.steps = 0
        self.nu = hparams["nu"]
        self.momentum = hparams["momentum"]
        self.temperature = hparams["temperature"]
        self.boda_start_step = hparams["boda_start_step"]
        self.feat_update_freq = hparams["feat_update_freq"]

        # 'env_labels' can be None in evaluation, but not in training
        if env_labels is not None:
            # number of samples per domain-class pair
            self.n_samples_table = torch.tensor([
                count_samples_per_class(env_labels[env], num_classes) for env in sorted(env_labels)])

            self.centroid_classes = torch.tensor(np.hstack([np.unique(env_labels[env]) for env in sorted(env_labels)]))
            self.centroid_envs = torch.tensor(np.hstack([
                i * np.ones_like(np.unique(env_labels[env])) for i, env in enumerate(sorted(env_labels))]))

            self.register_buffer('train_centroids', torch.zeros(self.centroid_classes.size(0), self.featurizer.n_outputs))

    @staticmethod
    def pairwise_dist(x, y):
        return torch.cdist(x, y)

    @staticmethod
    def macro_alignment_loss(x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()
        return mean_diff + cova_diff

    def update_feature_stats(self, env_feats):
        if self.steps == 0 or self.steps % self.feat_update_freq != 0:
            return

        train_feats = [torch.stack(x, dim=0) for x in env_feats['feats'].values()]
        train_labels = [torch.stack(x, dim=0) for x in env_feats['labels'].values()]

        curr_centroids = torch.empty((0, self.train_centroids.size(-1))).to(train_feats[0].device)
        for env in range(len(train_feats)):
            curr_centroids = torch.cat((
                curr_centroids,
                torch.stack([train_feats[env][torch.where(train_labels[env] == c)[0]].mean(0)
                             for c in torch.unique(train_labels[env])])
            ))
        factor = 0 if self.steps == self.feat_update_freq else self.momentum
        self.train_centroids = \
            (1 - factor) * curr_centroids.to(self.train_centroids.device) + factor * self.train_centroids

    def update(self, minibatches, env_feats=None):
        self.update_feature_stats(env_feats)

        n_envs = len(minibatches)
        all_y = torch.cat([y for _, y in minibatches])
        all_envs = torch.cat([env * torch.ones_like(y) for env, (_, y) in enumerate(minibatches)])
        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifiers = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        # cross-entropy loss
        loss_x = 0
        for i in range(n_envs):
            loss_x += F.cross_entropy(classifiers[i], targets[i])
        loss_x /= n_envs

        # BoDA loss
        if self.steps >= self.boda_start_step:
            pairwise_dist = -1 * self.pairwise_dist(self.train_centroids, torch.cat(features))
            # balanced distance
            n_per_sample = self.n_samples_table[all_envs.long(), all_y.long()]
            logits = torch.div(pairwise_dist, n_per_sample.to(pairwise_dist.device))
            # calibrated distance
            n_samples_numerator = self.n_samples_table[self.centroid_envs.long(), self.centroid_classes.long()]
            n_samples_denominator = self.n_samples_table[all_envs.long(), all_y.long()]
            size_h, size_w = n_samples_numerator.size(0), n_samples_denominator.size(0)
            cal_weights = (n_samples_numerator.unsqueeze(1).expand(-1, size_w) /
                           n_samples_denominator.unsqueeze(0).expand(size_h, -1)) ** self.nu
            logits *= cal_weights.to(logits.device)
            logits = torch.div(logits, self.temperature)
            mask_same_d_c = torch.eq(
                self.centroid_classes.contiguous().view(-1, 1).to(all_y.device), all_y.contiguous().view(-1, 1).T).float() * torch.eq(
                self.centroid_envs.contiguous().view(-1, 1).to(all_envs.device), all_envs.contiguous().view(-1, 1).T).float()
            log_prob = logits - torch.log((torch.exp(logits) * (1 - mask_same_d_c)).sum(0, keepdim=True))
            # compute mean of log-likelihood over positive
            mask_cls = torch.eq(self.centroid_classes.contiguous().view(-1, 1).to(all_y.device),
                                all_y.contiguous().view(-1, 1).T).float()
            mask_env = torch.eq(self.centroid_envs.contiguous().view(-1, 1).to(all_envs.device),
                                all_envs.contiguous().view(-1, 1).T).float()
            mask = mask_cls * (1 - mask_env)
            log_prob_pos = log_prob * mask
            loss_b = - log_prob_pos.sum() / mask.sum()

        # macro alignment loss
        # during warm-up stage, helps BoDA loss converge
        # in MDLT, brings marginal improvement to BoDA; in DG, helps improve performance
        # to remove, simply set "macro_weight=0" in hparams_registry
        penalty = 0
        for i in range(n_envs):
            for j in range(i + 1, n_envs):
                penalty += self.macro_alignment_loss(features[i], features[j])
        if n_envs > 1:
            penalty /= (n_envs * (n_envs - 1) / 2)

        self.optimizer.zero_grad()
        loss = loss_x + self.hparams['macro_weight'] * penalty
        if self.steps >= self.boda_start_step:
            loss += self.hparams['boda_weight'] * loss_b
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), f"Loss explosion: {loss.item()}"

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        if self.steps > self.boda_start_step:
            return {'loss': loss_x.item(), 'boda_loss': loss_b.item(), 'penalty': penalty}
        else:
            return {'loss': loss_x.item(), 'penalty': penalty}


class Fish(Algorithm):
    """Gradient Matching for Domain Generalization, Shi et al. 2021."""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Fish, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                                weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    @staticmethod
    def fish(meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, env_feats=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, conditional, class_balance):
        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains, hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.discriminator = networks.MLP(self.featurizer.n_outputs, num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes, self.featurizer.n_outputs)

        # optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) + list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) + list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0], ), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad**2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if self.update_count.item() % (1+d_steps_per_g) < d_steps_per_g:
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = classifier_loss + (self.hparams['lambda'] * -disc_loss)
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(DANN, self).__init__(
            input_shape, num_classes, num_domains, hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CDANN, self).__init__(
            input_shape, num_classes, num_domains, hparams, conditional=True, class_balance=True)


class IRM(ERM):
    """Invariant Risk Minimization"""
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(IRM, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    def update(self, minibatches, env_feats=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"], self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj
            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, env_feats=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)

    def update(self, minibatches, env_feats=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)

            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)

            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)

        That is, when calling .step(), we want grads to be Gi + beta * Gj

        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)

            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)
        self.optimizer.step()

        return {'loss': objective}


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains, hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    @staticmethod
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, env_feats=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifiers = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifiers[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MMD, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(CORAL, self).__init__(input_shape, num_classes, num_domains, hparams, gaussian=False)


class MTL(Algorithm):
    """
    A neural network version of
    Domain Generalization by Marginal Transfer Learning (https://arxiv.org/abs/1711.07910)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(MTL, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs * 2,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.register_buffer('embeddings',
                             torch.zeros(num_domains, self.featurizer.n_outputs))

        self.ema = self.hparams['mtl_ema']

    def update(self, minibatches, env_feats=None):
        loss = 0
        for env, (x, y) in enumerate(minibatches):
            loss += F.cross_entropy(self.predict(x, env), y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def update_embeddings_(self, features, env=None):
        return_embedding = features.mean(0)

        if env is not None:
            return_embedding = self.ema * return_embedding + \
                               (1 - self.ema) * self.embeddings[env]
            self.embeddings[env] = return_embedding.clone().detach()

        return return_embedding.view(1, -1).repeat(len(features), 1)

    def predict(self, x, env=None):
        features = self.featurizer(x)
        embedding = self.update_embeddings_(features, env).normal_()
        return self.classifier(torch.cat((features, embedding), 1))


class SagNet(Algorithm):
    """
    Style Agnostic Network
    Algorithm 1 from: https://arxiv.org/abs/1910.11645
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, env_labels=None):
        super(SagNet, self).__init__(input_shape, num_classes, num_domains, hparams, env_labels)
        # featurizer network
        self.network_f = networks.Featurizer(input_shape, self.hparams)
        # content network
        self.network_c = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        # style network
        self.network_s = networks.Classifier(
            self.network_f.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        def opt(p):
            return torch.optim.Adam(p, lr=hparams["lr"], weight_decay=hparams["weight_decay"])

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = hparams["sag_w_adv"]

    def forward_c(self, x):
        # learning content network on randomized style
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        # learning style network on randomized content
        return self.network_s(self.randomize(self.network_f(x), "content"))

    @staticmethod
    def randomize(x, what="style", eps=1e-5):
        device = "cuda" if x.is_cuda else "cpu"
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1).to(device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, env_feats=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        # learn content
        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # learn style
        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        # learn adversary
        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))
