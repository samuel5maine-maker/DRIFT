"""Unlatched consolidation for task-free MAS (H32).

DRIFT's `tfmas` appends the *penalised* loss to its plateau window, so once
``omega`` is non-zero the quantity it monitors is inflated by the very penalty
the anchor created, and further plateaus become harder to detect.  On CoraFull
the measured effect is decisive: the task cross-entropy sits below the .2
threshold on 15-27% of steps across the whole stream, but the penalised loss
does so on only 1.3-3.8%, so the learner anchors two or three times near the
start and then never again, pinning a 70-class stream to a state reached in its
first few hundred updates.

This learner changes exactly one thing: the plateau window holds the task
cross-entropy instead of the penalised loss.  The thresholds .2 and .1, the
peak latch, the importance estimate, the cumulative average into ``omega``, the
anchor copy, the MAS weight, the window length, the backbone and the optimiser
are all untouched, and no constant is introduced.  With
``monitor_task_loss=False`` the released rule runs unchanged.
"""

import dgl
import numpy as np
import torch

from .mas_geometry_model import NET as MASGeometryNET


class NET(MASGeometryNET):
    def __init__(self, model, args, dataset=None):
        super().__init__(model, args, dataset)
        hp = getattr(args, 'mas_geometry_args', {}) or {}
        self.monitor_task_loss = bool(hp.get('monitor_task_loss', True))
        self.consolidation_steps = []
        self.monitored_losses = []
        self.task_losses = []
        self.total_losses = []
        self.first_seen_steps = {}

    def compute_accounting(self):
        accounting = super().compute_accounting()
        arrivals = max(len(self.first_seen_steps), 1)
        accounting.update({
            'monitor_task_loss': self.monitor_task_loss,
            'consolidation_steps': list(self.consolidation_steps),
            'distinct_classes_delivered': len(self.first_seen_steps),
            'steps_per_new_class': self.optimizer_steps / arrivals,
            'consolidations_per_new_class': self.count_updates / arrivals,
        })
        return accounting

    def observe_cis(self, args, g, features, labels, train_ids):
        self.net.train()
        self.net.zero_grad()

        for label in labels[train_ids].unique():
            if label not in self.seen_classes:
                self.seen_classes.append(label)
            self.first_seen_steps.setdefault(int(label), self.optimizer_steps)
        offset1, offset2 = 0, max(self.seen_classes) + 1
        if offset2 % 2 != 0:
            offset2 += 1

        sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else \
            dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))
        _, _, blocks = sampler.sample_blocks(g, train_ids)
        output_labels = labels[train_ids]
        input_features = blocks[0].srcdata['feat']

        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        task_loss = self.ce(output[:, offset1:offset2], output_labels)

        loss = task_loss
        if len(self.star_variables) != 0 and len(self.omegas) != 0:
            for index, parameter in enumerate(self.net.parameters()):
                loss = loss + self.MAS_weight / 2. * torch.sum(
                    self.omegas[index] * (parameter - self.star_variables[index]) ** 2)

        loss.backward()
        self.opt.step()

        # The released rule monitors the penalised loss, which locks itself out once an
        # anchor exists; this rule monitors the task loss the detector is about.
        task_value = float(task_loss.detach())
        self.task_losses.append(task_value)
        self.total_losses.append(float(loss.detach()))
        monitored = task_loss if self.monitor_task_loss else loss
        self.loss_window.append(monitored.cpu().detach().numpy())
        if len(self.loss_window) > self.loss_window_length:
            del self.loss_window[0]
        self.loss_window_mean = np.mean(self.loss_window)
        self.loss_window_variance = np.var(self.loss_window)
        self.monitored_losses.append(float(self.loss_window_mean))
        if not self.new_peak_detected and \
                self.loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
            self.new_peak_detected = True

        plateau = (self.loss_window_mean < self.loss_window_mean_threshold
                   and self.loss_window_variance < self.loss_window_variance_threshold)
        if self.new_peak_detected and plateau:
            self.count_updates += 1
            self.consolidation_steps.append(self.optimizer_steps)
            self.last_loss_window_mean = self.loss_window_mean
            self.last_loss_window_variance = self.loss_window_variance
            self.new_peak_detected = False

            gradients = [0 for _ in self.net.parameters()]
            self.net.zero_grad()
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            output = output[:, offset1:offset2]
            output.pow_(2)
            output.mean().backward()
            for index, parameter in enumerate(self.net.parameters()):
                gradients[index] += torch.abs(parameter.grad.data.clone())

            omegas_old = self.omegas[:]
            self.omegas = []
            self.star_variables = []
            for index, parameter in enumerate(self.net.parameters()):
                if len(omegas_old) != 0:
                    self.omegas.append(1 / self.count_updates * gradients[index]
                                       + (1 - 1 / self.count_updates) * omegas_old[index])
                else:
                    self.omegas.append(gradients[index])
                self.star_variables.append(parameter.data.clone().detach())

        self.loss_window_means.append(self.loss_window_mean)
        self.loss_window_variances.append(self.loss_window_variance)

        if self.project_classifier:
            self._project_classifier()
        if self.ema is not None:
            with torch.no_grad():
                if not self._ema_initialized:
                    for target, source in zip(self.ema.parameters(), self.net.parameters()):
                        target.copy_(source)
                    self._ema_initialized = True
                else:
                    for target, source in zip(self.ema.parameters(), self.net.parameters()):
                        target.mul_(self.ema_alpha).add_(source, alpha=1 - self.ema_alpha)
        self.optimizer_steps += 1
