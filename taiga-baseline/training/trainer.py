import os
import sys

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

from input.utils import compute_accuracy, compute_cls_metrics, compute_reg_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
#from tools.hypdatatools_img import get_geotrans


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, device,
                 visualizer, metadata, options, hyper_labels_reg, checkpoint=None):
        self.modelTrain = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.options = options
        self.metadata = metadata
        self.categorical = metadata['categorical']
        self.visualizer = visualizer
        self.checkpoint = checkpoint
        self.hyper_labels_reg = hyper_labels_reg
        #self.hypGt = get_geotrans(self.options.hyper_data_header)
        self.save_every = 10  # specify number of epochs to save model

        save_dir = self.options.save_dir or self.options.model
        self.ckpt_path = '../checkpoint/{}'.format(save_dir)
        self.image_path = '../checkpoint/{}/images'.format(save_dir)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

    def compute_uncertainty_loss(self, task_loss):
        """
        Compute uncertainty loss
        :param task_loss: a list contains loss for each task, classification loss first
        :return: uncertainty loss
        """
        task_loss_cls = task_loss[:self.modelTrain.n_cls]
        task_loss_reg = task_loss[-self.modelTrain.n_reg:]

        cls_loss = torch.sum(
            torch.exp(-self.modelTrain.log_sigma_cls) * task_loss_cls + self.modelTrain.log_sigma_cls / 2)
        reg_loss = torch.sum(
            0.5 * torch.exp(-self.modelTrain.log_sigma_reg) * task_loss_reg + self.modelTrain.log_sigma_reg / 2)

        return cls_loss + reg_loss

    def train(self, train_loader, val_loader, test_loader):
        epoch = self.options.epoch
        if self.checkpoint:
            start_epoch = self.checkpoint['epoch'] + 1
            train_step = self.checkpoint['train_step'] + 1
            start_step = train_step
            initial_task_loss = self.checkpoint['initial_task_loss']
        else:
            start_epoch = 1
            train_step = 0
            start_step = 0
        sum_loss = 0.0
        avg_losses = []
        val_losses = []
        train_accuracies = torch.tensor([0.0] * len(self.categorical))
        accuracy_list = []

        loss_window = None
        task_loss_window = None
        task_weights_window = None
        gradnorm_loss_window = None
        accuracy_window = None
        label_names = self.metadata['cls_label_names']

        losses = []

        weights = []
        task_losses = []
        loss_ratios = []
        grad_norm_losses = []
        best = {
            'avg_mae': 1,
            'epoch': 0,
            'train_step': 0
        }

        print('Start training from epoch: ', start_epoch)
        for e in range(start_epoch, epoch + 1):
            print('Epoch {} starts'.format(e))
            print()
            # set model in training mode
            self.modelTrain.train()
            self.modelTrain.model.train()
            epoch_loss = 0.0

            for idx, (src, tgt_cls, tgt_reg, data_idx) in enumerate(train_loader):

                src = src.to(self.device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)

                #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                task_loss, pred_cls, _ = self.modelTrain(src, tgt_cls, tgt_reg)
                #prof.export_chrome_trace('./checkpoint/{}/profiler_log_{}.json'.format(self.options.save_dir, idx))
                #print(prof)

                if self.options.loss_balancing == 'uncertainty':
                    loss = self.compute_uncertainty_loss(task_loss)
                    # loss = torch.sum(task_loss)
                else:
                    weighted_task_loss = self.modelTrain.task_weights * task_loss
                    loss = torch.sum(weighted_task_loss)

                if train_step == 0:
                    initial_task_loss = task_loss.data  # set L(0)
                    # print('init_task_loss', initial_task_loss)

                sum_loss += loss.item()
                epoch_loss += loss.item()
                losses.append(loss.item())

                # compute accuracy
                if pred_cls.nelement() != 0:
                    batch_train_accuracies, _, _ = compute_accuracy(pred_cls, tgt_cls, self.categorical)
                    train_accuracies += batch_train_accuracies

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # clear gradient of w_i(t) to update by GN loss
                if self.options.loss_balancing != 'uncertainty':
                    self.modelTrain.task_weights.grad.data.zero_()
                if self.options.loss_balancing == 'grad_norm':
                    # get layer of shared weights
                    W = self.modelTrain.get_last_shared_layer()

                    norms = []
                    for i in range(len(task_loss)):
                        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                        norms.append(torch.norm(self.modelTrain.task_weights[i] * gygw[0]))
                    norms = torch.stack(norms)

                    loss_ratio = task_loss / initial_task_loss
                    inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

                    # compute the mean norm \tilde{G}_w(t)
                    mean_norm = torch.mean(norms.data)

                    alpha = 1.0
                    # compute the GradNorm loss
                    # this term has to remain constant
                    constant_term = (mean_norm * (inverse_train_rate ** alpha))

                    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

                    # compute the gradient for the weights
                    self.modelTrain.task_weights.grad = \
                        torch.autograd.grad(grad_norm_loss, self.modelTrain.task_weights)[0]
                    # grad_norm_loss.backward()
                else:
                    grad_norm_loss = torch.tensor([0.0], device=self.device)
                    loss_ratio = torch.tensor([0] * len(task_loss), device=self.device)

                self.optimizer.step()

                # re-normalize
                normalize_coeff = self.modelTrain.task_count / torch.sum(self.modelTrain.task_weights.data, dim=0)
                self.modelTrain.task_weights.data = self.modelTrain.task_weights.data * normalize_coeff

                if train_step % self.options.report_frequency == 0:
                    avg_losses.append(np.mean(losses[-100:]))

                    # record
                    task_losses.append(task_loss.data.cpu().numpy())
                    loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                    if self.options.loss_balancing == 'uncertainty':
                        reg_weights = 0.5 * torch.exp(-self.modelTrain.log_sigma_reg)
                        cls_weights = torch.exp(-self.modelTrain.log_sigma_cls)
                        reg_weights = reg_weights.data.cpu().numpy()
                        cls_weights = cls_weights.data.cpu().numpy()
                        uncertainty_weights = np.concatenate((cls_weights, reg_weights))
                        weights.append(uncertainty_weights)
                    else:
                        weights.append(self.modelTrain.task_weights.data.cpu().numpy())
                    grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())

                    print('Step {:<7}: loss = {:.5f}, average loss = {:.5f}, task loss = {}, weights= {}, uncertainty '
                          'logvar: cls= {}, reg= {} '
                          .format(train_step,
                                  loss.item(),
                                  avg_losses[-1],
                                  " ".join(map("{:.5f}".format, task_loss.data.cpu().numpy())),
                                  " ".join(map("{:.5f}".format, self.modelTrain.task_weights.data.cpu().numpy())),
                                  " ".join(map("{:.5f}".format, self.modelTrain.log_sigma_cls.data.cpu().numpy())),
                                  " ".join(map("{:.5f}".format, self.modelTrain.log_sigma_reg.data.cpu().numpy()))
                                  ))

                    if self.visualizer is not None:
                        loss_window = self.visualizer.line(
                            X=np.arange(start_step, train_step + 1, self.options.report_frequency),
                            Y=avg_losses,
                            update='update' if loss_window else None,
                            win=loss_window,
                            opts={'title': "Training loss",
                                  'xlabel': "Step",
                                  'ylabel': "Loss"}
                        )

                        task_loss_window = self.visualizer.line(
                            X=np.arange(start_step, train_step + 1, self.options.report_frequency),
                            Y=task_losses,
                            update='update' if task_loss_window else None,
                            win=task_loss_window,
                            opts={'title': "Training task losses",
                                  'xlabel': "Step",
                                  'ylabel': "Loss",
                                  'legend': list(range(self.modelTrain.task_count))}
                        )

                        task_weights_window = self.visualizer.line(
                            X=np.arange(start_step, train_step + 1, self.options.report_frequency),
                            Y=weights,
                            update='update' if task_weights_window else None,
                            win=task_weights_window,
                            opts={'title': "Task weights",
                                  'xlabel': "Step",
                                  'ylabel': "Loss",
                                  'legend': list(range(self.modelTrain.task_count))}
                        )
                        if self.options.loss_balancing == 'grad_norm':
                            gradnorm_loss_window = self.visualizer.line(
                                X=np.arange(start_step, train_step + 1, self.options.report_frequency),
                                Y=grad_norm_losses,
                                update='update' if gradnorm_loss_window else None,
                                win=gradnorm_loss_window,
                                opts={'title': "Grad norm losses",
                                    'xlabel': "Step",
                                    'ylabel': "Loss"}
                            )

                train_step += 1

            epoch_loss = epoch_loss / len(train_loader)
            if self.options.loss_balancing == 'grad_norm':
                print('Epoch {:<3}: Total loss={:.5f}, gradNorm_loss={:.5f}, loss_ratio={}, weights={}, task_loss={}'
                      .format(e,
                              loss.item(),
                              grad_norm_loss.data.cpu().numpy(),
                              " ".join(map("{:.5f}".format, loss_ratio.data.cpu().numpy())),
                              " ".join(map("{:.5f}".format, self.modelTrain.task_weights.data.cpu().numpy())),
                              " ".join(map("{:.5f}".format, task_loss.data.cpu().numpy())))
                      )

            if not self.options.no_classification:
                train_accuracies = train_accuracies * 100 / len(train_loader.dataset)
                train_avg_accuracy = torch.mean(train_accuracies)
                accuracies = torch.cat((train_accuracies, train_avg_accuracy.view(1)))
                accuracy_legend = ['train_{}'.format(i) for i in range(len(train_accuracies))]
                accuracy_legend.append('train_avg')
                print('Average epoch loss={:.5f}, avg train accuracy={:.5f}, train accuracies={}'.format(
                    epoch_loss,
                    train_avg_accuracy,
                    train_accuracies
                ))

            metric = epoch_loss
            if val_loader is not None:
                val_loss, val_balanced_accuracies, val_avg_accuracy, val_accuracies, conf_matrices, avg_mae \
                    = self.validate(e, val_loader)
                print('val_avg_accuracy', val_avg_accuracy)
                print('val_accuracies', val_accuracies)
                if (val_avg_accuracy != 0.0):
                    print('Validation loss: {:.5f}, validation accuracy: {:.2f}%, task accuracies: {}'
                      .format(val_loss, val_avg_accuracy.data.cpu().numpy(), val_accuracies.data.cpu().numpy()))
                val_losses.append(val_loss)

                # print('--- Test set validation ---')
                # test_loss, _, test_avg_accuracy, test_accuracies, _, _ \
                #     = self.validate(e, test_loader)
                # print('Test validation loss: {:.5f}, test validation accuracy: {:.2f}%, task accuracies: {}'
                #       .format(test_loss, test_avg_accuracy.data.cpu().numpy(), test_accuracies.data.cpu().numpy()))

                if not self.options.no_classification:
                    accuracies = torch.cat((accuracies, val_accuracies, val_avg_accuracy.view(1)))
                    accuracy_legend = accuracy_legend + ['val_{}'.format(i) for i in range(len(val_accuracies))]
                    accuracy_legend.append('val_avg')
                # metric = val_loss
                # metric = -val_avg_accuracy
                metric = -np.mean(val_balanced_accuracies)

                if (e % self.save_every == 0 or e == 1 or e == self.options.epoch) \
                        and not self.options.no_classification and self.visualizer:
                    for i in range(len(conf_matrices)):
                        self.visualizer.heatmap(conf_matrices[i], opts={
                            'title': '{} at epoch {}'.format(label_names[i], e),
                            'xmax': 100
                        })

                # avg_balanced_acc = np.mean(val_balanced_accuracies)
                # if avg_mae < best['avg_mae']:
                #     best['avg_mae'] = avg_mae
                #     best['epoch'] = e
                #     best['train_step'] = train_step

                #     self.save_checkpoint(best['epoch'], best['train_step'], avg_mae, initial_task_loss)
                self.save_checkpoint(e, train_step, avg_mae, initial_task_loss)
            else:
                if e % self.save_every == 1 or e == self.options.epoch:
                    self.save_checkpoint(e, train_step, avg_mae, initial_task_loss)
            if not self.options.no_classification and self.visualizer:
                accuracy_list.append(accuracies.data.cpu().numpy())
                accuracy_window = self.visualizer.line(
                    X=np.arange(start_epoch, e + 1, 1),
                    Y=accuracy_list,
                    update='update' if accuracy_window else None,
                    win=accuracy_window,
                    opts={'title': "Training and Validation accuracies",
                          'xlabel': "Epoch",
                          'ylabel': "Accuracies",
                          'legend': accuracy_legend}
                )

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metric)
            elif self.scheduler is not None:
                # other scheduler types
                self.scheduler.step()

            # Get current learning rate. Is there any better way?
            lr = None
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] is not None:
                    lr = param_group['lr']
                    break
            print('Current learning rate at epoch {}: {}'.format(e, lr))
            print('===============================================')

    @staticmethod
    def multiclass_roc_auc_score(y_true, y_pred, average="macro"):
        lb = LabelBinarizer()

        lb.fit(y_true)
        y_true = lb.transform(y_true)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_true, y_pred, average=average)

    def validate(self, epoch, val_loader):
        # set model in validation mode
        self.modelTrain.eval()
        self.modelTrain.model.eval()

        sum_loss = 0
        pred_cls_logits = torch.tensor([], dtype=torch.float)
        tgt_cls_logits = torch.tensor([], dtype=torch.float)
        all_pred_reg = torch.tensor([], dtype=torch.float)  # on cpu
        all_tgt_reg = torch.tensor([], dtype=torch.float)  # on cpu
        data_indices = torch.tensor([], dtype=torch.long)

        for idx, (src, tgt_cls, tgt_reg, data_idx) in enumerate(val_loader):
            src = src.to(self.device, dtype=torch.float32)
            tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
            tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)
            data_indices = torch.cat((data_indices, data_idx), dim=0)

            with torch.no_grad():
                task_loss, batch_pred_cls, batch_pred_reg = self.modelTrain(src, tgt_cls, tgt_reg)

                weighted_task_loss = self.modelTrain.task_weights * task_loss
                loss = torch.sum(weighted_task_loss)

                sum_loss += loss.item()
                if not self.options.no_classification:
                    # concat batch predictions
                    pred_cls_logits = torch.cat((pred_cls_logits, batch_pred_cls.cpu()), dim=0)
                    tgt_cls_logits = torch.cat((tgt_cls_logits, tgt_cls.cpu()), dim=0)

                if not self.options.no_regression:
                    batch_pred_reg = batch_pred_reg.to(torch.device('cpu'))
                    tgt_reg = tgt_reg.to(torch.device('cpu'))
                    all_tgt_reg = torch.cat((all_tgt_reg, tgt_reg), dim=0)
                    all_pred_reg = torch.cat((all_pred_reg, batch_pred_reg), dim=0)

        average_loss = sum_loss / len(val_loader)

        val_balanced_accuracies, avg_accuracy, task_accuracies, conf_matrices = compute_cls_metrics(pred_cls_logits, tgt_cls_logits,
                                                                                   self.options,
                                                                                   self.categorical)

        absolute_errors = torch.abs(all_pred_reg - all_tgt_reg)
        mae_error_per_task = torch.mean(absolute_errors, dim=0)
        avg_mae = torch.mean(mae_error_per_task)

        compute_reg_metrics(val_loader, all_pred_reg, all_tgt_reg, epoch, self.options, self.metadata,
                            self.hyper_labels_reg, self.image_path, should_save=False, mode='validation')
        return average_loss, val_balanced_accuracies, avg_accuracy, task_accuracies, conf_matrices, avg_mae

    def save_checkpoint(self, epoch, train_step, avg_mae, initial_task_loss):
        """
        Saving model's state dict
        :param epoch: epoch at which model is saved
        :param train_step: train_step at which model is saved
        :param avg_mae: avg_mae achieved by current model
        :param initial_task_loss: tensor of first step's task loss
        :return:
        """
        state = {
            'epoch': epoch,
            'train_step': train_step,
            'initial_task_loss': initial_task_loss,
            'model': self.modelTrain.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'options': self.options
        }
        torch.save(state, '{}/model_e{}_{:.5f}.pt'.format(self.ckpt_path, epoch, avg_mae))
        print('Saved model at epoch {} at {}/model_e{}_{:.5f}.pt'.format(epoch, self.ckpt_path, epoch, avg_mae))
