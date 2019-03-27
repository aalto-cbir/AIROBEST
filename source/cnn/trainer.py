import os
import sys

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from input.utils import plot_largest_error_patches, plot_error_histogram, plot_pred_vs_target, \
    export_error_points

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from tools.hypdatatools_img import get_geotrans


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
        self.hypGt = get_geotrans(self.options.hyper_data_header)
        self.save_every = 10  # specify number of epochs to save model

        save_dir = self.options.save_dir or self.options.model
        self.ckpt_path = './checkpoint/{}'.format(save_dir)
        self.image_path = './checkpoint/{}/images'.format(save_dir)
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

    def train(self, train_loader, val_loader):
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

        print('Start training from epoch: ', start_epoch)
        for e in range(start_epoch, epoch + 1):
            # set model in training mode
            self.modelTrain.train()
            self.modelTrain.model.train()
            epoch_loss = 0.0

            for idx, (src, tgt_cls, tgt_reg, data_idx) in enumerate(train_loader):
                src = src.to(self.device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)

                task_loss, pred_cls, _ = self.modelTrain(src, tgt_cls, tgt_reg)
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
                    batch_train_accuracies, _, _ = self.compute_accuracy(pred_cls, tgt_cls)
                    train_accuracies += batch_train_accuracies

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # clear gradient of w_i(t) to update by GN loss
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

                    alpha = 0.16
                    # compute the GradNorm loss
                    # this term has to remain constant
                    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)

                    grad_norm_loss = torch.tensor(torch.sum(torch.abs(norms - constant_term)))

                    # compute the gradient for the weights
                    self.modelTrain.task_weights.grad = \
                    torch.autograd.grad(grad_norm_loss, self.modelTrain.task_weights)[0]
                    # grad_norm_loss.backward()
                else:
                    grad_norm_loss = torch.tensor([0.0], device=self.device)
                    loss_ratio = torch.tensor([0] * len(task_loss), device=self.device)

                self.optimizer.step()

                # renormalize
                normalize_coeff = self.modelTrain.task_count / torch.sum(self.modelTrain.task_weights.data, dim=0)
                self.modelTrain.task_weights.data = self.modelTrain.task_weights.data * normalize_coeff

                if train_step % self.options.report_frequency == 0:
                    avg_losses.append(np.mean(losses[-100:]))

                    # record
                    task_losses.append(task_loss.data.cpu().numpy())
                    loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                    weights.append(self.modelTrain.task_weights.data.cpu().numpy())
                    grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())

                    print('Step {:<7}: loss = {:.5f}, average loss = {:.5f}, task loss = {}, weights= {}'
                          .format(train_step,
                                  loss.item(),
                                  avg_losses[-1],
                                  " ".join(map("{:.5f}".format, task_loss.data.cpu().numpy())),
                                  " ".join(map("{:.5f}".format, self.modelTrain.task_weights.data.cpu().numpy()))))

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

            if e % self.save_every == 0 or e == self.options.epoch:
                self.save_checkpoint(e, train_step, initial_task_loss)
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
                val_loss, val_avg_accuracy, val_accuracies, conf_matrices = self.validate(e, val_loader)
                print('Validation loss: {:.5f}, validation accuracy: {:.2f}%, task accuracies: {}'
                      .format(val_loss, val_avg_accuracy.data.cpu().numpy(), val_accuracies.data.cpu().numpy()))
                val_losses.append(val_loss)
                if not self.options.no_classification:
                    accuracies = torch.cat((accuracies, val_accuracies, val_avg_accuracy.view(1)))
                    accuracy_legend = accuracy_legend + ['val_{}'.format(i) for i in range(len(val_accuracies))]
                    accuracy_legend.append('val_avg')
                metric = val_loss
                # metric = -val_avg_accuracy

                if (e % self.save_every == 0 or e == 1 or e == self.options.epoch) \
                        and not self.options.no_classification and self.visualizer:
                    for i in range(len(conf_matrices)):
                        self.visualizer.heatmap(conf_matrices[i], opts={
                            'title': '{} at epoch {}'.format(label_names[i], e),
                            'xmax': 100
                        })

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

    def validate(self, epoch, val_loader):
        # set model in validation mode
        self.modelTrain.eval()
        self.modelTrain.model.eval()

        sum_loss = 0
        N_samples = len(val_loader.dataset)
        val_accuracies = torch.tensor([0.0] * len(self.categorical))  # treat all class uniformly
        avg_accuracy = torch.tensor(0.0)
        pred_cls_indices = torch.tensor([], dtype=torch.long)
        tgt_cls_indices = torch.tensor([], dtype=torch.long)
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
                    batch_accuracies, batch_pred_indices, batch_tgt_indices = self.compute_accuracy(batch_pred_cls, tgt_cls)
                    # batch_pred_indices = batch_pred_indices.cpu()
                    # batch_tgt_indices = batch_tgt_indices.cpu()
                    val_accuracies += batch_accuracies
                    # concat batch predictions
                    pred_cls_indices = torch.cat((pred_cls_indices, batch_pred_indices), dim=0)
                    tgt_cls_indices = torch.cat((tgt_cls_indices, batch_tgt_indices), dim=0)

                if not self.options.no_regression:
                    batch_pred_reg = batch_pred_reg.to(torch.device('cpu'))
                    tgt_reg = tgt_reg.to(torch.device('cpu'))
                    all_tgt_reg = torch.cat((all_tgt_reg, tgt_reg), dim=0)
                    all_pred_reg = torch.cat((all_pred_reg, batch_pred_reg), dim=0)

        # return average validation loss

        average_loss = sum_loss / len(val_loader)
        conf_matrices = []
        if not self.options.no_classification:
            val_accuracies = val_accuracies * 100 / N_samples
            avg_accuracy = torch.mean(val_accuracies)

            print('--Metrics--')
            val_balanced_accuracies = []
            for i in range(tgt_cls_indices.shape[-1]):
                conf_matrix = confusion_matrix(tgt_cls_indices[:, i], pred_cls_indices[:, i])
                # convert to percentage along rows
                conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
                conf_matrix = np.around(100 * conf_matrix, decimals=2)
                conf_matrices.append(conf_matrix)
                label_accuracy = np.around(np.mean(conf_matrix.diagonal()), decimals=2)
                val_balanced_accuracies.append(label_accuracy)
                print('---Task %s---' % i)
                precision = precision_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
                recall = recall_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
                f1 = f1_score(tgt_cls_indices[:, i], pred_cls_indices[:, i], average='weighted')
                print('Precision:', precision)
                print('Recall:', recall)
                print('F1 score', f1)

            avg_balanced_accuracy = np.around(np.mean(val_balanced_accuracies), decimals=2)
            print('Average balanced accuracy: %s, label accuracies: %s' % (avg_balanced_accuracy, val_balanced_accuracies))
            print('-------------')
        # scatter plot prediction vs target labels
        if not self.options.no_regression:
            if epoch % self.save_every == 0 or epoch == 1 or epoch == self.options.epoch:
                n_reg = self.modelTrain.model.n_reg
                coords = np.array(val_loader.dataset.coords)

                cmap = plt.get_cmap('viridis')
                colors = [cmap(i) for i in np.linspace(0, 1, n_reg)]
                names = self.metadata['reg_label_names']

                rmsq_errors = torch.abs(all_pred_reg - all_tgt_reg)

                sum_errors = torch.sum(rmsq_errors, dim=1)
                plot_error_histogram(sum_errors, 100, 'all_tasks', epoch, self.image_path)

                k = N_samples // 10  # 10% of the largest error
                value, indices = torch.topk(sum_errors, k, dim=0, largest=True, sorted=False)

                topk_points = coords[indices]
                task_label = self.hyper_labels_reg[:, :, 0]
                # chose the first task label just for visualization
                plot_largest_error_patches(task_label, topk_points, val_loader.dataset.patch_size,
                                           'all_tasks', self.image_path, epoch)

                export_error_points(coords, rmsq_errors, self.hypGt, sum_errors, names, epoch, self.ckpt_path)

                for i in range(n_reg):
                    x, y = all_tgt_reg[:, i], all_pred_reg[:, i]

                    plot_pred_vs_target(x, y, colors[i], names[i], self.image_path, epoch)

                    # plot error histogram
                    mse_errors = torch.abs(x - y)
                    plot_error_histogram(mse_errors, 100, names[i], epoch, self.image_path)

                    # plot top k largest errors on the map
                    task_label = self.hyper_labels_reg[:, :, i]
                    value, indices = torch.topk(mse_errors, k, dim=0, largest=True, sorted=False)

                    topk_points = coords[indices]
                    plot_largest_error_patches(task_label, topk_points, val_loader.dataset.patch_size,
                                               names[i], self.image_path, epoch)

        return average_loss, avg_accuracy, val_accuracies, conf_matrices

    def compute_accuracy(self, predict, tgt):
        """
        Return number of correct prediction of each tgt label
        :param predict: tensor of predicted outputs
        :param tgt: tensor of ground truth labels
        :return: number of correct predictions for every single classification task
        """

        # reshape tensor in (*, n_cls) format
        # this is mainly for LeeModel that output the prediction for all pixels
        # from the source image with shape (batch, patch, patch, n_cls)
        predict = predict.cpu()
        tgt = tgt.cpu()

        n_cls = tgt.shape[-1]
        predict = predict.view(-1, n_cls)
        tgt = tgt.view(-1, n_cls)
        #####
        pred_indices = []
        tgt_indices = []
        n_correct = torch.tensor([0.0] * len(self.categorical))
        num_classes = 0
        for idx, (key, values) in enumerate(self.categorical.items()):
            count = len(values)
            pred_class = predict[:, num_classes:(num_classes + count)]
            tgt_class = tgt[:, num_classes:(num_classes + count)]
            pred_index = pred_class.argmax(-1)  # get indices of max values in each row
            tgt_index = tgt_class.argmax(-1)
            pred_indices.append(pred_index)
            tgt_indices.append(tgt_index)
            true_positive = torch.sum(pred_index == tgt_index).item()
            n_correct[idx] += true_positive
            num_classes += count

        pred_indices = torch.stack(pred_indices, dim=1)
        tgt_indices = torch.stack(tgt_indices, dim=1)
        return n_correct, pred_indices, tgt_indices

    def save_checkpoint(self, epoch, train_step, initial_task_loss):
        """
        Saving model's state dict
        :param initial_task_loss: tensor of first step's task loss
        :param train_step: the last training step when model is saved
        :param epoch: the epoch when model is saved
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
        torch.save(state, '{}/model_{}.pt'.format(self.ckpt_path, epoch))
        print('Saved model at epoch %d' % epoch)
