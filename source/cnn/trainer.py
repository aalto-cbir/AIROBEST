import os

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(object):
    def __init__(self, model, optimizer, criterion_cls, criterion_reg, scheduler, device,
                 visualizer, metadata, options, checkpoint=None):
        self.modelTrain = model
        self.optimizer = optimizer
        self.criterion_cls = criterion_cls
        self.criterion_reg = criterion_reg
        self.device = device
        self.scheduler = scheduler
        self.options = options
        self.metadata = metadata
        self.categorical = metadata['categorical']
        self.visualizer = visualizer
        self.checkpoint = checkpoint

    def train(self, train_loader, val_loader):
        epoch = self.options.epoch
        if self.checkpoint:
            start_epoch = self.checkpoint.epoch + 1
            train_step = self.checkpoint.train_step + 1
            initial_task_loss = self.checkpoint.initial_task_loss
        else:
            start_epoch = 1
            train_step = 0
        save_every = 1  # specify number of epochs to save model
        sum_loss = 0.0
        avg_losses = []
        val_losses = []
        val_accuracies = []
        loss_window = None
        task_loss_window = None
        task_weights_window = None
        gradnorm_loss_window = None

        # set model in training mode
        self.modelTrain.train()

        losses = []

        weights = []
        task_losses = []
        loss_ratios = []
        grad_norm_losses = []

        print('Start training from epoch: ', start_epoch)
        for e in range(start_epoch, epoch + 1):
            epoch_loss = 0.0

            for idx, (src, tgt_cls, tgt_reg) in enumerate(train_loader):
                src = src.to(self.device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)

                task_loss, _, _ = self.modelTrain(src, tgt_cls, tgt_reg)
                weighted_task_loss = self.modelTrain.task_weights * task_loss
                loss = torch.sum(weighted_task_loss)

                if train_step == 0:
                    initial_task_loss = task_loss.data  # set L(0)
                    # print('init_task_loss', initial_task_loss)

                sum_loss += loss.item()
                epoch_loss += loss.item()
                losses.append(loss.item())

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
                    self.modelTrain.task_weights.grad = torch.autograd.grad(grad_norm_loss, self.modelTrain.task_weights)[0]
                    # grad_norm_loss.backward()
                else:
                    grad_norm_loss = torch.Tensor([0.0], device=self.device)
                    loss_ratio = torch.Tensor([0] * len(task_loss), device=self.device)

                self.optimizer.step()

                # renormalize
                normalize_coeff = self.modelTrain.task_count / torch.sum(self.modelTrain.task_weights.data, dim=0)
                self.modelTrain.task_weights.data = self.modelTrain.task_weights.data * normalize_coeff

                if train_step % self.options.report_frequency == 0:
                    avg_losses.append(np.mean(losses[-100:]))

                    # record
                    if torch.cuda.is_available():
                        task_losses.append(task_loss.data.cpu().numpy())
                        loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                        weights.append(self.modelTrain.task_weights.data.cpu().numpy())
                        grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
                    else:
                        task_losses.append(task_loss.data.numpy())
                        loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                        weights.append(self.modelTrain.task_weights.data.numpy())
                        grad_norm_losses.append(grad_norm_loss.data.numpy())

                    print('Step {:<7}: loss = {:.5f}, average loss = {:.5f}, task loss = {}, weights= {}'
                          .format(train_step,
                                  loss.item(),
                                  avg_losses[-1],
                                  " ".join(map("{:.5f}".format, task_loss.data.cpu().numpy())),
                                  " ".join(map("{:.5f}".format, self.modelTrain.task_weights.data.cpu().numpy()))))

                    if self.visualizer is not None:
                        loss_window = self.visualizer.line(
                            X=np.arange(0, train_step + 1, self.options.report_frequency),
                            Y=avg_losses,
                            update='update' if loss_window else None,
                            win=loss_window,
                            opts={'title': "Training loss",
                                  'xlabel': "Step",
                                  'ylabel': "Loss"}
                        )

                        task_loss_window = self.visualizer.line(
                            X=np.arange(0, train_step + 1, self.options.report_frequency),
                            Y=task_losses,
                            update='update' if task_loss_window else None,
                            win=task_loss_window,
                            opts={'title': "Training task losses",
                                  'xlabel': "Step",
                                  'ylabel': "Loss",
                                  'legend': list(range(self.modelTrain.task_count))}
                        )

                        task_weights_window = self.visualizer.line(
                            X=np.arange(0, train_step + 1, self.options.report_frequency),
                            Y=weights,
                            update='update' if task_weights_window else None,
                            win=task_weights_window,
                            opts={'title': "Task weights",
                                  'xlabel': "Step",
                                  'ylabel': "Loss",
                                  'legend': list(range(self.modelTrain.task_count))}
                        )

                        gradnorm_loss_window = self.visualizer.line(
                            X=np.arange(0, train_step + 1, self.options.report_frequency),
                            Y=grad_norm_losses,
                            update='update' if gradnorm_loss_window else None,
                            win=gradnorm_loss_window,
                            opts={'title': "Grad norm losses",
                                  'xlabel': "Step",
                                  'ylabel': "Loss"}
                        )

                train_step += 1

            epoch_loss = epoch_loss / len(train_loader)
            print('Epoch {:<3}: GradNorm_loss={:.5f}, total loss={:.5f}, loss_ratio={}, weights={}, task_loss={}'.format(
                e,
                grad_norm_loss.data.cpu().numpy(),
                loss.item(),
                " ".join(map("{:.5f}".format, loss_ratio.data.cpu().numpy())),
                " ".join(map("{:.5f}".format, self.modelTrain.task_weights.data.cpu().numpy())),
                " ".join(map("{:.5f}".format, task_loss.data.cpu().numpy())))
            )
            print('Average epoch loss: {:.5f}'.format(epoch_loss))
            metric = epoch_loss
            if val_loader is not None:
                val_loss, val_avg_accuracy, val_cls_accuracies = self.validate(val_loader)
                print('Validation loss: {:.5f}, validation accuracy: {:.2f}%, task accuracies: {}'
                      .format(val_loss, val_avg_accuracy.data.numpy(), val_cls_accuracies.data.numpy()))
                val_losses.append(val_loss)
                val_accuracies.append(val_avg_accuracy)
                # metric = val_loss
                metric = -val_avg_accuracy

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
            print('Current learning rate: {}'.format(lr))
            if e % save_every == 0:
                self.save_checkpoint(e, train_step, initial_task_loss)

    def validate(self, val_loader):
        sum_loss = 0
        N_samples = 0
        sum_accuracy = torch.Tensor([0.0] * len(self.categorical))
        for idx, (src, tgt_cls, tgt_reg) in enumerate(val_loader):
            src = src.to(self.device, dtype=torch.float32)
            tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
            tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)
            N_samples += len(src)

            with torch.no_grad():
                task_loss, pred_cls, pred_reg = self.modelTrain(src, tgt_cls, tgt_reg)
                weighted_task_loss = self.modelTrain.task_weights * task_loss
                loss = torch.sum(weighted_task_loss)

                sum_loss += loss.item()
                sum_accuracy += self.compute_accuracy(pred_cls, tgt_cls)

        # return average validation loss
        average_loss = sum_loss / len(val_loader)
        accuracies = sum_accuracy * 100 / len(val_loader.dataset)
        avg_accuracy = torch.mean(accuracies)
        return average_loss, avg_accuracy, accuracies

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
        n_cls = tgt.shape[-1]
        predict = predict.view(-1, n_cls)
        tgt = tgt.view(-1, n_cls)
        #####

        n_correct = torch.Tensor([0.0] * len(self.categorical))
        num_classes = 0
        for idx, (key, values) in enumerate(self.categorical.items()):
            count = len(values)
            pred_class = predict[:, num_classes:(num_classes + count)]
            tgt_class = tgt[:, num_classes:(num_classes + count)]
            pred_indices = pred_class.argmax(-1)  # get indices of max values in each row
            tgt_indices = tgt_class.argmax(-1)
            true_positive = torch.sum(pred_indices == tgt_indices).item()
            n_correct[idx] += true_positive
            num_classes += count

        return n_correct

    def save_checkpoint(self, epoch, train_step, initial_task_loss):
        """
        Saving model's state dict
        :param initial_task_loss: tensor of first step's task loss
        :param train_step: the last training step when model is saved
        :param epoch: the epoch when model is saved
        :return:
        """
        save_dir = self.options.save_dir or self.options.model
        path = './checkpoint/{}'.format(save_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        state = {
            'epoch': epoch,
            'train_step': train_step,
            'initial_task_loss': initial_task_loss,
            'model': self.modelTrain.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'options': self.options
        }
        torch.save(state, '{}/{}_{}.pt'.format(path, save_dir, epoch))
        print('Saved model at epoch %d' % epoch)
