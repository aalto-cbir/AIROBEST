import os

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(object):
    def __init__(self, model, optimizer, criterion_cls, criterion_reg, scheduler, device,
                 visualizer, metadata, options):
        self.model = model
        # self.modelTrain = ModelTrain(model, criterion_cls, criterion_reg, device)
        # self.task_weights = torch.nn.Parameter(torch.ones(model.task_count))
        # self.task_weights = torch.tensor([1.] * model.task_count, device=device, requires_grad=True)
        self.optimizer = optimizer
        self.criterion_cls = criterion_cls
        self.criterion_reg = criterion_reg
        self.device = device
        self.scheduler = scheduler
        self.options = options
        self.metadata = metadata
        self.visualizer = visualizer

    def train(self, train_loader, val_loader):
        epoch = self.options.epoch
        start_epoch = self.options.start_epoch + 1 if 'start_epoch' in self.options else 1
        save_every = 1  # specify number of epochs to save model
        train_step = 0
        sum_loss = 0.0
        avg_losses = []
        val_losses = []
        val_accuracies = []
        loss_window = None

        # set model in training mode
        self.model.train()
        # self.model.to(self.device)

        losses = []

        weights = []
        task_losses = []
        loss_ratios = []
        grad_norm_losses = []

        print('Start training from epoch: ', start_epoch)
        for e in range(start_epoch, epoch + 1):
            self.model.train()
            epoch_loss = 0.0

            for idx, (src, tgt_cls, tgt_reg) in enumerate(train_loader):
                src = src.to(self.device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)
                # tgt = tgt.to(device, dtype=torch.int64)

                # pred_cls, pred_reg = self.model(src)
                # loss_cls = self.criterion_cls(pred_cls, tgt_cls)
                # loss_reg = self.criterion_reg(pred_reg, tgt_reg)
                # task_loss = torch.tensor([loss_cls, loss_reg], dtype=torch.float, requires_grad=True)
                task_loss, _, _ = self.model(src, tgt_cls, tgt_reg)
                weighted_task_loss = self.model.task_weights * task_loss
                loss = torch.sum(weighted_task_loss)
                # loss = 1 * loss_cls + 3 * loss_reg

                if e == 1:
                    initial_task_loss = task_loss.data  # set L(0)
                    # print('init_task_loss', initial_task_loss)

                sum_loss += loss.item()
                epoch_loss += loss.item()
                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # clear gradient of w_i(t) to update by GN loss
                self.model.task_weights.grad.data.zero_()
                # print('Grad: ', self.model.task_weights.grad)
                self.options.use_gradnorm = True  # TODO: add config option
                if self.options.use_gradnorm:
                    # get layer of shared weights
                    W = self.model.get_last_shared_layer()

                    norms = []
                    for i in range(len(task_loss)):
                        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                        norms.append(torch.norm(self.model.task_weights[i] * gygw[0]))
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
                    self.model.task_weights.grad = torch.autograd.grad(grad_norm_loss, self.model.task_weights)[0]
                    # grad_norm_loss.backward()
                    # print('weight grad:', self.model.task_weights, self.model.task_weights.grad)

                self.optimizer.step()

                if train_step % self.options.report_frequency == 0:
                    avg_losses.append(np.mean(losses[-100:]))
                    print('Training loss at step {}: {:.5f}, average loss: {:.5f}, task loss: {}, weights: {}'
                          .format(train_step, loss.item(), avg_losses[-1], task_loss.data.cpu().numpy(), self.model.task_weights.data.cpu().numpy()))

                    if self.visualizer is not None:
                        loss_window = self.visualizer.line(
                            X=np.arange(0, train_step + 1, self.options.report_frequency),
                            Y=avg_losses,
                            update='update' if loss_window else None,
                            win=loss_window,
                            opts={'title': "Training loss", 'xlabel': "Step",
                                  'ylabel': "Loss"}
                        )

                train_step += 1

            # renormalize
            normalize_coeff = self.model.task_count / torch.sum(self.model.task_weights.data, dim=0)
            self.model.task_weights.data = self.model.task_weights.data * normalize_coeff

            # record
            if torch.cuda.is_available():
                task_losses.append(task_loss.data.cpu().numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(self.model.task_weights.data.cpu().numpy())
                grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
            else:
                task_losses.append(task_loss.data.numpy())
                loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
                weights.append(self.model.task_weights.data.numpy())
                grad_norm_losses.append(grad_norm_loss.data.numpy())

            epoch_loss = epoch_loss / len(train_loader)
            print('Epoch {}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}, total loss={}'.format(
                e, loss_ratio.data.cpu().numpy(), self.model.task_weights.data.cpu().numpy(),
                task_loss.data.cpu().numpy(), grad_norm_loss.data.cpu().numpy(), loss.item()))
            print('Average epoch loss: {:.5f}'.format(epoch_loss))
            metric = epoch_loss
            if val_loader is not None:
                val_loss, val_accuracy = self.validate(val_loader)
                print('Validation loss: {:.5f}, validation accuracy: {:.2f}%'.format(val_loss, val_accuracy))
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                # metric = val_loss
                metric = -val_accuracy

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
                self.save_checkpoint(e)

        task_losses = np.array(task_losses)
        weights = np.array(weights)

    def validate(self, val_loader):
        sum_loss = 0
        N_samples = 0
        sum_accuracy = 0
        for idx, (src, tgt_cls, tgt_reg) in enumerate(val_loader):
            src = src.to(self.device, dtype=torch.float32)
            tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
            tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)
            N_samples += len(src)

            with torch.no_grad():
                # pred_cls, pred_reg = self.model(src)
                # loss_cls = self.criterion_cls(pred_cls, tgt_cls)
                # loss_reg = self.criterion_reg(pred_reg, tgt_reg)
                # loss = 1 * loss_cls + 3 * loss_reg
                task_loss, pred_cls, pred_reg = self.model(src, tgt_cls, tgt_reg)
                weighted_task_loss = self.model.task_weights * task_loss
                loss = torch.sum(weighted_task_loss)

                sum_loss += loss.item()
                sum_accuracy += self.compute_accuracy(pred_cls, tgt_cls)

        # return average validation loss
        average_loss = sum_loss / len(val_loader)
        # accuracy = n_correct * 100 / N_samples
        accuracy = sum_accuracy * 100 / len(val_loader)
        return average_loss, accuracy

    def compute_accuracy(self, predict, tgt):
        """
        Return number of correct prediction of each tgt label
        :param predict:
        :param tgt:
        :return:
        """
        n_correct = 0  # vector or scalar?

        # reshape tensor in (*, n_cls) format
        # this is mainly for LeeModel that output the prediction for all pixels
        # from the source image with shape (batch, patch, patch, n_cls)
        n_cls = tgt.shape[-1]
        predict = predict.view(-1, n_cls)
        tgt = tgt.view(-1, n_cls)
        #####

        categorical = self.metadata['categorical']
        num_classes = 0
        for idx, values in categorical.items():
            count = len(values)
            pred_class = predict[:, num_classes:(num_classes + count)]
            tgt_class = tgt[:, num_classes:(num_classes + count)]
            pred_indices = pred_class.argmax(-1)  # get indices of max values in each row
            tgt_indices = tgt_class.argmax(-1)
            true_positive = torch.sum(pred_indices == tgt_indices).item()
            n_correct += true_positive
            num_classes += count

        # return n_correct divided by number of labels * batch_size
        return n_correct / (len(predict) * len(categorical.keys()))

    def save_checkpoint(self, epoch):
        """
        Saving model's state dict
        :param epoch: the epoch when model is saved
        :return:
        """
        save_dir = self.options.save_dir or self.options.model
        path = './checkpoint/{}'.format(save_dir)
        if not os.path.exists(path):
            os.makedirs(path)

        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'options': self.options
        }
        torch.save(state, '{}/{}_{}.pt'.format(path, save_dir, epoch))
        print('Saved model at epoch %d' % epoch)
