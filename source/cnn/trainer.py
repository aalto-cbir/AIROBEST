import os

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(object):
    def __init__(self, model, optimizer, criterion_cls, criterion_reg, scheduler, device,
                 visualizer, metadata, options):
        self.model = model
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
        self.model.to(self.device)

        losses = []

        print('Start training from epoch: ', start_epoch)
        for e in range(start_epoch, epoch + 1):
            self.model.train()
            epoch_loss = 0.0

            for idx, (src, tgt_cls, tgt_reg) in enumerate(train_loader):
                src = src.to(self.device, dtype=torch.float32)
                tgt_cls = tgt_cls.to(self.device, dtype=torch.float32)
                tgt_reg = tgt_reg.to(self.device, dtype=torch.float32)
                # tgt = tgt.to(device, dtype=torch.int64)

                self.optimizer.zero_grad()
                pred_cls, pred_reg = self.model(src)
                loss_cls = self.criterion_cls(pred_cls, tgt_cls)
                loss_reg = self.criterion_reg(pred_reg, tgt_reg)
                loss = 1 * loss_cls + 3 * loss_reg

                sum_loss += loss.item()
                epoch_loss += loss.item()
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()

                if train_step % self.options.report_frequency == 0:
                    avg_losses.append(np.mean(losses[-100:]))
                    print('Training loss at step {}: {:.5f}, average loss: {:.5f}, cls_loss: {:.5f}, reg_loss: {:.5f}'
                          .format(train_step, loss.item(), avg_losses[-1], loss_cls.item(), loss_reg.item()))
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

            epoch_loss = epoch_loss / len(train_loader)
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
                pred_cls, pred_reg = self.model(src)
                loss_cls = self.criterion_cls(pred_cls, tgt_cls)
                loss_reg = self.criterion_reg(pred_reg, tgt_reg)
                loss = 1 * loss_cls + 3 * loss_reg

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
