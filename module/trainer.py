import pickle
from time import time
from itertools import combinations

import cupy as np
import matplotlib.pyplot as plt

from module.functions import *

class SingleTrainer :
    def __init__(self,
                 config,
                 model,
                 optimizer) :
        self.config = config
        self.model = model
        self.optimizer = optimizer

        self.best_loss = np.inf
        self.best_epoch = None
        self.train_loss_list = []
        self.valid_loss_list = []

    def load_params(self, load_fn, src_path='drive/MyDrive/deeplearning/') :
        with open(src_path + load_fn, 'rb') as f :
            params = pickle.load(f)

        for i, p in enumerate(params['params']) :
            self.model.params[i][...] = p

    def weight_tying(self, params, grads) :
        params, grads = params[:], grads[:]

        while True :
            length = len(params)
            for a, b in combinations(np.arange(length - 1), 2) :
                a, b = int(a), int(b)
                if params[a].shape == params[b].shape :
                    if params[a] is params[b] :
                        grads[a] += grads[b]
                        params.pop(b)
                        grads.pop(b)
                        break
                elif params[a].shape == params[b].T.shape :
                    if np.all(params[a] == params[b].T) :
                        grads[a] += grads[b].T
                        params.pop(b)
                        grads.pop(b)
                        break
            else :
                break

        return params, grads

    def train(self, x, y, valid_data=None, shuffle=True) :
        data_size = len(x)
        max_iters = data_size // self.config.batch_size
        total_train_loss = 0
        loss_cnt = 0
        valid_loss_cnt = 0

        if valid_data is not None :
            valid_X, valid_y = valid_data
            valid_max_iters = len(valid_X) // self.config.valid_batch_size
            total_valid_loss = 0

        start = time.time()
        stop_cnt = 0
        if self.config.early_stopping > 0 :
            print(
                '(Message) The training will be automatically stopped when best score not updated during {} epochs'.format(
                    self.config.early_stopping))

        for epoch in range(self.config.epochs) :

            if (self.config.warmup_epochs > 0) and (epoch + 1 <= self.config.warmup_epochs) :
                self.optimizer.lr = self.config.lr * ((epoch + 1) / self.config.warmup_epochs)
                print(f'warmup stage - current lr: {self.optimizer.lr}')
            stop_cnt += 1

            if stop_cnt > self.config.early_stopping > 0 :
                print('(Message) No improvement during {} epochs, training stopped'.format(self.config.early_stopping))
                return

            if shuffle :
                idx = np.random.permutation(data_size)
                x = x[idx]
                y = y[idx]

            for iters in range(max_iters) :
                batch_x = x[iters * self.config.batch_size :(iters + 1) * self.config.batch_size]
                batch_y = y[iters * self.config.batch_size :(iters + 1) * self.config.batch_size]
                train_loss = self.model.forward(batch_x, batch_y)
                self.model.backward()
                params, grads = self.weight_tying(self.model.params, self.model.grads)
                clip_grads(grads, self.config.max_grad_norm)
                self.optimizer.update(params, grads)

                p_norm, g_norm = get_norm(params, grads)

                total_train_loss += train_loss
                loss_cnt += 1

            if valid_data is not None :
                for iters in range(valid_max_iters) :
                    valid_batch_x = valid_X[
                                    iters * self.config.valid_batch_size :(iters + 1) * self.config.valid_batch_size]
                    valid_batch_y = valid_y[
                                    iters * self.config.valid_batch_size :(iters + 1) * self.config.valid_batch_size]
                    valid_loss = self.model.forward(valid_batch_x, valid_batch_y)
                    total_valid_loss += valid_loss
                    valid_loss_cnt += 1

            end = time.time()
            if self.config.verbose > 0 :
                if (epoch + 1) % self.config.verbose == 0 :
                    if valid_data is None :
                        avg_train_loss = total_train_loss / loss_cnt

                        if avg_train_loss < self.best_loss :
                            stop_cnt = 0
                            self.best_loss = avg_train_loss
                            self.best_epoch = epoch + 1
                        else :
                            if self.config.lr_decay > 0 :
                                self.optimizer.lr = self.optimizer.lr * self.config.lr_decay

                        print(
                            '| EPOCH ({} / {}) |  train_loss={:.4f}  best_loss={:.4f}  |param|={:.2e}  |grad|={:.2e} ({:.2f}sec)'.format(
                                epoch + 1, self.config.epochs, avg_train_loss, self.best_loss, p_norm, g_norm,
                                end - start
                            ))
                        self.train_loss_list.append(avg_train_loss)
                        total_train_loss = 0
                        loss_cnt = 0
                    else :
                        avg_train_loss = total_train_loss / loss_cnt
                        avg_valid_loss = total_valid_loss / valid_loss_cnt

                        if avg_valid_loss < self.best_loss :
                            stop_cnt = 0
                            self.best_loss = avg_valid_loss
                            self.best_epoch = epoch + 1
                        else :
                            if self.config.lr_decay > 0 :
                                self.optimizer.lr = self.optimizer.lr * self.config.lr_decay

                        print(
                            '| EPOCH ({} / {}) |  train_loss={:.4f}  valid_loss={:.4f}  best_loss={:.4f}  |param|={:.2e}  |grad|={:.2e} ({:.2f}sec)'.format(
                                epoch + 1, self.config.epochs, avg_train_loss, avg_valid_loss, self.best_loss, p_norm,
                                g_norm, end - start
                            ))
                        self.train_loss_list.append(avg_train_loss)
                        self.valid_loss_list.append(avg_valid_loss)
                        total_train_loss, total_valid_loss = 0, 0
                        loss_cnt, valid_loss_cnt = 0, 0

    def print_result(self) :
        print()
        print('=' * 10, 'Result', '=' * 10)
        print('Best loss', self.best_loss)
        print('Best epoch', self.best_epoch)

    def plot(self, ylim=None) :
        x = np.arange(len(self.train_loss_list))
        if ylim is not None :
            plt.ylim(*ylim)
        plt.plot(x, self.train_loss_list, label='Train loss')
        if self.valid_loss_list :
            plt.plot(x, self.valid_loss_list, label='Valid loss')
        plt.title('Training result')
        plt.xlabel('Epochs'.format(self.config.verbose))
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
