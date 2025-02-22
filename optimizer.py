import numpy as np
import math
import hparams as hp
from torch.optim.lr_scheduler import _LRScheduler

class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps, total_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.current_epoch = current_steps
        self.n_total_steps = total_steps//hp.accumulate_steps
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    # Transformer의 구현 방식
    def _get_lr_scale(self):
        return np.min([
            np.power(self.current_epoch, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.current_epoch])

    # # 예시: cosine decay with warmup
    # def _get_lr_scale(self):
    #     if self.current_epoch < self.n_warmup_steps:
    #         return self.current_epoch / max(1, self.n_warmup_steps)
    #     else:
    #         return 0.5*max(0.0, 1 + math.cos(math.pi * (self.current_epoch - self.n_warmup_steps) / max(1, self.n_total_steps - self.n_warmup_steps)))
        
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.current_epoch += 1
        lr = self.init_lr * self._get_lr_scale()    # learning rate이 warm-up 때 hp.learning_rate을 초과하지 않도록 함
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


# class ScheduledOptim(_LRScheduler):
#     ''' A simple wrapper class for learning rate scheduling '''
#     def __init__(self, optimizer, last_epoch=-1):        
#         self._optimizer = optimizer
#         self.T_0 = hp.T_0  # 한 주기 epoch (맨 처음 주기, 변동 X)
#         self.T_mult = hp.T_mult    # 한 주기의 크기를 얼마만큼 늘려나갈 것인지 factor
#         self.base_eta_max = hp.learning_rate
#         self.eta_max = hp.learning_rate    # lr 최댓값
#         self.warmup_epochs = hp.warmup_steps    # warm up steps
#         self.T_i = hp.T_0  # 한 주기 step (계속 변하는 주기, 변동 O)
#         self.gamma = hp.gamma  # lr 최댓값을 cycle마다 얼마나 줄일 것인지 factor
#         self.cycle = 0
#         self.current_epoch = last_epoch
#         super(ScheduledOptim, self).__init__(optimizer, last_epoch)
        
#     def step_and_update_lr(self, epoch):
#         self._update_learning_rate(epoch)
#         self._optimizer.step()

#     def zero_grad(self):
#         self._optimizer.zero_grad()

#     # 예시: cosine decay with warmup
#     # base lr은 업데이트 직전 lr
#     def _get_lr(self):
#         if self.current_epoch < self.warmup_epochs:
#             return [base_lr + (self.eta_max-base_lr)*self.current_epoch / self.warmup_epochs for base_lr in self.base_lrs]
#         else:
#             return [base_lr + (self.eta_max-base_lr)*(1+math.cos(math.pi*(self.current_epoch-self.warmup_epochs)/(self.T_i-self.warmup_epochs)))/2 for base_lr in self.base_lrs]
        
#     def _update_learning_rate(self, epoch):
#         ''' Learning rate scheduling per step '''
#         self.current_epoch += 1 # [0, T0] 범위의 상대 epoch. 인자로 받는 epoch은 절대 epoch
#         if self.current_epoch >= self.T_i:    # 한 cycle이 끝났으니 cycle 주기 변경
#             self.cycle += 1
#             self.current_epoch -= self.T_i  # 주기 안의 값으로 정규화
#             self.T_i = (self.T_i - self.warmup_epochs) * self.T_mult + self.warmup_epochs

#         if epoch >= self.T_0:
#             if self.T_mult == 1:
#                 self.current_epoch = epoch % self.T_0
#                 self.cycle = epoch // self.T_0
#             else:
#                 n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
#                 self.cycle = n
#                 self.current_epoch = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
#                 self.T_i = self.T_0 * self.T_mult ** n
#         else:
#             self.T_i = self.T_0
#             self.current_epoch = epoch
                
#         self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
#         self.last_epoch = math.floor(epoch)
#         for param_group , lr in zip(self._optimizer.param_groups, self._get_lr()):
#             param_group['lr'] = lr