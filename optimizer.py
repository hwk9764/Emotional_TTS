import numpy as np
import math
import hparams as hp

class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling '''

    def __init__(self, optimizer, d_model, n_warmup_steps, current_steps, total_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.n_total_steps = total_steps//hp.accumulate_steps
        self.init_lr = hp.learning_rate #np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    # Transformer의 구현 방식
    '''def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])'''

    # 예시: cosine decay with warmup
    def _get_lr_scale(self):
        if self.n_current_steps < self.n_warmup_steps:
            return self.n_current_steps / max(1, self.n_warmup_steps)
        else:
            return 0.5*max(0.0, 1 + math.cos(math.pi * (self.n_current_steps - self.n_warmup_steps) / max(1, self.n_total_steps - self.n_warmup_steps)))
        
    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()    # learning rate이 warm-up 때 hp.learning_rate을 초과하지 않도록 함
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
