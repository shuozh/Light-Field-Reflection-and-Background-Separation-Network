import warnings
from torch.optim.lr_scheduler import _LRScheduler
EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

class myLR(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1):
        super(myLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)
        if self.last_epoch >= 200:
            if self.last_epoch% 100 == 0:
                return [group['lr'] / 10
                        for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


    def _get_closed_form_lr(self):
        if self.last_epoch >= 200:
            if self.last_epoch% 100 == 0:
                return [base_lr / 10 for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)




