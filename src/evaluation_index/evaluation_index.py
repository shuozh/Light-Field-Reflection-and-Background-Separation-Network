from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np


def compare_ncc(x, y):
    return np.mean((x-np.mean(x)) * (y-np.mean(y))) / (np.std(x) * np.std(y))


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate**2)
    else:
        alpha = 0.
    return np.sum((correct - alpha*estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i+window_size, j:j+window_size, c]
                estimate_curr = estimate[i:i+window_size, j:j+window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr**2)
    # assert np.isnan(ssq/total)
    return ssq / total


def quality_assess(pred, gt):
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1)
    ssim = structural_similarity(gt, pred, data_range=1, multichannel=True)
    lmse = local_error(gt, pred, 20, 10)
    ncc = compare_ncc(gt, pred)
    return psnr,ssim,lmse,ncc


class AverageMeter(object):
    '''Compute and stores the average and current value'''

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmstr.format(**self.__dict__)

class AverageMeterJustAVG(object):
    '''Compute and stores the average and current value'''

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmstr = '{name} {avg' + self.fmt + '}'
        return fmstr.format(**self.__dict__)

class AverageMeterJustValue(object):
    '''Compute and stores the average and current value'''

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmstr = '{name} {val' + self.fmt + '}'
        return fmstr.format(**self.__dict__)

class Meter(object):
    '''Compute and stores  value'''
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmstr = '{name} {val' + self.fmt + '}'
        return fmstr.format(**self.__dict__)

class StrMeter(object):
    '''Compute and stores  value'''
    def __init__(self, name, fmt=':s'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.str = 'null'

    def update(self, str):
        self.str = str

    def __str__(self):
        fmstr = '{name} {str' + self.fmt + '}'
        return fmstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class visdomMeter(object):
    def __init__(self, name,step):
        self.name = name
        self.step=step
    def update(self, val):
        self.val = val


