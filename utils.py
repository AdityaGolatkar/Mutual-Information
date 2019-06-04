from io import BytesIO
import os
import errno
import torch
import torch.nn as nn

def mkdir(directory):
    '''Make directory and all parents, if needed.
    Does not raise and error if directory already exists.
    '''

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)


def get_error(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(100. - correct_k.mul_(100.0 / batch_size))
    return res

def get_binary_error(output, target):
    batch_size = target.size(0)

    correct = 0.
    logits = output
    pred = logits > 0.
    correct = pred.eq(target).float().mean()
    error = (1. - correct) * 100.
    return error.item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def imshow(image, colormap=False, video=False):
    import imageio
    import iterm2_tools
    from matplotlib import cm
    from scipy.misc import bytescale
    from PIL import Image
    from iterm2_tools.images import display_image_bytes

    if type(image).__name__ == 'Variable':
        image = image.data
    if 'torch.cuda' in type(image).__module__:
        image = image.cpu()
    if 'Tensor' in type(image).__name__:
        image = image.numpy()

    if colormap:
        image = (cm.Blues(image) * 255).astype(np.uint8)
    else:
        image = bytescale(image)

    if image.ndim == 4:
        video = True
    if image.ndim==3 and (image.shape[0] not in [1,3] and image.shape[-1] not in [1,3]):
        video = True

    if video:
        if image.shape[1] == 3:
            image = image.transpose([2,3,1]).astype(np.uint8)
        image = image.squeeze()
        if image.ndim == 2:
            image = image[None]
        images = [im for im in image]
        s = imageio.mimsave(imageio.RETURN_BYTES, images, format='gif', duration=0.3)
        print(display_image_bytes(s))
    else:
        if image.shape[0] == 3:
            image = image.transpose([1,2,0]).astype(np.uint8)
        image = image.squeeze()
        s = imageio.imsave(imageio.RETURN_BYTES, image, format='png')
        s = display_image_bytes(s)
        # Depending on the version of iterm2_tools, display_image_bytes can
        # either print directly to stdout or return the string to print.
        if s is not None:
            print(s)

def output_plot(use_iterm2=True, format='png'):
    import matplotlib.pyplot as plt

    output = BytesIO()
    plt.savefig(output, format=format, dpi=None, bbox_inches='tight' )
    if use_iterm2:
        from iterm2_tools.images import display_image_bytes
        s = display_image_bytes(output.getvalue(), filename="plot.{}".format(format))
        # Depending on the version of iterm2_tools, display_image_bytes can
        # either print directly to stdout or return the string to print.
        if s is not None:
            print(s)
    else:
        sys.stdout.write(output.getvalue())
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x
