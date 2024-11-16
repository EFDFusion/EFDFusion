import os
import numpy as np
import torch
import shutil
from torch.nn.modules.container import T
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
def gauss_cdf(x):
    return 0.5*(1+torch.erf(x/torch.sqrt(torch.tensor(2.))))

def gauss_kernel(kernlen=21,nsig=3,channels=1):
    interval=(2*nsig+1.)/(kernlen)
    x=torch.linspace(-nsig-interval/2.,nsig+interval/2.,kernlen+1,).cuda()
    kern1d=torch.diff(gauss_cdf(x))
    kernel_raw=torch.sqrt(torch.outer(kern1d,kern1d))
    kernel=kernel_raw/torch.sum(kernel_raw)
    out_filter=kernel.view(1,1,kernlen,kernlen)
    out_filter = out_filter.repeat(channels,1,1,1)
    return  out_filter

def blur(x):

    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(device)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    return torch.nn.functional .conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))




def rgb_to_ycrcb(image):

    transform_matrix = torch.tensor([[0.299, 0.587, 0.114],
                                     [-0.169, -0.331, 0.5],
                                     [0.5, -0.419, -0.081]])

    image_float = image.float() / 255.0
    image_float = image_float.permute(0, 2, 3, 1).contiguous()
    image_float = image_float.view(-1, 3)

    ycrcb_image = torch.matmul(image_float, transform_matrix.T)
    ycrcb_image[:, 0] += 16.0
    ycrcb_image[:, 1:] += 128.0

    ycrcb_image = ycrcb_image.view(image.size(0), image.size(2), image.size(3), 3)
    ycrcb_image = ycrcb_image.permute(0, 3, 1, 2)

    return ycrcb_image

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6



def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'),exist_ok=True)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def tensor_to_image(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im

def save_tensor_image(tensor, save_path, input_name, tensor_name, epoch):

    image = tensor_to_image(tensor)
    dir_path = os.path.join(save_path, 'result', tensor_name)
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{input_name}_{tensor_name}_{epoch}.png"
    image_path = os.path.join(dir_path, file_name)
    Image.fromarray(image).save(image_path, 'PNG')