import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from utils import save_tensor_image
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from model import *
from multi_read_data import lowlight_loader

parser = argparse.ArgumentParser("EFDFusion")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=5000, help='epochs')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--save', type=str, default=r'.\finetune', help='location of the data corpus')
parser.add_argument('--model', type=str, default=r'.\model\weights_2000.pt', help='location of the data corpus')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)
    model = Finetunemodel(args.model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)
    train_low_data_names = r'.\test_pic\vis'
    train_ir_data_names = r'.\test_pic\ir'
    TrainDataset = lowlight_loader(img_dir=train_low_data_names, ir_img_dir=train_ir_data_names, task='train')
    test_low_data_names = r'.\test_pic\vis'
    test_ir_data_names = r'.\test_pic\ir'
    TestDataset = lowlight_loader(img_dir=test_low_data_names, ir_img_dir=test_ir_data_names,task='test')
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=False)
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False)
    total_step = 0

    model.train()
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (img_lowlight, img_ir,  img_name) in enumerate(train_queue):
            total_step += 1
            img_lowlight = Variable(img_lowlight, requires_grad=False).cuda()
            img_ir = Variable(img_ir, requires_grad=False).cuda()
            optimizer.zero_grad()
            optimizer.param_groups[0]['capturable'] = True
            loss = model._loss(img_lowlight,img_lowlight, img_ir)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            print('finetune-step:{} loss:{}'.format(epoch, loss.item()))
            utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        if epoch % 10 == 0 and total_step != 0:
            model.eval()
            with torch.no_grad():
                for iteration, (img_lowlight, img_ir,  img_name) in enumerate(test_queue):
                    img_lowlight = Variable(img_lowlight, requires_grad=False).cuda()
                    img_ir = Variable(img_ir, requires_grad=False).cuda()
                    image_name = img_name[0].split('/')[-1].split('.')[0]
                    i,  r, n,d,fuse = model(img_lowlight,img_lowlight, img_ir)
                    input_name = '%s' % (image_name)
                    r_name = '%s' % ('enhance')
                    f_name = '%s' % ('fuse')
                    save_tensor_image(r, args.save, input_name, r_name, str(epoch))
                    save_tensor_image(fuse, args.save, input_name, f_name, str(epoch))

if __name__ == '__main__':
    main()
