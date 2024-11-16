import os
import sys
from utils import save_tensor_image
import argparse
import torch.utils
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
parser.add_argument('--save', type=str, default=r'.\result', help='location of the data corpus')
parser.add_argument('--model', type=str, default=r'.\model\weights_2000.pt', help='location of the data corpus')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.makedirs(args.save, exist_ok=True)

test_low_data_names = r'.\test_pic\vis'
test_ir_data_names = r'.\test_pic\ir'
TestDataset = lowlight_loader(img_dir=test_low_data_names, ir_img_dir=test_ir_data_names, task='test')
test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0, shuffle=False)
def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    model = Finetunemodel(args.model)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for iteration, (img_lowlight, img_ir, img_name) in enumerate(test_queue):
            img_lowlight = Variable(img_lowlight, requires_grad=False).cuda()
            img_ir = Variable(img_ir, requires_grad=False).cuda()
            image_name = img_name[0].split('/')[-1].split('.')[0]
            epoch = 'result'
            i, r, n,d,fuse = model(img_lowlight,img_lowlight, img_ir)
            input_name = '%s' % (image_name)
            f_name = '%s' % ('Fusion')
            r_name = '%s' % ('Enhance')
            save_tensor_image(fuse, args.save, input_name, f_name, str(epoch))
            save_tensor_image(r, args.save, input_name, r_name, str(epoch))

if __name__ == '__main__':
    main()
