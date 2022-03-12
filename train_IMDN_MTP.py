import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import architecture
from data import DIV2K, Set5_val
import utils
import skimage.color as sc
import random
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description="IMDN")
parser.add_argument("--batch_size", type=int, default=16,
                    help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=1,
                    help="testing batch size")
parser.add_argument("-nEpochs", type=int, default=1000,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=200,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--resume", default="", type=str,
                    help="path to checkpoint")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--threads", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default="training_data/",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=800,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--test_every", type=int, default=1000)
parser.add_argument("--scale", type=int, default=2,
                    help="super-resolution scale")
parser.add_argument("--patch_size", type=int, default=192,
                    help="output patch size")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--isY", action="store_true", default=False)
parser.add_argument("--ext", type=str, default='.npy')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--model_path", type=str, default='/content/drive/MyDrive/SR')
parser.add_argument("--patch_change", type=str, default='')
parser.add_argument("--valid_root", type=str, default='/content')
parser.add_argument("--iskaggle", type=str, default='no')
parser.add_argument("--valid_freq", type=int, default=20)

args = parser.parse_args()
print(args)
torch.backends.cudnn.benchmark = True
# random seed
seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)
print("Ramdom Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda else 'cpu')

print("===> Loading datasets")

trainset = DIV2K.div2k(args)
training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

testset = Set5_val.DatasetFromFolderVal(args.valid_root+"/DIV2K_valid_HR",
                                       args.valid_root+"/DIV2K_valid_LR_bicubic/X4/",
                                       args.scale)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.testBatchSize,
                                 shuffle=False)                                

print("===> Building models")
args.is_train = True

model = architecture.IMDN(upscale=args.scale)
l1_criterion = nn.L1Loss()

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.pretrained:

    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['net'], strict=True)
        args.start_epoch = checkpoint['epoch']

    else:
        print("===> no models found at '{}'".format(args.pretrained))

print("===> Setting Optimizer")

optimizer = optim.Adam(model.parameters(), lr=args.lr)

model_folder = args.model_path

def train(epoch):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))


def valid():
    model.eval()

    avg_psnr, avg_ssim = 0, 0
    for batch in testing_data_loader:
        lr_tensor, hr_tensor = batch[0], batch[1]
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            pre = model(lr_tensor)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])
        crop_size = args.scale
        cropped_sr_img = utils.shave(sr_img, crop_size)
        cropped_gt_img = utils.shave(gt_img, crop_size)
        if args.isY is True:
            im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
            im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
        else:
            im_label = cropped_gt_img
            im_pre = cropped_sr_img
        avg_psnr += utils.compute_psnr(im_pre, im_label)
        avg_ssim += utils.compute_ssim(im_pre, im_label)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader)))


def save_checkpoint(epoch):
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    checkpoint_path = os.path.join(model_folder,'model.pth')
    state_dict = model.state_dict()
    state = {
            "net": state_dict,
            "epoch": epoch,
        }
    torch.save(state, checkpoint_path)
    if args.iskaggle == 'y':
      os.system('cp -r '+'/kaggle/working/model.pth '+'/kaggle/working/SRTrain')
      os.system('cd /kaggle/working/SRTrain')
      os.system('git add model.pth')
      os.system("git commit -m 'ts'")
      os.system("git push -u origin main")
      os.system("cd /kaggle/working/IMDN")
    print("===> Checkpoint saved to {}".format(checkpoint_path))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print("===> Training")
print_network(model)
for epoch in range(args.start_epoch, args.nEpochs + 1):
    train(epoch)
    if epoch%20==0:
      save_checkpoint(epoch)
    if epoch%(args.valid_freq)==0:
      valid() 
    if epoch%200==0:
        valid() 
        if args.patch_change == 'up':
          args.patch_size = args.patch_size + 64
        elif args.path_change == 'down':
          args.patch_size = args.patch_size - 64
        print('patch_size',args.patch_size)
        trainset = DIV2K.div2k(args)
        training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
      