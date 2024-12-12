from __future__ import print_function
import argparse
import csv
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from torch.utils.data import Dataset, sampler, RandomSampler
import torch.nn.functional as F

'''
1. Parameters
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../data/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--generateroot', default='../output/generatedata/', help='path to the generated dataset')
parser.add_argument('--outf', default='../output/performance/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed',default=10000, type=int, help='manual seed')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
parser.add_argument('--dataname', default='abalone_train.csv', help='file name of dataset')

# Parameters added for WGAN
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iterations per each G iteration')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')

# Parameters added for APA (Adaptive Pseudo Augmentation)
parser.add_argument('--apa_target',default=0.6)
parser.add_argument('--apa_interval', default=4)
parser.add_argument('--apa_kimg',default=0.2)
parser.add_argument('--aug_p',default=0)
opt = parser.parse_args()

'''
Function: Construct the dataset
init function:
    Parameter description: dataset: dataset
    Return value: self.data: data; self.labels: labels
len function:
    Return value: length of the dataset
getitem function:
    Parameter description: idx: index
    Return value: data and label at index idx
'''
class MyData(Dataset):
    def __init__(self,dataset):
        self.data,self.labels=dataset

    def __getitem__(self, idx):
        img=self.data[idx]
        label=[]
        return img,label

    def __len__(self):
        return len(self.data)

'''
Function: Construct CSV dataset
init function:
    Parameter description: path: path to the CSV dataset; class_label: class label
    Return value: X: data; y: labels
len function:
    Return value: length of the dataset
getitem function:
    Parameter description: idx: index
    Return value: data and label at index idx
'''
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path,class_label):
        # load the CSV file as a dataframe
        df = pd.read_csv(path, header=None)
        # store the inputs and outputs
        self.data = (df.values[:, :-1]).tolist()
        self.label = (df.values[:, -1]).tolist()

        self.X= []
        self.y=[]

        for i in range(0,len(self.label)):
            if self.label[i]==class_label:
                self.X.append(self.data[i])
                self.y.append(self.label[i])

        self.X=np.array(self.X)
        self.y=np.array(self.y)

        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_dim(self):
        return len(self.X[0])

'''
Function: Get data from the original dataset
Parameter description: fi: dataset name; str1: path
Return value: x_data: data; y: labels
'''
def function(fi,str1):
    dtfold_path = str1 + fi

    df = pd.read_csv(dtfold_path, header=None)

    label_list = (df.values[:, -1]).tolist()
    label_list=sorted(list(set(label_list)), reverse=False)
    dataset_list=[]
    for i in range(0,len(label_list)):
        dataset_tmp=CSVDataset(dtfold_path,label_list[i])
        dataset_list.append(dataset_tmp)

    return dataset_list,label_list

'''
Function: Return pseudo-augmented real data
Parameter description: real_img: real data; pseudo_data: generated samples; aug_p: degree of pseudo-augmentation
Return value: pseudo-augmented real data
'''
def adaptive_pseudo_augmentation(real_img,pseudo_data,aug_p):
    # Apply Adaptive Pseudo Augmentation (APA)
    batch_size = real_img.shape[0]
    pseudo_flag = torch.ones([batch_size, real_img.shape[1]])
    # If a random number is smaller than the degree of pseudo-augmentation, set flag to 1; otherwise, 0. This creates a pseudo-augmentation matrix of the same size as real_img
    pseudo_flag = torch.where(torch.rand([batch_size, 1]) < aug_p,
                              pseudo_flag, torch.zeros_like(pseudo_flag))

    # If the pseudo-augmentation matrix is the same as a zero matrix, return real_img (no augmentation)
    # Otherwise, check if the pseudo-augmented data is non-empty and return a mixture of pseudo-augmented data and original data
    if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
        return real_img
    else:
        return pseudo_data * pseudo_flag + real_img * (1 - pseudo_flag)


print(opt)
real_cpu=None

#1. Some parameter settings
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Set random seed for torch
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# List of sample sizes
numimages_lis=[300,500,1000,2000,5000]
for num in numimages_lis:
    # 1. Load the dataset
    aug_p=0
    aug_p_lis = []
    dataname = opt.dataname
    fi_test = dataname.replace("train", "test")
    print(dataname)
    data_lis,lable_lis=function(dataname,opt.dataroot)

    sample_dataset = []
    for data_i in range(0,len(data_lis)):
        #2.1 Determine the number of samples to generate and model output path
        if data_i == 0:
            samp_num = int(0.5 * num)
        else:
            samp_num = num - int(0.5 * num)

        dataset=data_lis[data_i]
        outf = opt.outf + '/model/'
        try:
            os.makedirs(outf)
        except OSError:
            pass

        assert dataset
        # 2.2 Set up the DataLoader
        batchSize = int(len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,shuffle=True, num_workers=0)
        use_mps = opt.mps and torch.backends.mps.is_available()
        if opt.cuda:
            device = torch.device("cuda:0")
        elif use_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # 2.3 Define network architecture
        ngpu = int(opt.ngpu)
        data_dim=dataset.get_dim()
        nz=int(8*data_dim)
        ngf=16
        ndf=16
        nx=int(data_dim)


        # custom weights initialization called on netG and netD
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)


        class Generator(nn.Module):
            def __init__(self, ngpu):
                super(Generator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.Linear(nz, ngf),
                    nn.ReLU(True),
                    nn.Linear(ngf, nx),

                )

            def forward(self, input):
                input = input.view(input.size(0), input.size(1))
                if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                return output

        class Discriminator(nn.Module):
            def __init__(self, ngpu):
                super(Discriminator, self).__init__()
                self.ngpu = ngpu
                self.main = nn.Sequential(
                    nn.Linear(nx, ndf),
                    nn.ReLU(True),
                    nn.Linear(ndf, 1),
                    # nn.Sigmoid()
                )

            def forward(self, input):
                input = input.view(input.size(0),input.size(1) )
                if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
                    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
                else:
                    output = self.main(input)
                # output = output.mean(0)
                # return output.view(1)
                return output


         # 2.4 Define and train the model
        # 1. Initialize Generator and Discriminator
        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)

        #define discriminator
        netD = Discriminator(ngpu).to(device)
        netD.apply(weights_init)

        optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
        optimizerD0 = optim.RMSprop(netD.parameters(), lr=opt.lr)


        input = torch.FloatTensor(batchSize, data_dim).to(device)


        one = torch.FloatTensor([1]).to(device)
        mone = one * -1
        mone=mone.to(device)
        if opt.dry_run:
            opt.niter = 1

        #5.training
        gen_iterations = 0
        for epoch in range(opt.niter):
            data_iter = iter(dataloader)
            i = 0
            while i < len(dataloader):
                ############################
                # (1) Update D network
                ###########################
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                    # train the discriminator Diters times
                    if gen_iterations < 25 or gen_iterations % 500 == 0:
                        Diters = 100
                    else:
                        Diters = opt.Diters
                    j = 0
                    while j < Diters and i < len(dataloader):
                        j += 1
                        for p in netD.parameters():
                            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                        # Using try-except to handle StopIteration gracefully
                        try:
                            data = next(data_iter)
                        except StopIteration:
                            break  # Exit the loop if data is exhausted

                        i += 1

                        # train with real
                        D_x_sum = []
                        D_G_z1_sum = []
                        noise = torch.FloatTensor(batchSize, nz, 1, 1).to(device)
                        fixed_noise = torch.randn(batchSize, nz, 1, 1).normal_(0, 1).to(device)

                        real_cpu_tmp, _ = data
                        batch_size = real_cpu_tmp.size(0)
                        noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
                        with torch.no_grad():
                            noisev = Variable(noise)  # totally freeze netG
                        fake = Variable(netG(noisev).data)


                        # APA
                        if epoch % opt.apa_interval == 0:
                            # 1.Calculate the degree of overfitting for the discriminator
                            out_real_tmp = netD(real_cpu_tmp)
                            out_false_tmp = netD(fake.detach())

                            out_real_tmp=F.sigmoid(out_real_tmp)
                            out_false_tmp=F.sigmoid(out_false_tmp)

                            out_real=torch.log(out_real_tmp/(1-out_real_tmp))
                            out_false=torch.log(out_false_tmp/(1-out_false_tmp))

                            over_fit_degree1 = (np.sign(out_real.detach())).mean(0)
                            over_fit_degree2 = (-np.sign(out_false.detach())).mean(0)
                            over_fit_degree3 = (over_fit_degree1 + over_fit_degree2) / 2

                            # 2.Calculate the adjustment factor for augmentation probability (p) based on the degree of overfitting
                            adjust = np.sign(over_fit_degree1 - opt.apa_target) * (
                                    batch_size * opt.apa_interval) / (opt.apa_kimg * 1000)

                            # Update the augmentation probability (p) ensuring it remains non-negative
                            aug_p = torch.max((opt.aug_p + adjust),torch.tensor(0))

                            # 3Perform adaptive pseudo augmentation on the real data based on the updated p value
                            real_cpu = adaptive_pseudo_augmentation(real_cpu_tmp, fake, aug_p)
                            aug_p_lis.append(aug_p)
                        else:
                            real_cpu=real_cpu_tmp


                        netD.zero_grad()


                        if opt.cuda:
                            real_cpu = real_cpu.to(device)
                        input.resize_as_(real_cpu).copy_(real_cpu)
                        inputv = Variable(input)

                        errD_real = netD(inputv)
                        errD_real=errD_real.mean(0).view(1)
                        errD_real.backward(one)

                        # train with fake
                        inputv = fake
                        errD_fake = netD(inputv)
                        errD_fake=errD_fake.mean(0).view(1)
                        errD_fake.backward(mone)
                        errD = errD_real - errD_fake

                        output = netD(real_cpu)
                        output=output.mean(0).view(1)
                        D_x = output.mean().item()
                        D_x_sum.append(D_x)

                        output = netD(fake.detach())
                        output=output.mean(0).view(1)
                        D_G_z1 = output.mean().item()
                        D_G_z1_sum.append(D_G_z1)

                    ############################
                    # (2) Update G network
                    ###########################
                    for p in netD.parameters():
                        p.requires_grad = False  # to avoid computation
                    netG.zero_grad()
                    # in case our last batch was the tail batch of the dataloader,
                    # make sure we feed a full batch of noise
                    noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
                    noisev = Variable(noise)
                    fake = netG(noisev)
                    output = netD(fake)
                    output=output.mean(0).view(1)

                    errG = output
                    errG.backward(one)
                    optimizerG.step()
                    gen_iterations += 1
                    D_G_z2 = output.mean().item()
                    D_x_avg = torch.mean(torch.tensor(D_x_sum), dim=-1)
                    D_G_z1_avg=torch.mean(torch.tensor(D_G_z1_sum), dim=-1)

                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, opt.niter, i, len(dataloader),
                            errD.item(), errG.item(), D_x_avg, D_G_z1_avg, D_G_z2))
                    log_txt=str('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, opt.niter, i, len(dataloader),
                            errD.item(), errG.item(), D_x_avg, D_G_z1_avg, D_G_z2))
                    f = open(outf+'log.txt', 'a', encoding='UTF-8')
                    f.write(str(log_txt))
                    f.write(str('\n'))

                    if opt.dry_run:
                        break

                if epoch==opt.niter-1:
                    # do checkpointing
                    # if num==300:
                    #     torch.save(netG.state_dict(), '%s/netG_epoch_%d_class_%d.pth' % (outf, epoch,data_i+1))
                    #     torch.save(netD0.state_dict(), '%s/netD_epoch_%d_num_0_class_%d.pth' % (outf, epoch,data_i+1))

                    save_path = opt.generateroot  # 采样结果保存的位置
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass

                    # 生成数据
                    noise = torch.randn(samp_num, nz, 1, 1, device=device)
                    fake = netG(noise)
                    fake_tmp = fake.cpu()

                    for accpte_sam in fake_tmp:
                        tmp = accpte_sam.tolist()
                        tmp.append(lable_lis[data_i])
                        sample_dataset.append(tmp)

    csv_path1 = save_path + dataname[:-4] + str(num) + '.csv'
    with open(csv_path1, 'w', newline='', encoding='utf-8') as s1:
        writer1 = csv.writer(s1)
        writer1.writerows(sample_dataset)




