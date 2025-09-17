"""
PECNet Training Script
PECNet训练脚本
"""

import math
import os
import argparse

import torch as th
import torchbox as tb
from torch.optim.lr_scheduler import LambdaLR

from PECNet.dataset import readsamples
from PECNet import PECNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--datacfg', type=str, default='./data.yaml')
parser.add_argument('--modelcfg', type=str, default='./PECNet.yaml')
parser.add_argument('--solvercfg', type=str, default='./solver.yaml')

# 训练相关超参数
parser.add_argument('--loss_type', type=str, default='Entropy', help='Entropy, Contrast, LogFro')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--size_batch', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=1500)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--scheduler', type=str, default='StepLR')
parser.add_argument('--snapshot_name', type=str, default='2020')

# 其他
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--mkpetype', type=str, default='RealPE', help='make phase error(SimPoly, SimSin...)')

cfg = parser.parse_args()
seed = cfg.seed

# 是否进行频移
ftshift = False

# CUDA/CuDNN 配置
cudaTF32, cudnnTF32 = False, False
benchmark, deterministic = True, True

# ========== 读取 yaml 配置 ==========
datacfg = tb.loadyaml(cfg.datacfg)
modelcfg = tb.loadyaml(cfg.modelcfg)
solvercfg = tb.loadyaml(cfg.solvercfg)
# 从 solvercfg 读取默认 epoch/batch, 如果命令行里没给就用solvercfg
solvercfg['nepoch'] = cfg.num_epochs if cfg.num_epochs is not None else solvercfg['sbatch']
solvercfg['sbatch'] = cfg.size_batch if cfg.size_batch is not None else solvercfg['sbatch']
num_epochs = solvercfg['nepoch']
size_batch = solvercfg['sbatch']

# 如果 data.yaml 里定义了 SAR_AF_DATA_PATH, 优先使用
if 'SAR_AF_DATA_PATH' in os.environ.keys():
    datafolder = os.environ['SAR_AF_DATA_PATH']
else:
    datafolder = datacfg['SAR_AF_DATA_PATH']

print(cfg)
print(modelcfg)

# ========== 读取训练集和测试集文件路径 (从 data.yaml 中读取) ==========
fileTrain = [datafolder + datacfg['filenames'][i] for i in datacfg['trainid']]

print("--->Train files:", fileTrain)

# ========== 读取数据 (以 readsamples 为例) ==========
keys = [['UCD_cleaned', 'PE_cleaned']]

# 读取训练数据
Xall, pall = readsamples(
    fileTrain,
    keys=keys,
    nsamples=[420],
    mode='sequentially',
    seed=cfg.seed
)

N = Xall.shape[0]
th.manual_seed(1)
idx = th.randperm(N)
train_num = int(0.8 * N) 
idx_train = idx[:train_num]
idx_test = idx[train_num:]

Xtrain = Xall[idx_train]
ptrain = pall[idx_train]
Xtest = Xall[idx_test]
ptest = pall[idx_test]


# ========== 输出文件地址 ==========
outfolder = (
        './snapshot/' + modelcfg['model'] + '/' + cfg.mkpetype + '/' + cfg.loss_type
        + '/' + cfg.optimizer + '/' + cfg.scheduler + '/' + cfg.snapshot_name
)
losslog = tb.LossLog(outfolder)

os.makedirs(outfolder + '/images', exist_ok=True)
os.makedirs(outfolder + '/weights', exist_ok=True)

device = cfg.device
devicename = 'CPU' if device == 'cpu' else th.cuda.get_device_name(tb.str2num(device, int))

checkpoint_path = outfolder + '/weights/'

# ========== 定义损失函数 ==========
loss_ent_func = tb.EntropyLoss('natural', cdim=-1, dim=(1, 2), reduction='mean')
loss_cts_func = tb.ContrastLoss('way1', cdim=-1, dim=(1, 2), reduction='mean')
loss_fro_func = tb.Pnorm(p=1, cdim=-1, dim=(1, 2), reduction='mean')

# 设定随机状态
th.backends.cuda.matmul.allow_tf32 = cudaTF32
th.backends.cudnn.allow_tf32 = cudnnTF32
th.backends.cudnn.benchmark = benchmark
th.backends.cudnn.deterministic = deterministic
tb.setseed(seed)

# ========== 构造网络并初始化 ==========
net = PECNet(Na=Na, Nr=Nr, convp=modelcfg['lpaf'], ftshift=False, seed=cfg.seed)
net.to(device=device)
print(net)

# ========== 构造优化器 ==========
opt = solvercfg['Optimizer'][cfg.optimizer]
opt['lr'] = cfg.lr if cfg.lr is not None else opt['lr']

if cfg.optimizer.lower() in ['adamw']:
    optimizer = th.optim.AdamW(
        [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'initial_lr': opt['lr']}],
        lr=opt['lr'],
        betas=opt['betas'],
        eps=opt['eps'],
        weight_decay=opt['weight_decay'],
        amsgrad=opt['amsgrad']
    )
elif cfg.optimizer.lower() == 'adam':
    optimizer = th.optim.Adam(
        net.parameters(),
        lr=opt['lr'],
        betas=opt['betas'],
        eps=opt['eps'],
        weight_decay=opt['weight_decay']
    )
else:
    raise NotImplementedError(f"Unsupported optimizer: {cfg.optimizer}")

tb.device_transfer(optimizer, 'optimizer', device=device)

# ========== 定义调度器 ==========
numSamples = Xtrain.shape[0]
numBatch = numSamples // size_batch
total_steps = num_epochs * numBatch

warmup_ratio = 0.1
warmup_steps = int(warmup_ratio * total_steps)


def lr_lambda(current_step):
    if current_step < warmup_steps:
        # 线性warmup: 0 -> 1
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # 余弦退火
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))


if cfg.scheduler.lower() == 'lambdalr':
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
elif cfg.scheduler.lower() == 'steplr':
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
else:
    scheduler = None

print("=> total_steps=", total_steps, "warmup_steps=", warmup_steps)
print("---", cfg.optimizer, opt)
print("---", cfg.scheduler)
print("---", device)
print("---", devicename)
print("---Torch Version: ", th.__version__)
print("---Torch CUDA Version: ", th.version.cuda)
print("---CUDNN Version: ", th.backends.cudnn.version())
print("---CUDA TF32: ", cudaTF32)
print("---CUDNN TF32: ", cudnnTF32)
print("---CUDNN Benchmark: ", benchmark)
print("---CUDNN Deterministic: ", deterministic)

# ========== 把数据转到 float32 ==========
Xtrain = Xtrain.to(th.float32)
Xtest = Xtest.to(th.float32)
ptrain = ptrain.to(th.float32)
ptest = ptest.to(th.float32)

# 记录最小损失
lossmintrain = float('inf')
lossmintest = float('inf')

# ========== 正式开始训练 ==========
for epoch in range(num_epochs):
    lossvtrain = net.train_epoch(
        Xtrain,
        sizeBatch=size_batch,
        loss_ent_func=loss_ent_func,
        loss_cts_func=loss_cts_func,
        loss_fro_func=loss_fro_func,
        loss_type=cfg.loss_type,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=None,
        device=device
    )

    print(optimizer.param_groups[0]['lr'])

    idxtrain = list(range(0, Xtrain.shape[0], max(1, Xtrain.shape[0] // 16)))

    lossvtest = net.valid_epoch(
        Xtest,
        sizeBatch=size_batch,
        loss_ent_func=loss_ent_func,
        loss_cts_func=loss_cts_func,
        loss_fro_func=loss_fro_func,
        loss_type=cfg.loss_type,
        epoch=epoch,
        device=device
    )
    idxtest = list(range(0, Xtest.shape[0], max(1, Xtest.shape[0] // 32)))

    losslog.add('train', lossvtrain)
    losslog.add('test', lossvtest)
    losslog.plot()

    if lossvtrain < lossmintrain:
        th.save({
            'epoch': epoch,
            'network': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path + 'best_train.pth.tar')
        lossmintrain = lossvtrain

    if lossvtest < lossmintest:
        th.save({
            'epoch': epoch,
            'network': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path + 'best_test.pth.tar')
        lossmintest = lossvtest

# ========== 训练结束, 保存最终模型 ==========
th.save({
    'epoch': num_epochs,
    'network': net.state_dict(),
    'optimizer': optimizer.state_dict(),
}, checkpoint_path + 'final.pth.tar')

print("Training losses:", losslog.get('train'))
print("Test losses:", losslog.get('test'))

avg_train_loss = sum(losslog.get('train')) / len(losslog.get('train'))
avg_test_loss = sum(losslog.get('test')) / len(losslog.get('test'))

print(f"Average training loss over entire run: {avg_train_loss}")
print(f"Average test loss over entire run: {avg_test_loss}")
print("Done.")
