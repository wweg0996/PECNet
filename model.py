"""
PECNet (Phase Error Correction Network) Model
相位误差校正网络模型

基于卷积神经网络和Transformer的混合架构，用于双地基合成孔径雷达图像相位误差校正
"""

import time
from PECNet.dataset import saveimage
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from collections import OrderedDict
import torchbox as tb


class PECNet(nn.Module):
    """
    Args:
        Na (int): 方位向样本数
        Nr (int): 距离向样本数  
        convp (list): 卷积层参数配置
        ftshift (bool): 是否进行频移
        seed (int): 随机种子
    """

    def __init__(self, Na, Nr, convp, ftshift=True, seed=None):
        super().__init__()
        self.Na = Na
        self.Nr = Nr
        self.ftshift = ftshift
        self.seed = seed

        # ========== 卷积特征提取部分 ========== #
        FD = OrderedDict()
        print(convp)
        # 第一层卷积
        FD['conv1'] = nn.Conv2d(
            in_channels=2,
            out_channels=convp[0][0],
            kernel_size=convp[0][1:3],
            stride=convp[0][3:5],
            padding=convp[0][5:7],
            dilation=convp[0][7:9],
            groups=convp[0][9]
        )
        FD['in1'] = nn.InstanceNorm2d(convp[0][0])
        FD['relu1'] = nn.LeakyReLU()
        FD['dropout1'] = nn.Dropout2d(p=0.2)

        # 后续卷积层
        for n in range(1, len(convp)):
            FD['conv' + str(n + 1)] = nn.Conv2d(
                in_channels=convp[n - 1][0],
                out_channels=convp[n][0],
                kernel_size=convp[n][1:3],
                stride=convp[n][3:5],
                padding=convp[n][5:7],
                dilation=convp[n][7:9],
                groups=convp[n][9]
            )
            FD['in' + str(n + 1)] = nn.InstanceNorm2d(convp[n][0])
            FD['relu' + str(n + 1)] = nn.LeakyReLU()
            FD['dropout' + str(n + 1)] = nn.Dropout2d(p=0.2)

        FD['gapool'] = nn.AdaptiveAvgPool2d((None, 1))

        self.features = nn.Sequential(FD)

        # ========== Transformer 设置 ========== #
        last_channels = convp[-1][0]

        # TransformerEncoder (2层, 4头, FFN=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=last_channels,  
            nhead=4,
            dim_feedforward=512,
            batch_first=True  
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # ========== 相位预测层 ========== #
        self.phase_predictor = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, Na)  # 输出 [N, Na]
        )

        if self.seed is not None:
            th.manual_seed(seed)

    def forward(self, X):
        """
        前向传播
        
        Args:
            X (torch.Tensor): 输入张量 [N, H, W, 2]，最后一维为(实部, 虚部)
            
        Returns:
            tuple: (校正后的信号, 预测的相位误差)
                - X (torch.Tensor): 校正后的复数信号
                - pa (torch.Tensor): 预测的相位误差 [N, Na]
        """
        d = X.dim()  
        Y = th.stack((X[..., 0], X[..., 1]), dim=1)

        feats = self.features(Y)
        feats = feats.squeeze(dim=-1)  
        feats = feats.permute(0, 2, 1)

        out = self.transformer(feats)

        out = out.permute(0, 2, 1)  
        out = self.pool(out)  
        out = out.squeeze(-1)  

        pa = self.phase_predictor(out)

        sizea = [1] * d
        sizea[0], sizea[-3], sizea[-2], sizea[-1] = pa.size(0), pa.size(1), 1, 2
        epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        X = tb.ifft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=self.ftshift)
        X = tb.ematmul(X, epa, cdim=-1)
        X = tb.fft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=self.ftshift)

        return X, pa

    def weights_init(self, m):
        """
        权重初始化
        
        Args:
            m: 网络模块
        """
        if isinstance(m, th.nn.Conv2d):
            th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
            m.bias.data.zero_()
        if isinstance(m, th.nn.Linear):
            th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
            m.bias.data.zero_()

    def train_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, optimizer,
                    scheduler, device):
        """
        训练一个epoch
        
        Args:
            X (torch.Tensor): 训练数据
            sizeBatch (int): 批次大小
            loss_ent_func: 熵损失函数
            loss_cts_func: 对比度损失函数
            loss_fro_func: Frobenius范数损失函数
            loss_type (str): 损失类型
            epoch (int): 当前epoch
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            
        Returns:
            float: 平均训练损失
        """
        self.train()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = tb.randperm(0, numSamples, numSamples)
        lossENTv, lossCTSv, lossFROv, lossvtrain = 0., 0., 0., 0.
        for b in range(numBatch):
            i = idx[b * sizeBatch:(b + 1) * sizeBatch]
            xi = X[i]
            xi = xi.to(device)

            optimizer.zero_grad()
            fi, casi = self.forward(xi)

            lossENT = loss_ent_func(fi)
            lossCTS = loss_cts_func(fi)
            lossFRO = loss_fro_func(fi)

            if loss_type == 'Entropy':
                loss = lossENT
            if loss_type == 'Entropy+LogFro':
                loss = lossENT + lossFRO
            if loss_type == 'Contrast':
                loss = lossCTS
            if loss_type == 'Entropy/Contrast':
                loss = lossENT / lossCTS

            loss.backward()

            lossvtrain += loss.item()
            lossCTSv += lossCTS.item()
            lossENTv += lossENT.item()
            lossFROv += lossFRO.item()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        tend = time.time()

        lossvtrain /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch
        print("--->Train epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvtrain, lossENTv, lossFROv, lossCTSv, tend - tstart))
        return lossvtrain

    def valid_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, device):
        """
        验证一个epoch
        
        Args:
            X (torch.Tensor): 验证数据
            sizeBatch (int): 批次大小
            loss_ent_func: 熵损失函数
            loss_cts_func: 对比度损失函数
            loss_fro_func: Frobenius范数损失函数
            loss_type (str): 损失类型
            epoch (int): 当前epoch
            device: 设备
            
        Returns:
            float: 平均验证损失
        """
        self.eval()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossENTv, lossCTSv, lossFROv, lossvvalid = 0., 0., 0., 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi = X[i]
                xi = xi.to(device)

                fi, casi = self.forward(xi)

                lossENT = loss_ent_func(fi)
                lossCTS = loss_cts_func(fi)
                lossFRO = loss_fro_func(fi)

                if loss_type == 'Entropy':
                    loss = lossENT
                if loss_type == 'Entropy+LogFro':
                    loss = lossENT + lossFRO
                if loss_type == 'Contrast':
                    loss = lossCTS
                if loss_type == 'Entropy/Contrast':
                    loss = lossENT / lossCTS

                lossvvalid += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

        tend = time.time()

        lossvvalid /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch

        print("--->Valid epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvvalid, lossENTv, lossFROv, lossCTSv, tend - tstart))

        return lossvvalid

    def test_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, device):
        """
        测试一个epoch
        
        Args:
            X (torch.Tensor): 测试数据
            sizeBatch (int): 批次大小
            loss_ent_func: 熵损失函数
            loss_cts_func: 对比度损失函数
            loss_fro_func: Frobenius范数损失函数
            loss_type (str): 损失类型
            epoch (int): 当前epoch
            device: 设备
            
        Returns:
            float: 平均测试损失
        """
        self.eval()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        lossENTv, lossCTSv, lossFROv, lossvtest = 0., 0., 0., 0.
        with th.no_grad():
            for b in range(numBatch):
                i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                xi = X[i]
                xi = xi.to(device)

                fi, casi = self.forward(xi)

                lossENT = loss_ent_func(fi)
                lossCTS = loss_cts_func(fi)
                lossFRO = loss_fro_func(fi)

                if loss_type == 'Entropy':
                    loss = lossENT
                if loss_type == 'Entropy+LogFro':
                    loss = lossENT + lossFRO
                if loss_type == 'Contrast':
                    loss = lossCTS
                if loss_type == 'Entropy/Contrast':
                    loss = lossENT / lossCTS

                lossvtest += loss.item()
                lossCTSv += lossCTS.item()
                lossENTv += lossENT.item()
                lossFROv += lossFRO.item()

        tend = time.time()

        lossvtest /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch

        print("--->Test epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvtest, lossENTv, lossFROv, lossCTSv, tend - tstart))

        return lossvtest

    def visual_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, device):
        """
        可视化epoch（用于多聚焦器场景）
        
        Args:
            X (torch.Tensor): 输入数据
            sizeBatch (int): 批次大小
            loss_ent_func: 熵损失函数
            loss_cts_func: 对比度损失函数
            device: 设备
        """
        self.eval()

        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = list(range(numSamples))
        Y = th.zeros_like(X)
        with th.no_grad():
            for n in range(self.Nf):
                tstart = time.time()
                for b in range(numBatch):
                    i = idx[b * sizeBatch:(b + 1) * sizeBatch]
                    xi = X[i]
                    xi = xi.to(device)

                    pyi, _ = self.focusers[n](xi)
                    Y[i] = pyi.detach().cpu()

                tend = time.time()
                lossentv = loss_ent_func(Y).item()
                lossctsv = loss_cts_func(Y).item()
                X = Y

                print(
                    'Focuser: %d, Entropy: %.4f, Contrast: %.4f, Time: %.2fs' % (n, lossentv, lossctsv, tend - tstart))
                saveimage(X, X, [0, 1, 2], prefixname='visual%d' % n, outfolder='snapshot/')

    def plot(self, xi, pai, idx, prefixname, outfolder, device):
        """
        绘制和保存结果图像
        
        Args:
            xi (torch.Tensor): 输入信号
            pai (torch.Tensor): 真实相位误差
            idx (list): 样本索引
            prefixname (str): 文件名前缀
            outfolder (str): 输出文件夹
            device: 设备
        """
        self.eval()
        with th.no_grad():
            xi, pai, = xi.to(device), pai.to(device)

            fi, ppai = self.forward(xi)

        saveimage(xi, fi, idx, prefixname=prefixname, outfolder=outfolder + '/images/')

        pai = pai.detach().cpu().numpy()
        ppai = ppai.detach().cpu().numpy()

        for i, ii in zip(range(len(idx)), idx):
            plt.figure()
            plt.plot(pai[i], '-b')
            plt.plot(ppai[i], '-r')
            plt.legend(['Real', 'Estimated'])
            plt.grid()
            plt.xlabel('Aperture time (samples)')
            plt.ylabel('Phase (rad)')
            plt.title('Estimated phase error (polynomial degree ' + str(self.Na) + ')')
            plt.savefig(outfolder + '/images/' + prefixname + '_phase_error_azimuth' + str(ii) + '.png')
            plt.close()
