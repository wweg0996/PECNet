import time
from dataset import saveimage
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from collections import OrderedDict
import torchbox as tb


class Focuser(nn.Module):
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

        # 这里你可以选择只对宽度方向做池化 (None,1) 等
        FD['gapool'] = nn.AdaptiveAvgPool2d((None, 1))

        self.features = nn.Sequential(FD)

        # ========== Transformer 设置 ========== #
        # 让 Transformer 的 d_model 与卷积输出通道数相同 (last_channels)
        last_channels = convp[-1][0]

        # 创建一个 TransformerEncoder (示例:2层, 4头, FFN=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=last_channels,  # 输入向量维度
            nhead=4,
            dim_feedforward=512,
            batch_first=True  # 重要: 保证输入形状为(N, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # 用一个简单的 1D pooling 将 Transformer 输出聚合成全局向量
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ========== 相位预测层 ========== #
        self.phase_predictor = nn.Sequential(
            nn.Linear(last_channels, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, Na)  # 输出 [N, Na]
        )
        # self.phase_predictor = nn.Sequential(
        #     nn.Linear(last_channels, 128),  # 可选隐藏层
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(128, 1)  # 关键：每个时刻输出 1 个标量相位
        # )

        if self.seed is not None:
            th.manual_seed(seed)

    def forward(self, X):
        """
        X: [N, H, W, 2] (假设)，最后一维 (实部, 虚部)
        """
        d = X.dim()  # 一般=4
        # => [N, 2, H, W]
        Y = th.stack((X[..., 0], X[..., 1]), dim=1)

        # 1) 卷积特征提取 => [N, C, H', 1]
        feats = self.features(Y)
        feats = feats.squeeze(dim=-1)  # => [N, C, H']
        # => [N, H', C] (batch_first=True)
        feats = feats.permute(0, 2, 1)

        # 2) TransformerEncoder => [N, H', C]
        out = self.transformer(feats)

        # 3) 池化 => 先变为 [N, C, H'] 再做自适应平均池化到 1
        out = out.permute(0, 2, 1)  # => [N, C, H']
        out = self.pool(out)  # => [N, C, 1]
        out = out.squeeze(-1)  # => [N, C]

        # 4) 相位预测 => [N, Na]
        pa = self.phase_predictor(out)
        # pa = self.phase_predictor(out)  # [N, H', 1]
        # pa = pa.squeeze(-1)  # [N, H']  (即 [N, Na])

        # 5) 构造相位校正 e^{j·theta} 并与 X 做 FFT/乘法
        sizea = [1] * d
        sizea[0], sizea[-3], sizea[-2], sizea[-1] = pa.size(0), pa.size(1), 1, 2
        epa = th.stack((th.cos(pa), -th.sin(pa)), dim=-1)
        epa = epa.reshape(sizea)

        X = tb.ifft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=self.ftshift)
        X = tb.ematmul(X, epa, cdim=-1)
        X = tb.fft(X, n=None, cdim=-1, dim=-3, keepcdim=True, norm=None, shift=self.ftshift)

        return X, pa

    def weights_init(self, m):
        if isinstance(m, th.nn.Conv2d):
            th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
            m.bias.data.zero_()
        if isinstance(m, th.nn.Linear):
            th.nn.init.orthogonal_(m.weight.data, th.nn.init.calculate_gain('leaky_relu'))
            m.bias.data.zero_()

    def train_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, optimizer,
                    scheduler, device):
        self.train()

        tstart = time.time()
        numSamples = X.shape[0]

        numBatch = int(numSamples / sizeBatch)
        idx = tb.randperm(0, numSamples, numSamples)
        lossENTv, lossCTSv, lossFROv, lossvtrain = 0., 0., 0., 0.
        # t1, t2, t3 = 0., 0., 0.
        for b in range(numBatch):
            # tstart1 = time.time()
            i = idx[b * sizeBatch:(b + 1) * sizeBatch]
            xi = X[i]
            xi = xi.to(device)

            optimizer.zero_grad()
            # x,pre
            fi, casi = self.forward(xi)
            # tend1 = time.time()

            # tstart2 = time.time()
            lossENT = loss_ent_func(fi)
            lossCTS = loss_cts_func(fi)
            lossFRO = loss_fro_func(fi)
            # tend2 = time.time()

            if loss_type == 'Entropy':
                loss = lossENT
            if loss_type == 'Entropy+LogFro':
                loss = lossENT + lossFRO
            if loss_type == 'Contrast':
                loss = lossCTS
            if loss_type == 'Entropy/Contrast':
                loss = lossENT / lossCTS

            loss.backward()

            # tstart3 = time.time()
            lossvtrain += loss.item()
            lossCTSv += lossCTS.item()
            lossENTv += lossENT.item()
            lossFROv += lossFRO.item()
            # tend3 = time.time()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # t1 += tend1 - tstart1
            # t2 += tend2 - tstart2
            # t3 += tend3 - tstart3

        tend = time.time()

        lossvtrain /= numBatch
        lossCTSv /= numBatch
        lossENTv /= numBatch
        lossFROv /= numBatch
        # print(t1, t2, t3)
        print("--->Train epoch: %d, loss: %.4f, entropy: %.4f, l1norm: %.4f, contrast: %.4f, time: %ss" %
              (epoch, lossvtrain, lossENTv, lossFROv, lossCTSv, tend - tstart))
        return lossvtrain

    def valid_epoch(self, X, sizeBatch, loss_ent_func, loss_cts_func, loss_fro_func, loss_type, epoch, device):
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
