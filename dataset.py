import torch as th
import numpy as np
import torchbox as tb

drange = [0, 255]


def readdata(datafile, key=['UCD', 'PE', 'EU', 'EF'], index=None):
    """
    读取数据文件，并返回key列表中对应的数据。
    如果某个key在数据中不存在，则返回None。

    参数：
        datafile (str): 数据文件路径，支持.mat和.h5/.hdf5格式
        key (list): 要读取的键列表，默认为 ['UCD', 'PE', 'EU', 'EF']
        index (list或None): 如果不为None，则对读取的数据进行切片操作，
                             格式为 [start, end, step] 或 [start, end]

    返回：
        tuple：依次返回key列表对应的数据（如果数据不存在，则对应返回None）
    """
    ext = datafile[datafile.rfind('.'):]
    if ext == '.mat':
        data = tb.loadmat(datafile)
    elif ext in ['.h5', '.hdf5']:
        data = tb.loadh5(datafile)
    else:
        raise ValueError("不支持的文件格式")

    results = []
    for k in key:
        # 如果数据中不存在该键，则返回None
        results.append(data[k] if k in data else None)
    del data

    if index is not None:
        # 对所有不为None的数据项进行切片
        # 若index[1]为-1，则自动使用数据的第一维大小（取第一个不为None的项）
        if index[1] == -1:
            for r in results:
                if r is not None:
                    index[1] = r.shape[0]
                    break
            else:
                raise ValueError("没有数据用于切片")
        if len(index) > 2:
            idx = slice(index[0], index[1], index[2])
        else:
            idx = slice(index[0], index[1])
        results = [r[idx] if r is not None else None for r in results]
    return tuple(results)


def readsamples(datafiles, keys=[['UCD', 'PE', 'EU', 'EF']], nsamples=[10], mode='sequentially', seed=None):
    """
    从给定的数据文件列表中读取样本。

    参数：
        datafiles (list): 数据文件路径列表
        keys (list): 每个文件对应的键列表。默认为 [['UCD', 'PE', 'EU', 'EF']]
        nsamples (list): 每个文件要采样的样本数（总数）。如果只有一个数字，则对所有文件使用同一数目。
        mode (str): 采样模式，可选 'sequentially'（顺序采样）、'uniformly'（均匀采样）或 'randomly'（随机采样）。
        seed (int或None): 随机种子，仅在mode为'randomly'时有效。

    返回：
        tuple: 至少返回必需数据 (Xs, pes)，以及可选数据 (eus, efs)，
               如果EU和EF未提供，则对应返回None。
    """
    nfiles = len(datafiles)
    # 如果只给出一个keys，则复制到每个文件
    if len(keys) == 1:
        keys = keys * nfiles
    if len(nsamples) == 1:
        nsamples = nsamples * nfiles

    Xs, pes = th.tensor([]), th.tensor([])  # 初始化Xs和pes
    eus, efs = None, None  # 初始化为None，如果有对应数据则拼接

    for datafile, key, n in zip(datafiles, keys, nsamples):
        # 读取数据
        data_tuple = readdata(datafile, key=key, index=None)
        # 根据返回数据的个数赋值，至少需要UCD和 PE
        if len(data_tuple) == 2:
            UCD, PE = data_tuple
            EU, EF = None, None
        elif len(data_tuple) == 3:
            UCD, PE, EU = data_tuple
            EF = None
        elif len(data_tuple) == 1:
            SI_cleaned = data_tuple
        else:
            UCD, PE, EU, EF = data_tuple

        # 必须有UCD和PE
        if UCD is None or PE is None:
            raise ValueError("数据文件中必须包含'UCD'和'PE'两个键的数据！")

        N = UCD.shape[0]
        if n > N:
            raise ValueError('请求的样本数超过文件中的样本数')

        idx = []
        if mode.lower() == 'sequentially':
            idx = list(range(n))  # 顺序采样
        elif mode.lower() == 'uniformly':
            step = N // n
            idx = list(range(0, N, step))[:n]  # 均匀采样
        elif mode.lower() == 'randomly':
            tb.setseed(seed)
            idx = tb.randperm(0, N, n)  # 随机采样
        else:
            raise ValueError("未知的采样模式")

        # 拼接各文件的采样数据，转换为tensor
        Xs = th.cat((Xs, th.from_numpy(UCD[idx])), axis=0)
        pes = th.cat((pes, th.from_numpy(PE[idx])), axis=0)

        if EU is not None:
            if eus is None:
                eus = th.from_numpy(EU[idx])
            else:
                eus = th.cat((eus, th.from_numpy(EU[idx])), axis=0)
        if EF is not None:
            if efs is None:
                efs = th.from_numpy(EF[idx])
            else:
                efs = th.cat((efs, th.from_numpy(EF[idx])), axis=0)

    return Xs, pes


def readdatas(datafiles, keys=[['SI_cleaned', 'ca', 'cr']], indexes=[None]):
    if type(datafiles) is str:
        return readdata(datafiles, key=keys[0], index=indexes[0])
    if type(datafiles) is tuple or list:
        Xs, pas, prs = th.tensor([]), th.tensor([]), th.tensor([])
        for datafile, key, index in zip(datafiles, keys, indexes):
            X, pa, pr = readdata(datafile, key=key, index=index)
            Xs = th.cat((Xs, th.from_numpy(X)), axis=0)
            pas = th.cat((pas, th.from_numpy(pa)), axis=0)
            prs = th.cat((prs, th.from_numpy(pr)), axis=0)

        return Xs, pas, prs


def get_samples(datafile, nsamples=10000, region=None, size=(512, 512), index=None, seed=2020):
    if datafile[datafile.rfind('.'):] == '.mat':
        data = tb.loadmat(datafile)
    if datafile[datafile.rfind('.'):] in ['.h5', '.hdf5']:
        data = tb.loadh5(datafile)
    SI = data['SI']

    del data

    if np.ndim(SI) == 3:
        SI = SI[np.newaxis, :, :, :]

    if region is not None:
        SI = SI[:, region[0]:region[1], region[2]:region[3], :]

    SI = _sample(SI, nsamples, size, index, seed)
    return SI


def _sample(SI, nsamples, size, index, seed=None):
    To = []
    N, Na, Nr, _ = SI.shape
    if nsamples < N:
        N = 1
    num_each = int(nsamples / N)
    tb.setseed(seed)
    for n in range(N):
        imgsize = SI[n].shape  # H-W-2
        if index is None:
            ys = tb.randperm(0, imgsize[0] - size[0] + 1, num_each)
            xs = tb.randperm(0, imgsize[1] - size[1] + 1, num_each)
        else:
            ys, xs = index[0], index[1]
        for k in range(num_each):
            To.append(SI[n, ys[k]:ys[k] + size[0], xs[k]:xs[k] + size[1], :])

    To = np.array(To)

    # N-2-H-W
    SI = th.tensor(To, dtype=th.float32, requires_grad=False)
    return SI


def saveimage(X, Y, idx, prefixname='train', outfolder='./snapshot/'):
    if X.dim() == 3:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    X = X.pow(2).sum(-1).sqrt()
    Y = Y.pow(2).sum(-1).sqrt()

    X, Y = tb.mapping(X, drange=(0, 255), mode='amplitude', method='3Sigma'), tb.mapping(Y, drange=(0, 255),
                                                                                         mode='amplitude',
                                                                                         method='3Sigma')
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    for i, ii in zip(range(len(idx)), idx):
        outfileX = outfolder + prefixname + '_unfocused' + str(ii) + '.tif'
        outfileY = outfolder + prefixname + '_focused' + str(ii) + '.tif'
        tb.imsave(outfileX, X[i])
        tb.imsave(outfileY, Y[i])


if __name__ == "__main__":
    pass
