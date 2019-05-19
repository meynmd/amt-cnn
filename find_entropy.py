'''
script to track entropy values of predicted pitch activations at each time frame (a potentially interesting application
of mlm)
'''
import argparse
import os
import sys
import math
import numpy as np

import torch
import torch.nn.functional as F


def crop_and_batch(orig_tensor, window_len, stride, max_batch):
    _, w = orig_tensor.shape
    assert(w >= window_len)
    rem = w % stride
    num_padding = 0
    if rem != 0:
        num_padding = stride - rem
        orig_tensor = F.pad(orig_tensor, (num_padding, 0))
        _, w = orig_tensor.shape

    start, stop = 0, window_len
    cropped_tensors = [orig_tensor[:, start : stop].unsqueeze(0)]
    while start < w - window_len:
        start += stride
        stop += stride
        cropped_tensors.append(orig_tensor[:, start: stop].unsqueeze(0))

    assert (stop == w)

    batches = []
    if len(cropped_tensors) > max_batch:
        num_batches = int(math.ceil(len(cropped_tensors) / max_batch))
        for i in range(num_batches):
            b = cropped_tensors[i*max_batch : min((i + 1)*max_batch, len(cropped_tensors))]
            batches.append(torch.stack(b))
    else:
        batches = [torch.stack(cropped_tensors)]

    return num_padding, batches


def calc_entropy(pred_batch):
    batch_entropies = []    # torch.zeros(pred_batch.shape[0], pred_batch.shape[-1])
    for j in range(pred_batch.shape[0]):
        p = pred_batch[j, 0]
        p = torch.max(p, torch.ones_like(p)*1e-9)
        p = torch.min(p, torch.ones_like(p)*(1.-1e-9))
        entropies = -p*torch.log(p) - (1. - p)*torch.log(1. - p)
        entropies = torch.sum(entropies, dim=0)
        batch_entropies.append(entropies)

    return torch.cat(batch_entropies, dim=0)


def find_entropies(net, x, cuda_dev, batch_size, max_w=1024):
    if x.shape[1] > max_w:
        padding, x_batches = crop_and_batch(x, max_w, max_w//2, batch_size)
    else:
        if x.shape[1] < max_w:
            x = F.pad(x, (max_w - x.shape[1], 0))
        x_batches = [x.unsqueeze(0).unsqueeze(0)]
        padding = 0

    entropies = []
    for j, batch in enumerate(x_batches):
        if cuda_dev:
            batch = batch.cuda(cuda_dev)
        predicted = torch.sigmoid(net(batch))
        ent = calc_entropy(predicted)
        entropies.append(ent)

    entropies = torch.cat(entropies, dim=0)

    return entropies[padding:]


def main(opts):
    # set cuda device
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # load and prepare piano roll matrix
    piano_roll_path = os.path.join(os.getcwd(), opts.target)
    piano_roll_array = np.load(piano_roll_path)
    pr_tensor = torch.from_numpy(piano_roll_array)
    pr_tensor = (pr_tensor > 0).type(torch.FloatTensor)

    print(pr_tensor.shape)

    if pr_tensor.shape[0] == 128:
        pr_tensor = pr_tensor[21: 109, :]

    if opts.len:
        pr_tensor = pr_tensor[:, : opts.len]

    left_pad = opts.window//2
    pr_tensor = F.pad(pr_tensor, (left_pad, 0), 'constant', 0.)

    # initialize network
    if opts.arch == 'baseline':
        import baseline
        net = baseline.LanguageModeler(rnn_size=opts.rnn_size, rnn_layers=1)

    elif opts.arch == 'cnn':
        import amt_cnn_8_0_1 as cnn
        net = cnn.AMT_CNN(use_cuda=opts.use_cuda, max_w=opts.window)
    else:
        raise NotImplementedError('for --arch, valid options are \'cnn\' or \'baseline\'')

    saved_state = torch.load(opts.model_weights, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    for param in net.parameters():
        param.requires_grad = False

    with torch.no_grad():
        entropies = find_entropies(net, pr_tensor, cuda_dev, opts.batch_size, max_w=opts.window)
        entropies = entropies.cpu().numpy()

        if opts.save:
            np.save(os.path.join(os.getcwd(), opts.save), entropies)
        else:
            print(','.join(['{:.4}'.format(entropies[i]) for i in range(entropies.shape[0])]))

        print(entropies.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--window", type=int, default=4096)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-m", "--model_weights", default=None)
    parser.add_argument("-t", "--target")
    parser.add_argument("-a", "--arch", default="cnn")
    parser.add_argument("-s", "--save", default=None)
    parser.add_argument("--len", type=int, default=None)
    args = parser.parse_args()

    print(args.__dict__)
    sys.stdout.flush()

    main(args)