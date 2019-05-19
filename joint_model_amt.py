import argparse, sys, math, os
import pickle
from collections import defaultdict
import numpy as np
from heapq import heappop, heappush
from itertools import count

import torch
import torch.nn.functional as F


class Beam(object):
    def __init__(self, capacity):
        super(Beam, self).__init__()
        self.counter = count()
        self.hq = defaultdict(lambda : [])
        self.q = []
        self.cap = capacity

    def insert(self, nll, key, seq):
        q = self.q
        hq = self.hq[tuple(key.flatten().tolist())]
        fits_in_q = len(q) < self.cap or nll < q[-1][0]
        fits_in_hq = len(hq) < self.cap or nll < hq[-1][0]
        if fits_in_q and fits_in_hq:
            heappush(hq, (nll, next(self.counter), key, seq))
            self.hq[key] = hq[:self.cap]
            heappush(q, (nll, next(self.counter), key, seq))
            self.q = q[:self.cap]

    def __getitem__(self, idx):
        nll, _, key, seq = self.q[idx]
        return nll, key, seq

    def __len__(self):
        return len(self.q)


def beam_search(
        posterior, mlm, acoustic_threshold=0.5, win_len=1024,
        beam_w=16, branch=8, hash_len=10, cuda_dev=None, arch='baseline'
):
    dev = 'cpu' if cuda_dev is None else 'cuda:{}'.format(cuda_dev)
    posterior = posterior.unsqueeze(0).unsqueeze(0)
    acoustic_predictions = (posterior > acoustic_threshold).type(torch.FloatTensor).to(device=dev)
    nonzero = (acoustic_predictions != 0).nonzero()
    first_ac_detection = nonzero[:, -1].min().item()
    starter = posterior[:, :, :, max(0, first_ac_detection - win_len): first_ac_detection + 1]

    beam = Beam(beam_w)
    num_s, hashes, samples = sample_from_multihot(
        starter[:, :, :, -1].unsqueeze(-1), branch, threshold=acoustic_threshold, cuda_dev=cuda_dev
    )

    for i in range(num_s):
        sample = samples[i].unsqueeze(0)
        seq = torch.cat((acoustic_predictions[:, :, :, : first_ac_detection], sample), dim=-1)
        beam.insert(0., hashes[i].unsqueeze(0), seq)

    t_init, t_end = first_ac_detection + 1, posterior.shape[-1]

    for t in range(t_init, t_end):
        acoustic_frame_p = posterior[:, :, :, t].unsqueeze(-1)
        num_s, hashes, samples = sample_from_multihot(
            acoustic_frame_p, branch, threshold=acoustic_threshold, cuda_dev=cuda_dev
        )

        prev_nlls, hss, seqs = zip(*beam)
        hss = torch.cat([h.repeat(num_s, 1) for h in hss], dim=0)
        new_hss = torch.cat((hss, hashes.repeat(len(beam), 1)), dim=-1)
        seqs = torch.cat([s.repeat(num_s, 1, 1, 1) for s in seqs], dim=0)
        proposed_seqs = torch.cat((seqs, samples.repeat(len(beam), 1, 1, 1)), dim=-1)
        cropped_seqs = proposed_seqs[:, :, :, max(0, t + 1 - win_len):]
        cropped_seqs = F.pad(cropped_seqs, (max(0, win_len - cropped_seqs.shape[-1]), 0))
        z_mlm = mlm(cropped_seqs)
        if arch == 'baseline':
            z_mlm = z_mlm.unsqueeze(1)

        beam_next = Beam(beam_w)
        nlls = F.binary_cross_entropy_with_logits(
            z_mlm, cropped_seqs[:, :, :, win_len//2:], reduction='none'
        )
        nlls = nlls[:, 0, :, -1]
        nlls = nlls.sum(dim=-1)

        for j in range(nlls.shape[0]):
            p_nll, c_nll = prev_nlls[j//num_s], nlls[j].item()
            beam_next.insert(
                p_nll + c_nll,
                new_hss[j],
                proposed_seqs[j].unsqueeze(0)
            )
        del beam
        beam = beam_next

        if t > 0 and t % 100 == 0:
            print('{} of {} timeframes processed'.format(t, posterior.shape[-1]))
            sys.stdout.flush()

    nll, _, seq = beam[0]
    return nll, seq[0, 0]


def hash_bin_88_tensor(t, dev):
    assert(t.shape == (88,))
    t_split = torch.stack(t.split(44), dim=0)
    base = 2*torch.ones(2, 44, dtype=torch.long, device=dev)
    xp = torch.arange(0, 44, dtype=torch.long, device=dev).unsqueeze(0).repeat(2, 1)
    hash_vecs = t_split*torch.pow(base, xp)
    hash_sums = torch.sum(hash_vecs, dim=1)
    return hash_sums


def sample_from_multihot(frame, k_distinct, k_max=1000, threshold=0.5, cuda_dev=None):
    dev = 'cpu' if cuda_dev is None else 'cuda:{}'.format(cuda_dev)
    u = torch.distributions.Uniform(torch.zeros(k_max, *frame.shape[1:]), 2.*threshold*torch.ones(*frame.shape[1:]))
    samples = u.sample().to(device=dev)
    samples = (frame.repeat(k_max, 1, 1, 1) > samples).type(torch.LongTensor).to(device=dev)

    used, num_used = torch.zeros(k_distinct, 1, 88, 1, device=dev), 0
    hash_set = torch.ones(k_distinct, 2, dtype=torch.long, device=dev)*(-1)
    for i in range(k_max):
        if num_used >= k_distinct:
            break
        s = samples[i].unsqueeze(0)

        hash = hash_bin_88_tensor(s.squeeze(-1).squeeze(0).squeeze(0), dev)

        if num_used > 0:
            hash_rep = hash.repeat(num_used, 1)
            if not torch.all(torch.any(torch.ne(hash_rep, hash_set[:num_used]), dim=1)):
                del hash, hash_rep
                continue

        used[num_used, :, :, :] = s
        hash_set[num_used, :] = hash
        num_used += 1

    del samples
    return num_used, hash_set[:num_used, :], used[:num_used, :, :, :]


def calc_nll(prob, bin_act):
    lh = bin_act*prob + (1 - bin_act)*(1. - prob)
    return -torch.log(lh)


def main(opts):
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # initialize the requested MLM
    if opts.arch == 'crnn':
        from conv_v5.crnn_v5_1 import LanguageModeler
        net = LanguageModeler(batch_size=opts.batch_size, rnn_size=opts.rnn_size, 
                              rnn_layers=1, use_cuda=opts.use_cuda, max_w=opts.max_window)

    elif opts.arch == 'baseline':
        from baseline import LanguageModeler
        net = LanguageModeler(rnn_size=opts.rnn_size, rnn_layers=1)

    elif opts.arch == 'cnn':
        from amt_cnn_8_0_1 import AMT_CNN as LanguageModeler
        net = LanguageModeler(batch_size=opts.batch_size, use_cuda=opts.use_cuda, max_w=opts.max_window)

    print('using model architecture {}'.format(net.name), file=sys.stderr)
    sys.stderr.flush()

    # load model weights
    saved_state = torch.load(opts.model_weights, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    for param in net.parameters():
        param.requires_grad = False

    # load the acoustic model output
    post_path = os.path.join(os.getcwd(), opts.posteriogram)
    posteriogram = np.load(post_path)  # matrix of pitch activations at each timeframe
    posteriogram = posteriogram/posteriogram.max()  # normalize to get valid prob. dist.
    if opts.start or opts.end:
        assert(opts.end > opts.start)
    if opts.end:
        if abs(opts.end) >= posteriogram.shape[-1]:
            raise Exception('--end <END> must be less than full length')
        else:
            posteriogram = posteriogram[:, : opts.end]
    if opts.start: 
        if abs(opts.start) >= posteriogram.shape[-1]:
            raise Exception('--start <START> must be less than full length')
        else:
            posteriogram = posteriogram[:, opts.start:]
    
    posteriogram = torch.from_numpy(posteriogram)

    # threshold the posteriogram activations
    acoustic_predictions = posteriogram.type(torch.FloatTensor)

    if cuda_dev is not None:
        acoustic_predictions = acoustic_predictions.cuda(cuda_dev)

    left_padding = 0
    output_path = os.path.join(os.getcwd(), opts.output)  # path to save the joint model's transcription

    with torch.no_grad():
        nll, sequence = beam_search(
            acoustic_predictions, net, acoustic_threshold=opts.a_thresh,
            beam_w=opts.beam_w, branch=opts.branch, cuda_dev=cuda_dev, 
            win_len=opts.max_window//2, arch=opts.arch
        )
        sequence = sequence.cpu().numpy()
        np.save(output_path, sequence)
        print('saving transcription to {}'.format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-p", "--posteriogram")
    parser.add_argument("--a_thresh", type=float, default=0.045)
    parser.add_argument("-w", "--max_window", type=int, default=1024)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-m", "--model_weights", default=None)
    parser.add_argument("--rnn_size", type=int, default=1024)
    parser.add_argument("-a", "--arch", default="baseline")
    parser.add_argument("--output", default="transcription.npy")
    parser.add_argument("-g", "--gt", default=None)
    parser.add_argument("-b", "--beam_w", type=int, default=16)
    parser.add_argument("--branch", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()
    print('arguments:\n{}\n'.format(args.__dict__))

    main(args)
