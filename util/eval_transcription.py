import argparse
import sys, os, glob
import numpy as np


def evaluate(target, predicted, start=0, end=0):
    y, p = (np.load(f) for f in (target, predicted))
    if y.shape[0] == 128:
        y = y[21: 109, :]
    if p.shape[0] == 128:
        p = p[21: 109, :]
    _, w = y.shape
    if start or end:
        assert(end > start)
    if end:
        if abs(end) >= w:
            raise Exception('--end <END> must be less than full length')
        else:
            target = target[:, : end]
            predicted = predicted[:, : end]
    if start:
        if abs(start) >= w:
            raise Exception('--start <START> must be less than full length')
        else:
            target = target[:, start:]
            predicted = predicted[:, end]

    try:
        p, r, f = calculate_performance(y, p)
        return p, r, f
    except Exception as e:
        print(
            '[WARNING] could not calculate performance for pair (\n\t{}\n\t{}\n) because of exception: {}'.format(
                target, predicted, e
            ), file=sys.stderr
        )
        return None, None, None


def calculate_performance(y, p):
    if y.shape[0] != p.shape[0]:
        raise Exception('target and predicted dim 0 size must match')
    if y.shape[1] != p.shape[1]:
        if abs(y.shape[1] - p.shape[1])/p.shape[1] > 0.005:
            print('warning: target and predicted dim 1 differ by {}. cropping...'.format(
                y.shape[1] - p.shape[1]
            ), file=sys.stderr)
            crop_len = min(y.shape[1], p.shape[1])
            if y.shape[1] > p.shape[1]:
                y = y[:, : crop_len]
            else:
                p = p[:, : crop_len]
        else:
            print('warning: target and predicted dim 1 differ by {}. resampling...'.format(
                   y.shape[1] - p.shape[1]
            ), file=sys.stderr)
            if y.shape[1] > p.shape[1]:
                bigger, smaller = y.shape[1], p.shape[1]
                y = y[:, np.minimum(
                          np.round(np.arange(0, bigger, bigger/smaller)).astype(int),
                          bigger - 1
                      )]
                if y.shape[1] != p.shape[1]:
                    delta = y.shape[1] - p.shape[1]
                    p = np.pad(p, ((0, 0), (delta, 0)), 'constant')
            else:
                bigger, smaller = p.shape[1], y.shape[1]
                p = p[:, np.minimum(
                        np.round(np.arange(0, bigger, bigger/smaller)).astype(int),
                        bigger - 1
                    )]
                if y.shape[1] != p.shape[1]:
                    delta = p.shape[1] - y.shape[1]
                    y = np.pad(y, ((0, 0), (delta, 0)), 'constant')

    y_hat = p.astype(int)
    y = y.astype(int)
    tp, fp, fn = count_performance(y, y_hat)

    p, r = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
    f = 2*p*r/(p + r + 1e-10)

    return p, r, f


'''
y: (88, T)
p: (batch, 88, T)
'''
def calculate_batch_performance(y, p):
    if y.shape[0] != p.shape[1]:
        raise Exception('target dim 0 and predicted dim 1 size must match')
    batch_size, _, t_max = p.shape
    y = np.stack([y for _ in range(batch_size)], axis=0)

    if y.shape[2] != p.shape[2]:
        if abs(y.shape[2] - p.shape[2]) / y.shape[2] > 0.05:
            print('warning: target and predicted sequence length differ by {}. skipping'.format(
                y.shape[2] - p.shape[2]
            ), file=sys.stderr)
            return None, None, None
        else:
            delta = abs(y.shape[2] - p.shape[2])
            if y.shape[2] > p.shape[2]:
                p = np.pad(p, ((0, 0), (0, 0), (delta // 2, delta - delta // 2)), 'constant')
            else:
                y = np.pad(y, ((0, 0), (0, 0), (delta // 2, delta - delta // 2)), 'constant')

    p = p.astype(float)
    tp, fp, fn = count_batch_performance(y, p)  # (batch)

    p, r = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
    f = 2*p*r/(p + r + 1e-10)

    return p, r, f


def count_performance(target, predicted):
    pred_minus_targ = predicted - target
    fp = (pred_minus_targ > 0).astype(int)
    fn = (pred_minus_targ < 0).astype(int)
    tp = target * predicted
    counts = tuple((t.sum()for t in (tp, fp, fn)))
    return counts


'''
y: (batch, 88, T)
p: (batch, 88, T)
'''
def count_batch_performance(target, predicted):
    pred_minus_targ = predicted - target
    fp = (pred_minus_targ == 1).astype(int)
    fn = (pred_minus_targ == -1).astype(int)
    tp = target * predicted
    counts = tuple((t.sum(axis=(1, 2))for t in (tp, fp, fn)))
    return counts


def main(args):
    target = os.path.join(os.getcwd(), args.gt)
    transcription = os.path.join(os.getcwd(), args.pred)

    p, r, f = evaluate(target, transcription)
    # print('P: {:.4}\nR: {:.4}\nF: {:.4}'.format(p, r, f))
    bn = os.path.splitext(os.path.splitext(os.path.splitext(os.path.basename(transcription))[0])[0])[0]

    print('{:.4},{:.4},{:.4}'.format(p, r, f), end='')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pred')
    ap.add_argument('gt')

    args = ap.parse_args()
    main(args)
