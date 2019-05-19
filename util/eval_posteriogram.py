import argparse
import os, glob, sys
import numpy as np


def count_performance(target, predicted):
    pred_minus_targ = predicted - target
    fp = (pred_minus_targ > 0).astype(int)
    fn = (pred_minus_targ < 0).astype(int)
    tp = target * predicted
    counts = tuple((t.sum()for t in (tp, fp, fn)))
    return counts


def evaluate(predicted, target, threshold, start=0, end=0):
    y, p = (np.load(f) for f in (target, predicted))
    if p.shape[0] == 128:
        p = p[21: 109, :]
    if y.shape[0] == 128:
        y = y[21: 109, :]

    if start and end:
        assert(end > start)
    if end:
        y = y[:, :end]
    if start:
        y = y[:, start:]

    if y.shape[0] != p.shape[0]:
        print('error: target and predicted dim 0 must match', file=sys.stderr)
        exit(1)
    if y.shape[1] != p.shape[1]:
        if abs(y.shape[1] - p.shape[1]) / y.shape[1] > 0.05:
            print('error: target and predicted dim 1 differ by {}. skipping'.format(
                y.shape[1] - p.shape[1]), file=sys.stderr)
            exit(1)
        else:
            print('warning: target and predicted dim 1 differ by {}'.format(
                y.shape[1] - p.shape[1], file=sys.stderr
            ))

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

    y = y.astype(int)
    y_hat = (p >= threshold).astype(int)
    tp, fp, fn = count_performance(y, y_hat)
    p, r = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)

    return p, r, f


def main(args):
    # print('evaluating {}'.format(args.pred), file=sys.stderr)
    p, r, f = evaluate(args.pred, args.gt, args.threshold)
    # print('P: {:.4}\nR: {:.4}\nF: {:.4}\n'.format(p, r, f), file=sys.stderr)

    print('{:.4},{:.4},{:.4}'.format(p, r, f), end='')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pred')
    ap.add_argument('gt')
    ap.add_argument('--threshold', '-t', type=float, default=0.5)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=0)
    args = ap.parse_args()
    main(args)
