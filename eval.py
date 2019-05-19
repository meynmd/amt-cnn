# evaluation script
# calculates P,R,F for all frames, for onsets and for offsets
# if no threshold specified, will first search ROC curve on validation set to maximize harmonic mean of F-measures for
# onset and offset

import os, sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import amt_dataset

def crop_and_batch(orig_tensor, window_len, stride, max_batch):
    _, w = orig_tensor.shape
    assert(w >= window_len)
    rem = w % stride
    if rem != 0:
        orig_tensor = F.pad(orig_tensor, (stride - rem, 0))
        _, w = orig_tensor.shape

    start, stop = 0, window_len
    cropped_tensors = [orig_tensor[:, start : stop].unsqueeze(0)]
    while start < w - window_len:
        start += stride
        stop += stride
        cropped_tensors.append(orig_tensor[:, start : stop].unsqueeze(0))

    assert (stop == w)

    batches = []
    if len(cropped_tensors) > max_batch:
        num_batches = int(math.ceil(len(cropped_tensors) / max_batch))
        for i in range(num_batches):
            b = cropped_tensors[i*max_batch : min((i + 1)*max_batch, len(cropped_tensors))]
            batches.append(torch.stack(b))
    else:
        batches = [torch.stack(cropped_tensors)]

    return batches


'''
find PRF measures only on boundary points
'''
def count_on_off_performance(target, predicted):
    y = target.cpu().unsqueeze(1).type(torch.FloatTensor)
    yh = predicted.cpu().unsqueeze(1).type(torch.FloatTensor)

    f = torch.FloatTensor([[[[-1., 1., 0.]]]]).transpose(-2, -1)
    gt_mask = F.conv2d(y, f, padding=(1, 0)).squeeze(1)
    p_mask = F.conv2d(yh, f, padding=(1, 0)).squeeze(1)

    # mask:  1 means note-on, -1 means note-off
    gt_onsets, gt_offsets = (gt_mask > 0).type(torch.LongTensor), (gt_mask < 0).type(torch.LongTensor)
    pred_onsets, pred_offsets = (p_mask > 0).type(torch.LongTensor), (p_mask < 0).type(torch.LongTensor)

    tp_on = gt_onsets*pred_onsets
    fp_on = pred_onsets - tp_on
    fn_on = torch.max(gt_onsets - pred_onsets, torch.zeros(gt_onsets.shape).type(torch.LongTensor))

    tp_off = gt_offsets*pred_offsets
    fp_off = pred_offsets - tp_off
    fn_off = torch.max(gt_offsets - pred_offsets, torch.zeros(gt_offsets.shape).type(torch.LongTensor))

    return tuple((t.sum().item() for t in (tp_on, fp_on, fn_on, tp_off, fp_off, fn_off)))


def count_on_off_performance_batch(target, predicted):
    y = target.cpu().type(torch.FloatTensor)
    yh = predicted.cpu().type(torch.FloatTensor)

    f = torch.FloatTensor([[[[-1., 1., 0.]]]])
    gt_mask = F.conv2d(y, f, padding=(0, 1)).squeeze(1)
    p_mask = F.conv2d(yh, f, padding=(0, 1)).squeeze(1)

    # mask:  1 means note-on, -1 means note-off
    gt_onsets, gt_offsets = (gt_mask > 0).type(torch.LongTensor), (gt_mask < 0).type(torch.LongTensor)
    pred_onsets, pred_offsets = (p_mask > 0).type(torch.LongTensor), (p_mask < 0).type(torch.LongTensor)

    tp_on = gt_onsets*pred_onsets
    fp_on = pred_onsets - tp_on
    fn_on = torch.max(gt_onsets - pred_onsets, torch.zeros(gt_onsets.shape).type(torch.LongTensor))

    tp_off = gt_offsets*pred_offsets
    fp_off = pred_offsets - tp_off
    fn_off = torch.max(gt_offsets - pred_offsets, torch.zeros(gt_offsets.shape).type(torch.LongTensor))

    return tuple((t.sum(dim=2).sum(dim=1) for t in (tp_on, fp_on, fn_on, tp_off, fp_off, fn_off)))


'''
find (true pos, false pos, false negative), P&E counts for all frames and pitches
'''
def count_performance(target, predicted):
    y, yh = target.cpu(), predicted.cpu()
    pred_minus_targ = yh - y
    fp = (pred_minus_targ == 1).type(torch.LongTensor)
    fn = (pred_minus_targ == -1).type(torch.LongTensor)
    tp = yh * y
    counts = tuple((t.sum().item() for t in (tp, fp, fn)))

    # P&E error
    n_ref, n_sys = torch.sum(y, dim=1), torch.sum(yh, dim=1)
    err_count = torch.sum(torch.max(n_ref, n_sys) - torch.sum(tp, dim=1)).item()
    subst_count = torch.sum(torch.min(n_ref, n_sys) - torch.sum(tp, dim=1)).item()

    zero = torch.zeros(n_ref.shape).type(torch.LongTensor)
    miss_count = torch.sum(torch.max(zero, n_ref - n_sys)).item()
    fa_count = torch.sum(torch.max(zero, n_sys - n_ref)).item()

    n_ref_count = torch.sum(y).item()

    return counts, (err_count, subst_count, miss_count, fa_count, n_ref_count)


'''
find precision, recall, f-measure

with this batching method, stride has to be set to max_w, otherwise calculations will not be quite correct
'''
def test(
        net, data, max_w,
        batch_size=32, padding=0,
        threshold=0.5, cuda_dev=None
):
    net.eval()
    stride = max_w//2
    labels = data.idx2name.items()
    p_total, r_total, f_total, err_t_total, err_s_total, err_m_total, err_f_total, num_pr_total = \
        0., 0., 0., 0., 0., 0., 0., 0
    on_p_total, on_r_total, on_f_total, off_p_total, off_r_total, off_f_total = 0., 0., 0., 0., 0., 0.

    for i, label in labels:
        class_datapoints = list(data.get_from_class(label))
        num_pr_total += len(class_datapoints)

        for (x, y), (composer, x_path) in class_datapoints:
            # tf_per_batch = batch_size*stride
            if padding:
                x = F.pad(x, (padding, 0))
            if x.shape[1] > max_w:
                x_batches = crop_and_batch(x, max_w, stride, batch_size)
            else:
                if x.shape[1] < max_w:
                    x = F.pad(x, (max_w - x.shape[1], 0))
                x_batches = [x.unsqueeze(0).unsqueeze(0)]

            x_tp, x_fp, x_fn, err_sum, subst_sum, miss_sum, fa_sum, n_ref_sum = 0., 0., 0., 0., 0., 0., 0., 0.
            x_true_onsets, x_false_offsets, x_false_onsets, x_missed_onsets, x_true_offsets, x_missed_offsets = \
                0., 0., 0., 0., 0., 0.
            for j, batch in enumerate(x_batches):
                y = batch.data.clone().type(torch.LongTensor)
                y = y[:, :, :, stride:].squeeze(1).transpose(1, 2)
                if cuda_dev is not None:
                    batch, y = batch.cuda(cuda_dev), y.cuda(cuda_dev)

                z = net(batch)
                p = torch.sigmoid(z.squeeze(1).transpose(1, 2))
                y_hat = (p >= threshold).type(torch.LongTensor)

                y = y[:, :, : -1]
                y_hat = y_hat[:, :, 1:]

                (batch_tp, batch_fp, batch_fn), (err_count, subst_count, miss_count, fa_count, n_ref_count) = \
                    count_performance(y, y_hat)

                tp_on, fp_on, fn_on, tp_off, fp_off, fn_off = count_on_off_performance(y, y_hat)

                x_tp += batch_tp
                x_fp += batch_fp
                x_fn += batch_fn
                # x_tn += num_tn

                x_true_onsets += tp_on
                x_false_onsets += fp_on
                x_missed_onsets += fn_on

                x_true_offsets += tp_off
                x_false_offsets += fp_off
                x_missed_offsets += fn_off

                err_sum += err_count
                subst_sum += subst_count
                miss_sum += miss_count
                fa_sum += fa_count

                n_ref_sum += n_ref_count

            # calculate P, R, F measure
            precision = x_tp/(x_tp + x_fp)
            recall = x_tp/(x_tp + x_fn)
            f_score = 2*precision*recall/(precision + recall + 1e-10)

            # calculate measures for note onsets, offsets
            on_precision = x_true_onsets/(x_true_onsets + x_false_onsets)
            on_recall = x_true_onsets/(x_true_onsets + x_missed_onsets)
            on_f_score = 2*on_precision*on_recall/(on_precision + on_recall + 1e-10)

            off_precision = x_true_offsets/(x_true_offsets + x_false_offsets)
            off_recall = x_true_offsets/(x_true_offsets + x_missed_offsets)
            off_f_score = 2*off_precision*off_recall/(off_precision + off_recall + 1e-10)

            err_total = err_sum/float(n_ref_sum)
            err_subs = subst_sum/float(n_ref_sum)
            err_miss = miss_sum/float(n_ref_sum)
            err_fa = fa_sum/float(n_ref_sum)

            p_total += precision
            r_total += recall
            f_total += f_score

            on_p_total += on_precision
            on_r_total += on_recall
            on_f_total += on_f_score
            off_p_total += off_precision
            off_r_total += off_recall
            off_f_total += off_f_score

            err_t_total += err_total
            err_s_total += err_subs
            err_m_total += err_miss
            err_f_total += err_fa

    print('\noverall stats:\nP: {:.4}\tR: {:.4}\tF: {:.4}'.format(
        p_total/num_pr_total, r_total/num_pr_total, f_total/num_pr_total)
    )
    print('\nerr_tot: {:.4}\terr_subs: {:.4}\terr_miss: {:.4}\terr_fa: {:.4}'.format(
        err_t_total/num_pr_total, err_s_total/num_pr_total, err_m_total/num_pr_total, err_f_total/num_pr_total
    ))

    print('\nonset precision: {:.4}\tonset recall: {:.4}\tonset f-measure: {:.4}'.format(
        on_p_total/num_pr_total, on_r_total/num_pr_total, on_f_total/num_pr_total
    ))

    print('\noffset precision: {:.4}\toffset recall: {:.4}\toffset f-measure: {:.4}'.format(
        off_p_total/num_pr_total, off_r_total/num_pr_total, off_f_total/num_pr_total
    ))

    return (p_total/num_pr_total, r_total/num_pr_total, f_total/num_pr_total), \
           (on_p_total/num_pr_total, on_r_total/num_pr_total, on_f_total/num_pr_total), \
           (off_p_total/num_pr_total, off_r_total/num_pr_total, off_f_total/num_pr_total)


'''
choose best threshold based on harmonic mean combination of onset and offset f-measure

with this batching method, stride has to be set to max_w, otherwise calculations will not be quite correct
'''
def roc_select(net, data, max_w, batch_size=32, cuda_dev=None, labels=None, num_per_class=None,
               increment=0.05, on_weight=0.5):
    net.eval()
    stride = max_w
    if labels is None:
        labels = data.idx2name.items()

    num_pr_total = 0.
    on_p_total, on_r_total, on_f_total, off_p_total, off_r_total, off_f_total = \
        (torch.zeros(int(1. / increment)) for _ in range(6))

    for i, label in labels:
        class_datapoints = list(data.get_from_class(label))

        if num_per_class is None:
            num_per_class = len(class_datapoints)
        else:
            num_per_class = min(num_per_class, len(class_datapoints))

        class_datapoints = class_datapoints[:num_per_class]
        num_pr_total += num_per_class

        for (x, y), _ in class_datapoints:
            net.eval()
            tf_per_batch = batch_size * stride
            if x.shape[1] > max_w:
                x_batches = crop_and_batch(x, max_w, stride, batch_size)
            else:
                if x.shape[1] < max_w:
                    x = F.pad(x, (max_w - x.shape[1], 0))
                x_batches = [x.unsqueeze(0).unsqueeze(0)]

            x_true_onsets, x_false_offsets, x_false_onsets, x_missed_onsets, x_true_offsets, x_missed_offsets = \
                (torch.zeros(int(1. / increment)) for _ in range(6))

            for j, batch in enumerate(x_batches):
                target = batch.clone().type(torch.LongTensor)
                if cuda_dev is not None:
                    batch, target = batch.cuda(cuda_dev), target.cuda(cuda_dev)
                z = net(batch)
                p = torch.sigmoid(z)

                y_hat = torch.cat([(p >= th).type(torch.FloatTensor) for th in np.arange(0., 1., increment)], dim=0)
                target = torch.cat([target for _ in range(int(1. / increment))], dim=0)

                tp_on, fp_on, fn_on, tp_off, fp_off, fn_off = count_on_off_performance_batch(target, y_hat)

                tp_on, fp_on, fn_on, tp_off, fp_off, fn_off = (t.view(int(1./increment), -1).sum(dim=1) \
                                                               for t in (tp_on, fp_on, fn_on, tp_off, fp_off, fn_off))

                x_true_onsets += tp_on.type(torch.FloatTensor)
                x_false_onsets += fp_on.type(torch.FloatTensor)
                x_missed_onsets += fn_on.type(torch.FloatTensor)

                x_true_offsets += tp_off.type(torch.FloatTensor)
                x_false_offsets += fp_off.type(torch.FloatTensor)
                x_missed_offsets += fn_off.type(torch.FloatTensor)

            # calculate measures for note onsets, offsets
            on_precision = x_true_onsets / (x_true_onsets + x_false_onsets + 1e-10)
            on_recall = x_true_onsets / (x_true_onsets + x_missed_onsets + 1e-10)
            on_f_score = 2 * on_precision * on_recall / (on_precision + on_recall + 1e-10)

            off_precision = x_true_offsets / (x_true_offsets + x_false_offsets + 1e-10)
            off_recall = x_true_offsets / (x_true_offsets + x_missed_offsets + 1e-10)
            off_f_score = 2 * off_precision * off_recall / (off_precision + off_recall + 1e-10)

            on_p_total += on_precision
            on_r_total += on_recall
            on_f_total += on_f_score
            off_p_total += off_precision
            off_r_total += off_recall
            off_f_total += off_f_score

    on_p_avg, on_r_avg, on_f_avg, off_p_avg, off_r_avg,off_f_avg = \
        (t/num_pr_total for t in (on_p_total, on_r_total, on_f_total, off_p_total, off_r_total,off_f_total))

    avg_f_hmean = 2*on_f_avg*off_f_avg/(on_f_avg + off_f_avg + 1e-10)

    print('\nthreshold          on F-measure      off F-measure     h. mean')
    for j in range(int(1. / increment)):
        print('{:.4f}              {:.4f}              {:.4f}              {:.4f}'.format(
            j*increment, on_f_avg[j].item(), off_f_avg[j].item(),
            avg_f_hmean[j].item()
        ))

    f_argmax = torch.argmax(avg_f_hmean, dim=0).item()
    best_threshold = f_argmax*increment
    print('choosing threshold {:.4} based on ROC curve calculated on validation set'.format(best_threshold))

    return best_threshold


def main(opts):
    if opts.use_cuda is not None:
        cuda_dev = int(opts.use_cuda)
    else:
        cuda_dev = None

    # load the data
    val_set = amt_dataset.AMT_Dataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "val")
    dataset = amt_dataset.AMT_Dataset(os.getcwd() + "/" + opts.data_dir, "labels.csv", "test")

    if opts.arch == 'baseline':
        import baseline
        net = baseline.LanguageModeler(rnn_size=opts.rnn_size, rnn_layers=1)

    elif opts.arch == 'cnn':
        import amt_cnn_8_0_1 as cnn
        net = cnn.AMT_CNN(use_cuda=opts.use_cuda, max_w=opts.max_w)

    for param in net.parameters():
        param.requires_grad = False

    print('evaluating model architecture {}'.format(net.name), file=sys.stderr)
    sys.stderr.flush()

    saved_state = torch.load(opts.model_weights, map_location='cpu')
    net.load_state_dict(saved_state)
    if cuda_dev is not None:
        net = net.cuda(cuda_dev)

    for param in net.parameters():
        param.requires_grad = False

    labels = dataset.idx2name.items()
    label2idx = {l : i for (i, l) in labels}

    with torch.no_grad():
        if opts.threshold is None:
            threshold = roc_select(net, val_set, opts.max_w, batch_size=opts.batch_size, cuda_dev=opts.use_cuda, increment=0.05)
        else:
            threshold = opts.threshold

        print('predicting using threshold {}'.format(threshold))
        sys.stdout.flush()

        test(
            net, dataset, opts.max_w,
            padding=opts.max_w//2, threshold=threshold,
            batch_size=opts.batch_size, cuda_dev=opts.use_cuda
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="data/piano-midi")
    parser.add_argument("-w", "--max_w", type=int, default=2048)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-c", "--use_cuda", type=int, default=None)
    parser.add_argument("-m", "--model_weights", default=None)
    parser.add_argument("--rnn_size", type=int, default=1024)
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("-a", "--arch", default="baseline")

    args = parser.parse_args()
    main(args)

