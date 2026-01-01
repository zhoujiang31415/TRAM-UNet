import torch


def evaluation_metrics(pred, target):
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    m = m1 + m2
    tp = (m1 * m2).sum()
    tn = m.numel() - torch.count_nonzero(m)

    accuracy = (tp + tn) / m.numel()
    dc = 2. * tp / (m1.sum() + m2.sum())
    iou = tp / (torch.count_nonzero(m1 + m2))
    sensitive = tp / m2.sum()
    specificity = tn / (m2.numel() - m2.sum())

    return [accuracy, dc, iou, sensitive, specificity]


