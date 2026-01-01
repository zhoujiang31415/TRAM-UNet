import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import evaluation_metrics


def eval_net(net, loader, device, criterion=None):
    """Evaluation with metrics and loss"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    
    tot, acc, dc, iou, sensitive, specificity = 0, 0, 0, 0, 0, 0
    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if criterion is not None:
                tot += criterion(mask_pred, true_masks).item()
            else:
                if net.n_classes > 1:
                    tot += F.cross_entropy(mask_pred, true_masks).item()
                else:
                    tot += F.binary_cross_entropy_with_logits(mask_pred, true_masks).item()

            if net.n_classes > 1:
                # If multi-category metrics are needed, add the calculation logic here.
                pass 
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                met = evaluation_metrics(pred, true_masks)
                acc += met[0].item()
                dc += met[1].item()
                iou += met[2].item()
                sensitive += met[3].item()
                specificity += met[4].item()
                
            pbar.update()

    net.train()

    return [tot / n_val, acc / n_val, dc / n_val, iou / n_val, sensitive / n_val, specificity / n_val]