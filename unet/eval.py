import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval_net(net, loader, device):
    """Evaluation"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=True) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot += F.cross_entropy(mask_pred, true_masks, ignore_index=3).item()
            pbar.update()
            
    net.train()
    return tot / n_val
