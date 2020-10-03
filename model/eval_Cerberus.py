import torch
import torch.nn.functional as F
#from tqdm import tqdm

def eval_net(net, loader, class_weights, device, combine=False):
    # Generates a confusion matrix and calculates the loss values (global + individual for every U-Net)
    net.eval()
    n_val = len(loader)  # the number of batch


    confusion_matrix = torch.zeros((3,3,3), dtype=torch.int).to(device=device)
    loss = torch.zeros(4, dtype=torch.float32).to(device=device)
    for batch in loader: # Batch size has to be one
        print(".", end = '')
        imgs_u, true_masks_u = batch['image_u'], batch['mask_u']
        imgs_v, true_masks_v = batch['image_v'], batch['mask_v']
        imgs_w, true_masks_w = batch['image_w'], batch['mask_w']

        imgs_u = imgs_u.to(device=device, dtype=torch.float32).to(device=device)
        imgs_v = imgs_v.to(device=device, dtype=torch.float32).to(device=device)
        imgs_w = imgs_w.to(device=device, dtype=torch.float32).to(device=device)
        true_masks_u = true_masks_u.to(device=device, dtype=torch.long)
        true_masks_v = true_masks_v.to(device=device, dtype=torch.long)
        true_masks_w = true_masks_w.to(device=device, dtype=torch.long)

        with torch.no_grad():
            masks_pred_u, masks_pred_v, masks_pred_w = net(imgs_u, imgs_v, imgs_w)

        true_masks = torch.cat((true_masks_u[None,:,:,:], true_masks_v[None,:,:,:], true_masks_w[None,:,:,:]), dim=0).to(device=device);
        masks_pred = torch.cat((masks_pred_u[None,:,:,:,:], masks_pred_v[None,:,:,:,:], masks_pred_w[None,:,:,:,:]), dim=0).to(device=device);


        class_weights = torch.zeros(3, dtype=torch.float32).to(device=device)
        for s in range(3):
            num = (true_masks[s,:,:]!=3).sum()
            class_weights[0] = torch.div(num, (true_masks[s,:,:,:]==0).sum()*1.0)
            class_weights[1] = torch.div(num, (true_masks[s,:,:,:]==1).sum()*1.0)
            class_weights[2] = torch.div(num, (true_masks[s,:,:,:]==2).sum()*1.0)
            loss[s] += F.nll_loss(masks_pred[s,:,:,:,:], true_masks[s,:,:,:], class_weights, ignore_index=3)
            
        true_masks2 = torch.cat((true_masks_u, true_masks_v, true_masks_w), dim=0).to(device=device);
        masks_pred2 = torch.cat((masks_pred_u, masks_pred_v, masks_pred_w), dim=0).to(device=device);
        num = (true_masks2!=3).sum()
        class_weights[0] = torch.div(num, (true_masks2==0).sum()*1.0)
        class_weights[1] = torch.div(num, (true_masks2==1).sum()*1.0)
        class_weights[2] = torch.div(num, (true_masks2==2).sum()*1.0)
        loss[3] += F.nll_loss(masks_pred2, true_masks2, class_weights, ignore_index=3)

        best_pred_mask_u = torch.argmax(masks_pred_u, axis=1)
        best_pred_mask_v = torch.argmax(masks_pred_v, axis=1)
        best_pred_mask_w = torch.argmax(masks_pred_w, axis=1)
        best_pred_mask = torch.cat((best_pred_mask_u[None,:,:,:], best_pred_mask_v[None,:,:,:], best_pred_mask_w[None,:,:,:]), dim=0)

        for s in range(3):
            for t in range(3):
                for v in range(3):
                    confusion_matrix[s, t, v] += torch.sum(torch.logical_and(best_pred_mask[s,:,:,:]==t, true_masks[s,:,:,:]==v))

    print('')
    net.train()
    return confusion_matrix, loss/n_val
