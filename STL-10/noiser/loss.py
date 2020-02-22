import torch

def entropy_balance_loss(mean_preds, balance_coeff=1):
    H = - (mean_preds * torch.log(mean_preds)).sum(dim=1).mean(dim=0)

    batch_mean_preds = mean_preds.mean(dim=0)
    H_batch = - (batch_mean_preds * torch.log(batch_mean_preds)).sum()

    loss = H - balance_coeff * H_batch

    return loss