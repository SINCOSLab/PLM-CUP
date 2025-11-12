import torch

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    zero_mask = torch.ne(true, 0.0)
    pred = pred[zero_mask]
    true = true[zero_mask]
    
    if true.numel() == 0:
        return torch.tensor(0.0).to(pred.device)
    
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    
    sum_true = torch.sum(torch.abs(true))
    if sum_true == 0:
        return torch.tensor(0.0).to(pred.device)
    
    return torch.sum(torch.abs(pred - true)) / sum_true

def metric(pred, real):
    try:
        mape = MAPE_torch(pred, real, 0).item()
    except RuntimeError:
        mape = 0.0
    
    mae = MAE_torch(pred, real, 0).item()
    wmape = WMAPE_torch(pred, real, 0).item() 
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape