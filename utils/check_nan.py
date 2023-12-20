import torch

def check_nan(name, data):
    is_nan = torch.isnan(data).any()
    print(f"NaN in {name}: {is_nan}")

