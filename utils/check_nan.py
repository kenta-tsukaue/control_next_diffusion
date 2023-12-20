import torch

def check_nan(name, data):
    is_nan = torch.isnan(data).any()
    print(f"NaN in {name}: {is_nan}")



def check_nan_list(name, list):
    print(f"NaN in {name}")
    for data in list:
        is_nan = torch.isnan(data).any()
        print(f"{is_nan}")

