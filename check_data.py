import torch

state_dict = torch.load('models/EUR_USD.pt')
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

