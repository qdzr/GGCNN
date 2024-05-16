import torch
print(torch.__version__)
model = torch.load('ckpt/epoch_0213_acc_0.6374.pth', map_location='cpu')
print(model)
torch.save(model, 'epoch_0213_acc_0.6374.pth', _use_new_zipfile_serialization=False)
