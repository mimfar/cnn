# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
batch_size = 10
input_channels = 3
output_channels = 12
kernel_size = 3
image_size = 50

m = nn.Conv2d(input_channels, output_channels, kernel_size) # 

x = torch.randn(batch_size, input_channels, image_size, image_size)

x1 = m(x)

m1 = nn.MaxPool2d(2, 2)
x2 = m1(x1)
print(f"Input shape: {x.shape}")

print(f"Conv Output shape: {x1.shape}")
print(f" Maxpool Output shape: {x2.shape}")
# %%

from models.base_cnn import BaseCNN

# %%

model = BaseCNN()

# %%

from torchvision import datasets, transforms
   # Load CIFAR-10 dataset
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False,  download=True, transform=transform)
    
# %%
from models.base_cnn import BaseCNN
import torch.nn.functional as F
import torch
import time

device = 'mps'
model = BaseCNN().to(device)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# data, target = next(iter(train_loader))
# data, target = data.to(device), target.to(device)
# print(data.shape,target.shape)
# output = model(data)
# print(output.shape)
# print(output[0])


#%%
for epoch in range(1):
    print(f"Epoch {epoch}")
    for ix, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        start_time = time.time()
        
        if ix % 100 == 0:
            with torch.no_grad():
                loss_val = []
                for data_val, target_val in val_loader:
                    data_val, target_val = data_val.to(device), target_val.to(device)
                    output_val = model(data_val)
                    loss_val.append(F.cross_entropy(output_val, target_val))
                Validation_Loss = sum(loss_val)/len(loss_val)
                print(f"ix: {ix}, Loss: Val. |{Validation_Loss:1.3f},Train. {loss:1.3f}, Time: {time.time() - start_time:1.3f}  ")




#%%
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('./training/history_lr_early_stopping.csv',index_col=0)
df.head(10)

plt.figure(figsize=(10, 5))
plt.plot(df[['train_accuracies','val_accuracies']], 
         label=['Training Accuracy','Validation Accuracy'])
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy %')
plt.title('Accuracy')
plt.ylim(0,100)
plt.xlim(0,df.shape[0]-1)
# plt.plot(df['val_accuracies'], label='Validation Accuracy')
plt.legend()
plt.show()



#%%
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_cnn import BaseCNN



def load_model(model, filepath='best_model.pth'):
    """
    Load a saved model checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file {filepath} not found!")
        return None
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(filepath, map_location='mps')
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"Learning rate: {checkpoint['learning_rate']:.6f}")
    
    return checkpoint


    # Initialize model
model = BaseCNN(num_classes=10, input_channels=3)
model = model.to('cpu')

checkpoint = load_model(model, 'best_model.pth')
# %%
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# %%
