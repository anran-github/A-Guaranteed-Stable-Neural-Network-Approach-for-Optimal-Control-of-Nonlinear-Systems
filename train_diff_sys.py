import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse
import ast


from network import P_Net

# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='DifSYS_NOM_Dataset_0.1.txt', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=0.0001, help='learning rate')
parser.add_argument('--pre_trained', type=str, default='weights_DiffSys/01/new_model_theta01_epoch450_11.774_u45.321.pth',help='input your pretrained weight path if you want')

args = parser.parse_args()

print(args)

# theta = float(args.dataset.split('_')[-1].split('.txt')[0])
theta = '01'
save_path = f'weights_DiffSys/{theta}'
if not os.path.exists(save_path):
    # os.mkdir(save_path.split('/')[0])
    os.mkdir(save_path)

# import dataset and purify outliers    
with open(args.dataset, 'r') as f:
    # file order: x1, x2, u, p1, p2, p3, r
    data = [ast.literal_eval(x) for x in f.readlines()]
    data = torch.tensor(data).squeeze(1)


input_data = torch.stack([data[:,0],data[:,1]],dim=1)
# !!!! label order sould be: p1,p2,p3,u
label_data = torch.stack([data[:,3],data[:,4],data[:,5],data[:,2]],dim=1)

# ==============DATA FILTER===============
# Filter with std and mean: For p1
# std,d_mean = torch.std_mean(label_data[:,1])    
# mask = label_data[:,1]<=(d_mean+std)
# input_data_valid = input_data[mask]
# label_data_valid = label_data[mask]

# ===========STD-MEAN Norm Output Label================
# std,mean = torch.std_mean(label_data_valid,dim=0)
# print('===========STD AND MEAN OF DATASET================')
# print(f'STD: {std},\nMEAN: {mean}')
# print('==================================================')
# label_normalized = (label_data_valid-mean)/(std+1e-10)

# ===========MIN-MAX Norm Output Label================
# d_min,_ = torch.min(label_data_valid,dim=0)
# d_max,_ = torch.max(label_data_valid,dim=0)
# print('===========MIN and MAX OF DATASET================')
# print(f'MIN: {d_min},\n MAX: {d_max}')
# print('==================================================')
# label_normalized = (label_data_valid-d_min)/(d_max-d_min+1e-10)


X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 1024*10
num_epochs = 35000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





model = P_Net(input_size=2,hidden_size1=8, hidden_size2=16,output_size=4).to(device)

if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//100, eta_min=args.lr*0.07)




global lowest_loss 
lowest_loss = np.inf

def test(model, test_loader,epoch):
    test_loss = 0.0
    u_losses = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            u_loss = criterion(outputs[:,3], targets[:,3].to(device))
            # print(outputs)
            # print(targets)
            # print(loss)
            if not 'cuda' in device.type:
                test_loss += loss.item() * inputs.size(0)
                u_losses  += u_loss.item()
            else:
                test_loss += loss.cpu().item() * inputs.size(0)
                u_losses  += u_loss.cpu().item()

            # loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # save model
    # Save the model
    # global lowest_loss
    # if lowest_loss>=test_loss:
    lowest_loss = test_loss
    print('Model Saved!')
    torch.save(model.state_dict(), os.path.join(
        save_path,'new_model_theta{}_epoch{}_{:.3f}_u{:.3f}.pth'.format(
            theta,epoch,lowest_loss,u_losses)))




test(model, test_loader,epoch=0)


losses = []
loss_avg = []
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss1 = 100*criterion(outputs[:,:3], targets[:,:3].to(device))
        loss2 = 10000*criterion(outputs[:,3], targets[:,3].to(device))
        # loss1 = criterion(outputs[:,:3], targets[:,:3].to(device))
        # loss2 = 10*criterion(outputs[:,3], targets[:,3].to(device))
        loss = loss1 + loss2
    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
    # update lr and gradient
    optimizer.step()
    scheduler.step()

    loss_avg.append(loss.item())  # Store the loss value for plotting

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(np.average(loss_avg))

    if epoch >= 100 and epoch%75==0:
        test(model, test_loader,epoch)

# Plot the loss dynamically
plt.clf()  # Clear previous plot
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
# plt.pause(0.05)  # Pause for a short time to update the plot
plt.savefig(os.path.join(save_path,'training_loss_{}.png'.format(num_epochs)))
plt.plot()




