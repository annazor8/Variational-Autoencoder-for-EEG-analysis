import torch 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_session=2
last_epoch=80
train_loss=[]
validation_loss=[]
neg_value = -1e6  
j=0
t=0
#for i in range(0, last_epoch, 5):

for i in range(last_epoch):
    pat_to_dict="/home/azorzetto/train{}/model_weights_backup{}/log_dict_epoch{}.pth".format(train_session, train_session, i+1) 
    #pat_to_dict="/home/azorzetto/trainShuffle/model_weights_backup_shuffle/log_dict_epoch{}.pth".format(i+5) #i+1 se non è shuffle
    log_dict = torch.load(pat_to_dict)
    if np.isnan(log_dict['train_loss']):
        train_loss.append(neg_value)
        j=j+1
    else:
        train_loss.append(log_dict['train_loss'])
    if np.isnan(log_dict['validation_loss']):
        validation_loss.append(neg_value)
        t=t+1
    else:
        validation_loss.append(log_dict['validation_loss'])

print("nan values in training loss {}".format(j))
print("nan values in validation loss {}".format(t))
epochs = range(1, 17)
df_train=pd.DataFrame(train_loss, columns=['Train Loss'])
df_train.to_csv("/home/azorzetto/train_loss.csv".format(train_session), index=False)
df_val=pd.DataFrame(validation_loss, columns=['Validation Loss'])
df_val.to_csv("/home/azorzetto/val_loss.csv".format(train_session), index=False)
# Plot the loss values
plt.figure()
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
#plt.xlim(0, 15)
#plt.ylim(top=0.05*np.max(validation_loss))
# Adding titles and labels
plt.title('Training and Validation Loss over Epochs for train {}'.format(train_session))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file or display it
plt.savefig('/home/azorzetto/train{}/loss_plot.png'.format(train_session))  # Save the plot as an image