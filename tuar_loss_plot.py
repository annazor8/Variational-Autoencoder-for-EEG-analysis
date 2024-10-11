import torch 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpld3

#--------------------------set parameters---------------------------------------------
#train_session='Shuffle9'
train_session=10
last_epoch=80
train_loss=[]
validation_loss=[]
j=0
t=0

#------------------------- extract the loss---------------------------------------
#for i in range(0, last_epoch, 5):
for i in range(last_epoch):
    path_to_dict="/home/azorzetto/train{}/model_weights_backup{}/log_dict_epoch{}.pth".format(train_session, train_session, i+1) 
    #path_to_dict="/home/azorzetto/trainShuffle9/model_weights_backup_shuffle9/log_dict_epoch{}.pth".format(i+1) #i+1 se non è shuffleformat(i+5)
    log_dict = torch.load(path_to_dict)
    train_loss.append(log_dict['train_loss'])
    validation_loss.append(log_dict['validation_loss'])
    if np.isnan(log_dict['train_loss']):
        #train_loss.append(neg_value)
        j=j+1
    if np.isnan(log_dict['validation_loss']): 
        t=t+1

print(f"nan values in training loss {j}")
print(f"nan values in validation loss {t}")

#-----------------------save the loss as csv file------------------------------------
epochs = range(1, last_epoch +1)
df_train=pd.DataFrame(train_loss, columns=['Train Loss'])
df_train.to_csv("/home/azorzetto/train{}/train_loss.csv".format(train_session), index=False)
df_val=pd.DataFrame(validation_loss, columns=['Validation Loss'])
df_val.to_csv("/home/azorzetto/train{}/val_loss.csv".format(train_session), index=False)

#---------------Plot the loss values as static plot png and dynamic plot html-----------------------------------------
plt.figure(figsize=(20, 16))
plt.plot(epochs, train_loss, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss over Epochs for train {}'.format(train_session))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# ---------------------------------Save the plot to a file or display it------------------------------------------------
plt.savefig('/home/azorzetto/train{}/loss_plot.png'.format(train_session), dpi =300)  # Save the plot as an image

html_content = mpld3.fig_to_html(plt.gcf())

# Aggiungere il CSS per centrare il grafico
centered_html_content = """
<style>
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
    }
</style>
""" + html_content
with open("/home/azorzetto/train{}/loss_plot.html".format(train_session), "w") as f:
    f.write(centered_html_content)