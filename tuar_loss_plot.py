import torch 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mpld3

train_session='Shuffle_jrj2'
last_epoch=200
train_loss=[]
validation_loss=[]
neg_value = -1e6  
j=0
t=0
#for i in range(0, last_epoch, 5):

for i in range(last_epoch):
    #pat_to_dict="/home/azorzetto/train{}/model_weights_backup{}/log_dict_epoch{}.pth".format(train_session, train_session, i+1) 
    pat_to_dict="/home/azorzetto/trainShuffle_jrj2/model_weights_backup_shuffle_jrj2/log_dict_epoch{}.pth".format(i+1) #i+1 se non è shuffleformat(i+5)
    log_dict = torch.load(pat_to_dict)
    if np.isnan(log_dict['train_loss']):
        #train_loss.append(neg_value)
        train_loss.append(log_dict['train_loss'])
        j=j+1
    else:
        train_loss.append(log_dict['train_loss'])
    if np.isnan(log_dict['validation_loss']):
        #validation_loss.append(neg_value)
        validation_loss.append(log_dict['validation_loss'])
        t=t+1
    else:
        validation_loss.append(log_dict['validation_loss'])


print("nan values in training loss {}".format(j))
print("nan values in validation loss {}".format(t))
epochs = range(1, 201)
df_train=pd.DataFrame(train_loss, columns=['Train Loss'])
df_train.to_csv("/home/azorzetto/train_loss.csv".format(train_session), index=False)
df_val=pd.DataFrame(validation_loss, columns=['Validation Loss'])
df_val.to_csv("/home/azorzetto/val_loss.csv".format(train_session), index=False)
# Plot the loss values
plt.figure(figsize=(12, 8))
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
#mpld3.save_html(plt.gcf(), "/home/azorzetto/train{}/loss_plot.html".format(train_session))

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

# Scrivere il contenuto centrato in un file
with open("/home/azorzetto/train{}/loss_plot.html".format(train_session), "w") as f:
    f.write(centered_html_content)