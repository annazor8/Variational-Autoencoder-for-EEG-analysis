import torch 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

train_loss=[]
validation_loss=[]
for i in range(80):
    pat_to_dict="/home/azorzetto/train2/model_weights_backup2/log_dict_epoch{}.pth".format(i+1)
    log_dict = torch.load(pat_to_dict)
    train_loss.append(log_dict['train_loss'])
    validation_loss.append(log_dict['validation_loss'])

epochs = range(1, 81)

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', color='red', marker='o')

# Adding titles and labels
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file or display it
plt.savefig('loss_plot.png')  # Save the plot as an image
plt.show()  # Display the plot