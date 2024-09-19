import numpy as np
from Tuar_plot_histogram import hist_computation_reconstructed

train_session=10
#shuffle_session="Shuffle{}".format(train_session)
epoch=10
#train_session=10
#dataset=np.load("/home/azorzetto/train{}/dataset.npz".format(train_session))
"""train_data = dataset['train_data']
validation_data=dataset["validation_data"]
test_data= dataset['test_data']
ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'T3', 'T4', 'T5', 'T6',
'A1', 'A2', 'Fz', 'Cz', 'Pz', 'T1', 'T2']
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)

print(test_data.max())
print(test_data.min())

print(train_data.max())
print(train_data.min())

print(validation_data.max())
print(validation_data.min())
"""
hist_computation_reconstructed(train_session, bins=150, path_to_model='/home/azorzetto/train{}/model_weights_backup{}/model_epoch{}.pth'.format(train_session, train_session, epoch))
#hist_computation_reconstructed(shuffle_session, bins=150, path_to_model='/home/azorzetto/trainShuffle_jrj2/model_weights_backup_shuffle_jrj2/model_epoch{}.pth'.format(epoch))