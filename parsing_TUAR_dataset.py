from tuar_training_utils import leave_one_session_out, get_data_TUAR
import dill

    
#directory_path='/home/azorzetto/data1/01_tcp_ar/01_tcp_ar' #dataset in workstation

directory_path='/home/azorzetto/dataset/01_tcp_ar' #dataset in local PC

#directory_path = '/home/lmonni/Documents/01_tcp_ar' #dataset in Lorenzo's PC

#load the datset and create the dictionary 
session_data=get_data_TUAR(directory_path)
print("loaded dataset")
combinations1, test_data1 = leave_one_session_out(session_data)  # NB combinations[0][0] is a list, combinations[0][1] is an array
print("cros validation split performed")
#combinations2,test_data2, train_label2, validation_label2= leave_one_subject_out(session_data, number_of_trials=2)
data_to_save = {
'combinations1': combinations1,
'test_data1': test_data1}

with open('my_session_data.dill', 'wb') as file:
    dill.dump(data_to_save, file)