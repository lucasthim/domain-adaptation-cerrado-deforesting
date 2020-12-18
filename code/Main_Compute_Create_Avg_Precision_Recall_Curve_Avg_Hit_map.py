import os
import sys
import numpy as np 

import matplotlib.pyplot as plt


num_samples = 20
results_folders = []

results_folders.append('./results_avg/teste_5_1/')
results_folders.append('./results_avg/teste_5_2/')
results_folders.append('./results_avg/teste_5_3/')
#results_folders.append('./results_avg/Unet_50ep_15patience_2runs_Tr_Amazonia_RO_Ts_Amazonia_RO/')
#results_folders.append('./results_avg/Unet_50ep_15patience_2runs_Tr_Cerrado_MA_Ts_Amazonia_RO/')

labels = []
labels.append('1-Tr:T,Ts:T,')
labels.append('2-Tr:S,Ts:T, ')
labels.append('3-DA')
#labels.append('3-Tr:S,Ts:T->S,CycleGAN, ')
#labels.append('4-Tr:S,Ts:T->S,CycleGAN Diff Norm, ')

colors = []
colors.append('#4169E1')
colors.append('#00BFFF')
colors.append('#FF0000')
#colors.append('#660033')

if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.subplot(111)
    Npoints = num_samples
    Interpolation = True
    Correct = True
    for rf in range(len(results_folders)):
                
        recall = np.zeros((1 , num_samples))
        precision = np.zeros((1 , num_samples))
        
        MAP = 0
        
        recall_i = np.zeros((1,num_samples))
        precision_i = np.zeros((1,num_samples))
        
        AP_i = []
        AP_i_ = 0
        folder_i = os.listdir(results_folders[rf])
        
        for i in range(len(folder_i)):
            result_folder_name = folder_i[i]
            if result_folder_name != 'Results.txt':
                #print(folder_i[i])
                recall_path = results_folders[rf] + folder_i[i] + '/Recall.npy'
                precision_path = results_folders[rf] + folder_i[i] + '/Precission.npy'
                
                recall__ = np.load(recall_path)
                precision__ = np.load(precision_path)
                
                #print(precision__)
                #print(recall__)
                
                if np.size(recall__, 1) > Npoints:
                    recall__ = recall__[:,:-1]
                if np.size(precision__, 1) > Npoints:
                    precision__ = precision__[:,:-1]
                
                recall__ = recall__/100
                precision__ = precision__/100
                
                if Correct:
                         
                    if precision__[0 , 0] == 0:
                        precision__[0 , 0] = 2 * precision__[0 , 1] - precision__[0 , 2]                
                    
                    if Interpolation:
                        precision = precision__[0,:]
                        precision__[0,:] = np.maximum.accumulate(precision[::-1])[::-1]
                    
                    
                    if recall__[0 , 0] > 0:
                        recall = np.zeros((1,num_samples + 1))
                        precision = np.zeros((1,num_samples + 1))
                        # Replicating precision value
                        precision[0 , 0] = precision__[0 , 0]
                        precision[0 , 1:] = precision__
                        precision__ = precision
                        # Attending recall
                        recall[0 , 1:] = recall__
                        recall__ = recall
                
                                
                p = precision__[:,:-1]
                dr = np.diff(recall__)
                
                recall_i = recall__
                precision_i = precision__
                
                mAP = 100 * np.matmul(p , np.transpose(dr))
               
        ax.plot(recall_i[0,:], precision_i[0,:], color=colors[rf], label=labels[rf] + 'mAP:' + str(np.round(mAP[0,0],1)))
        
    ax.legend()
    #plt.show()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.title('S: AMAZON RO, T: CERRADO MA')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('teste_5.png')