import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
import skimage.morphology
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.morphology import square, disk 
from sklearn.preprocessing import StandardScaler
#from tensordash.tensordash import Tensordash, Customdash

from Tools import *
from Models_FC114 import *
from Amazonia_Legal_RO import AMAZON_RO
from Amazonia_Legal_PA import AMAZON_PA
from Cerrado_Biome_MA import CERRADO_MA
parser = argparse.ArgumentParser(description='')
#Defining the meta-paramerts
# Model
parser.add_argument('--method_type', dest='method_type', type=str, default='Unet', help='method that will be used, could be used also (siamese_network)')
# Training parameters
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='number images in batch')
# Optimizer hyperparameters
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
# Image_processing hyperparameters
parser.add_argument('--data_augmentation', dest='data_augmentation', type=eval, choices=[True, False], default=True, help='if data argumentation is applied to the data')

# TODO LUCAS:Em quantas colunas ou linhas eu irei dividir minha imagem para gerar os quadradinhos (patches)
parser.add_argument('--source_vertical_blocks', dest='source_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--source_horizontal_blocks', dest='source_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')
parser.add_argument('--target_vertical_blocks', dest='target_vertical_blocks', type=int, default=10, help='number of blocks which will divide the image vertically')
parser.add_argument('--target_horizontal_blocks', dest='target_horizontal_blocks', type=int, default=10, help='number of blocks which will divide the image horizontally')

parser.add_argument('--fixed_tiles', dest='fixed_tiles', type=eval, choices=[True, False], default=True, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--defined_before', dest='defined_before', type=eval, choices=[True, False], default=False, help='decide if tiles will be choosen randomly or not')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=7, help='number of image channels')
parser.add_argument('--patches_dimension', dest='patches_dimension', type=int, default=128, help= 'dimension of the extracted patches')

# 
parser.add_argument('--overlap_s', dest='overlap_s', type=float, default= 0.75, help= 'stride cadence')
parser.add_argument('--overlap_t', dest='overlap_t', type=float, default= 0.75, help= 'stride cadence')

# compute ndvi refere-se a um indice. Era algum tipo de stack de bandas. compute_ndvi = False. Pode ignorar e manter assim.
parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=True, help='Cumpute and stack the ndvi index to the rest of bands')
parser.add_argument('--balanced_tr', dest='balanced_tr', type=eval, choices=[True, False], default=True, help='Decide wether a balanced training will be performed')
#parser.add_argument('--balanced_vl', dest='balanced_vl', type=eval, choices=[True, False], default=True, help='Decide wether a balanced training will be performed')

# TODO LUCAS:Parâmetro buffer para quando for converter de imagem vetorial para pixel
parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
parser.add_argument('--source_buffer_dimension_out', dest='source_buffer_dimension_out', type=int, default=2, help='Dimension of the buffer outside of the area')
parser.add_argument('--source_buffer_dimension_in', dest='source_buffer_dimension_in', type=int, default=0, help='Dimension of the buffer inside of the area')
parser.add_argument('--target_buffer_dimension_out', dest='target_buffer_dimension_out', type=int, default=2, help='Dimension of the buffer outside of the area')
parser.add_argument('--target_buffer_dimension_in', dest='target_buffer_dimension_in', type=int, default=0, help='Dimension of the buffer inside of the area')

parser.add_argument('--porcent_of_last_reference_in_actual_reference', dest='porcent_of_last_reference_in_actual_reference', type=int, default=100, help='Porcent of number of pixels of last reference in the actual reference')
parser.add_argument('--porcent_of_positive_pixels_in_actual_reference_s', dest='porcent_of_positive_pixels_in_actual_reference_s', type=int, default=10, help='Porcent of number of pixels of last reference in the actual reference in source domain')
parser.add_argument('--porcent_of_positive_pixels_in_actual_reference_t', dest='porcent_of_positive_pixels_in_actual_reference_t', type=int, default=10, help='Porcent of number of pixels of last reference in the actual reference in target domain')
parser.add_argument('--num_classes', dest='num_classes', type=int, default=3, help='Number of classes comprised in both domains')
# Phase
parser.add_argument('--phase', dest='phase', default='train', help='train, test, generate_image, create_dataset')
parser.add_argument('--training_type', dest='training_type', type=str, default='classification', help='classification|domain_adaptation|domain_adaptation_check')

# TODO LUCAS:Geralmente rodamos 10x
parser.add_argument('--runs', dest='runs', type=int, default=1, help='number of executions of the algorithm')
#parser.add_argument('--scatter_plot', dest='scatter_plot', type=eval, choices=[True, False], default=True, help='Decide if a scatter plot is done during the training')
#parser.add_argument('--change_every_epoch', dest='change_every_epoch', type=eval, choices=[True, False], default=False, help='Decide if the target set will be change every epoch in order to balance the training')
# Early stop parameter
parser.add_argument('--patience', dest='patience', type=int, default=10, help='number of epochs without improvement to apply early stop')
#Checkpoint dir
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='', help='Domain adaptation checkpoints')
# Images dir and names
parser.add_argument('--source_dataset', dest='source_dataset', type=str, default='Amazon',help='The name of the dataset used')
parser.add_argument('--target_dataset', dest='target_dataset', type=str, default='Cerrado',help='The name of the dataset used')
parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
parser.add_argument('--data_type', dest='data_type', type=str, default='.npy', help= 'Type of the input images and references')

parser.add_argument('--source_data_t1_year', dest='source_data_t1_year', type=str, default='', help='Year of the time 1 image')
parser.add_argument('--source_data_t2_year', dest='source_data_t2_year', type=str, default='', help='Year of the time 2 image')
parser.add_argument('--target_data_t1_year', dest='target_data_t1_year', type=str, default='', help='Year of the time 3 image')
parser.add_argument('--target_data_t2_year', dest='target_data_t2_year', type=str, default='', help='Year of the time 4 image')
parser.add_argument('--source_data_t1_name', dest='source_data_t1_name', type=str, default='', help='image 1 name')
parser.add_argument('--source_data_t2_name', dest='source_data_t2_name', type=str, default='', help='image 2 name')
parser.add_argument('--target_data_t1_name', dest='target_data_t1_name', type=str, default='', help='image 3 name')
parser.add_argument('--target_data_t2_name', dest='target_data_t2_name', type=str, default='', help='image 4 name')
parser.add_argument('--source_reference_t1_name', dest='source_reference_t1_name', type=str, default='', help='reference 1 name')
parser.add_argument('--source_reference_t2_name', dest='source_reference_t2_name', type=str, default='', help='reference 2 name')
parser.add_argument('--target_reference_t1_name', dest='target_reference_t1_name', type=str, default='', help='reference 1 name')
parser.add_argument('--target_reference_t2_name', dest='target_reference_t2_name', type=str, default='', help='reference 2 name')
#Dataset Main paths
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='/media/lvc/Dados/PEDROWORK/Trabajo_Domain_Adaptation/Dataset/', help='Dataset main path')
# #Architecture configuration
# parser.add_argument('--FE_Architecture', dest='FE_Architecture', type=str, default='', help='Decide the architecture of the Feature Extractor(FE)')
# parser.add_argument('--CL_Architecture', dest='CL_Architecture', type=str, default='', help='Decide the architecture of the Classifier(Cl)')
# parser.add_argument('--DR_Architecture', dest='DR_Architecture', type=str, default='', help='Decide the architecture of the Domain Regressor(DR)')
# parser.add_argument('--FE_flatten', dest='FE_flatten', type=eval, choices=[True, False], default=True, help='Decide wether a flaten is applied at the end of the choosen predefined architecture of FE')
args = parser.parse_args()


# TODO Lucas: P
def main():
    print(args)
    #histories = Customdash(ModelName = 'SLVC06_train_process', email = 'pedrosoto423@gmail.com', password = 'Bad87*be@tles63')
    
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')
    
    args.checkpoint_dir = './checkpoints/' + args.checkpoint_dir 
    
    if args.source_dataset == 'Amazon_RO':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = AMAZON_RO(args)
        
    if args.source_dataset == 'Amazon_PA':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = AMAZON_PA(args)
        
    if args.source_dataset == 'Cerrado_MA':
        args.dataset = 'Cerrado_Biome/'
        args.buffer_dimension_in = args.source_buffer_dimension_in
        args.buffer_dimension_out = args.source_buffer_dimension_out
        args.data_t1_name = args.source_data_t1_name
        args.data_t2_name = args.source_data_t2_name
        args.reference_t1_name = args.source_reference_t1_name
        args.reference_t2_name = args.source_reference_t2_name
        dataset_s = CERRADO_MA(args)
                            
    if args.target_dataset == 'Amazon_RO':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = AMAZON_RO(args)
        
    if args.target_dataset == 'Amazon_PA':
        args.dataset = 'Amazonia_Legal/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = AMAZON_PA(args)
        
    if args.target_dataset == 'Cerrado_MA':
        args.dataset = 'Cerrado_Biome/'
        args.buffer_dimension_in = args.target_buffer_dimension_in
        args.buffer_dimension_out = args.target_buffer_dimension_out
        args.data_t1_name = args.target_data_t1_name
        args.data_t2_name = args.target_data_t2_name
        args.reference_t1_name = args.target_reference_t1_name
        args.reference_t2_name = args.target_reference_t2_name
        dataset_t = CERRADO_MA(args)
        
        
       
    print(np.shape(dataset_s.images_norm))
    print(np.shape(dataset_t.images_norm))
    #print(np.shape(dataset_t.images_norm))          
    for i in range(args.runs):
        dataset = []
        print(i)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        print(dt_string)
        if args.training_type == 'classification':
            args.save_checkpoint_path = args.checkpoint_dir + '/' + args.method_type + '_' + dt_string + '/'
        if args.training_type == 'domain_adaptation' or args.training_type == 'domain_adaptation_balance':
            args.save_checkpoint_path = args.checkpoint_dir + '/' + 'Tr_M_' + dt_string + '/'
        if not os.path.exists(args.save_checkpoint_path):
            os.makedirs(args.save_checkpoint_path)
            #Writing the args into a file
        with open(args.save_checkpoint_path + 'commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        args.vertical_blocks = args.source_vertical_blocks
        args.horizontal_blocks = args.source_horizontal_blocks
        args.overlap = args.overlap_s
        args.porcent_of_positive_pixels_in_actual_reference = args.porcent_of_positive_pixels_in_actual_reference_s
        dataset_s.Tiles_Configuration(args, i)
        dataset_s.Coordinates_Creator(args, i)
        
        args.vertical_blocks = args.target_vertical_blocks
        args.horizontal_blocks = args.target_horizontal_blocks
        args.overlap = args.overlap_t
        args.porcent_of_positive_pixels_in_actual_reference = args.porcent_of_positive_pixels_in_actual_reference_t
        dataset_t.Tiles_Configuration(args, i)
        dataset_t.Coordinates_Creator(args, i)
        
        dataset.append(dataset_s)
        dataset.append(dataset_t) 
        
        print('[*]Initializing the model...')
        model = Models(args, dataset)
               
        model.Train()
        #histories.sendLoss(loss = model.best_val_acc, epoch = i, total_epochs = args.runs)
        
if __name__=='__main__':
    main()
