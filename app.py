# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:17:15 2023

@author: Allwin Noble
"""

from flask import Flask,render_template,request
from Grad_Cam_Helper import read_frames_gcam,select_frames_gcam,make_gradcam_heatmap,get_resized_heatmap,get_bounding_boxes_m,get_bbox_patches,create_animation
import os 
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from keras.models import load_model
from tensorflow.keras.layers import Conv3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from flask import send_from_directory
import argparse

app=Flask(__name__,static_folder='uploads')

app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 150 MB Maximum model size that can be uploaded

#initializing global variables
input_video_url=''
input_filename=''
output_video_url=''
result=''
model_path=''
gcam_input=''
conv3d_layers=[]
animation_paths={}
heatmap_images_paths={}
inference_heatmap_paths={}
parsed_args=None


@app.route('/<path:path>')
def send_css(path):
    return send_from_directory('templates',path)


@app.route('/', methods=['GET', 'POST'])
def index():
    #conv3d_layers=analyze_predictions()
    #return render_template('result.html',layer_names=conv3d_layers)
   
    if (request.form.get('without_model') =='Analyze new video'):
        print('only video true')
        return render_template('index.html',model_loaded=False,only_video=True)
    
          
    return render_template('index.html',only_video=False)

@app.route('/load_video',methods=['POST'])
def upload_video():
    if request.method=='POST':
        
        
        uploaded_video=request.files['video']
        if uploaded_video:
            global input_video_url,input_filename
            input_filename=uploaded_video.filename
            video_path=os.path.join('uploads',input_filename)
            uploaded_video.save(video_path)
            #Convert the saved video to mp4 using the moviepy library
            input_filename=convert_to_mp4(input_filename,video_path,'./uploads/')
            input_video_url=f'./uploads/{input_filename}'
            #print(video_url)
    return render_template('index.html',model_loaded=True,only_video=True,input_video_url=input_video_url,input_filename=input_filename)

@app.route('/heatMaps/<path:path>')
def send_heatMaps(path):
    return send_from_directory('heatMaps',path)

@app.route('/inferenceHeatmaps/<path:path>')
def send_inferenceHeatMaps(path):
    return send_from_directory('inferenceHeatmaps',path)

@app.route('/process',methods=['POST'])
def process():
        global output_video_url,result
        result,output_video_url=process_video(input_video_url)
        return render_template('index.html', model_loaded=True,only_video=True,result=result,input_video_url=input_video_url,input_filename=input_filename,output_video_url=output_video_url)

@app.route('/load_model',methods=['POST'])
def save_model_to_path():
    global model_path
    if not os.path.exists('./uploads/model'):
        os.makedirs('./uploads/model')
        print('Model folder created')
    uploaded_model=request.files['upload_model']
    model_path=os.path.join('./uploads/model/',uploaded_model.filename)
    if uploaded_model:
        if os.path.exists(model_path):
            print('The same model already exists in the models folder')
        else:
            uploaded_model.save(model_path)
            print('Model saved sucessfully to',model_path)
    return render_template('index.html',model_loaded=True,only_video=True)
    

@app.route('/make_predictions',methods=['POST'])
def make_predictions():
    class_map=read_class_map('classMap.txt')
    model=load_model(model_path)
    predicted=model.predict(gcam_input)
    predicted=np.argmax(predicted,axis=1)
    if(class_map==False):
        predicted_label=predicted[0]
    else:
        predicted_label=class_map[predicted[0]]
    actual_label=False
    if (parsed_args.pred_index!=None):
        if(parsed_args.pred_index!=predicted[0]):
            if(class_map==False):
                actual_label=parsed_args.pred_index
            else:
                actual_label=class_map[parsed_args.pred_index]
    
    return render_template('index.html',model_loaded=False,only_video=True,result=result,input_video_url=input_video_url,input_filename=input_filename,output_video_url=output_video_url,predicted_label=predicted_label,actual_label=actual_label)

@app.route('/get_heatmaps',methods=['POST'])
def get_heatmaps():
    global animation_paths,heatmap_images_paths,gcam_input,parsed_args,conv3d_layers,inference_heatmap_paths
    model=load_model(model_path)
    conv3d_layers=[layer.name for layer in model.layers if isinstance(layer, Conv3D)]
    class_map=read_class_map('classMap.txt')
    
    
    #video_data=output_video_url
    
    #model.summary()
    animation_paths={}
    heatmap_images_paths={}
    inference_heatmap_paths={}    
    predicted  = model.predict(gcam_input)
    predicted  = np.argmax(predicted,axis=1)
    if(class_map==False):
        print('predicted class is',predicted[0])
    else:
        print('predicted class is',class_map[predicted[0]])
    pred_index=predicted[0]
    print('pred index set as',pred_index)    
    
    
    #pred_index=0
    print(gcam_input.shape)
    #volume_size = gcam_input.shape  #(1, 17, 120, 160, 3)
   # print(request.form.get('layer_name'))
    #last_conv_layer_name='conv3d_11'
    animation_paths['Original Video']=input_video_url
    for layer in conv3d_layers:
        
        folder_name=input_filename.rsplit('.',1)[0]
        last_conv_layer_name = layer
        heatmap_path=f'./heatMaps/{folder_name}/{last_conv_layer_name}/'
        heatmap_image_paths_single_layer=[]
        
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path) 
            print('Folder created for heatmaps')
        else:
            print('Folder already exists')
            animation_paths[layer]=f'./heatMaps/{folder_name}/{last_conv_layer_name}/{last_conv_layer_name}_heatmap_video.mp4'
            print(f'Skipped for {last_conv_layer_name}')
            list_image_paths=os.listdir(heatmap_path)
            for i in list_image_paths[0:gcam_input.shape[1]]:
                heatmap_image_paths_single_layer.append(heatmap_path+i)
            heatmap_images_paths[last_conv_layer_name]=heatmap_image_paths_single_layer
            
            continue
        # Remove last layer's activation
        model.layers[-1].activation = None
        predicted_heatmap = make_gradcam_heatmap(gcam_input, model, last_conv_layer_name,pred_index=pred_index)
        
        #Heatmap with models predicted index
        predicted_resized_heatmap = get_resized_heatmap(predicted_heatmap, (gcam_input.shape[1],gcam_input.shape[2],gcam_input.shape[3]))######needs to be made dynamic he shape of upscaling
        
        #heatmap with users passed pred index
        actual_resized_heatmaps=0
        ###For infernece creating heatmaps with the actual pred index passed by the user as argument
        
        ##############################
      
        
        
        if(parsed_args.pred_index!=None):
            if(parsed_args.pred_index!=pred_index):
                actual_heatmap = make_gradcam_heatmap(gcam_input, model, last_conv_layer_name,pred_index=parsed_args.pred_index)
                actual_resized_heatmaps = get_resized_heatmap(actual_heatmap, (gcam_input.shape[1],gcam_input.shape[2],gcam_input.shape[3]))
                inference_heatmap_list=[]
                inference_heatmap_path=f'./inferenceHeatmaps/{folder_name}/{layer}/'
                if not os.path.exists(inference_heatmap_path):
                    os.makedirs(inference_heatmap_path)
                #generate 3 different inferenece images for start,middle and end frame at each layer
                for i in np.linspace(0,gcam_input.shape[1]-1,3):
                    
                    i=int(i)
                    actual_img_heatmap=actual_resized_heatmaps[i]
                    predicted_img_heatmap=predicted_resized_heatmap[i]
                    difference=(predicted_img_heatmap-actual_img_heatmap) #check dominant regions in actual_heatmap image
                    fig,ax = plt.subplots(figsize=(3,4))
                    plt.imshow(np.squeeze(gcam_input[0][i]),cmap='gray')
                    plt.imshow(difference,cmap='jet',alpha=0.3)
                    ax.axis('off')
                    plt.savefig(inference_heatmap_path+f'inference_{layer}_{i}.png',bbox_inches='tight')
                    plt.close()
                    
                    
                    inference_heatmap_list.append(inference_heatmap_path+f'inference_{layer}_{i}.png')
                inference_heatmap_paths[layer]=inference_heatmap_list     
            else:
                print('Predicted output and actual output are same ')
        #Image generation
        #-------------------
             
        for i in range (gcam_input.shape[1]):
            
            
            fig,ax = plt.subplots(figsize=(3,2))
        
            img0 = imshow(np.squeeze(gcam_input[0, i]), cmap='gray')
            img1 = imshow(np.squeeze(predicted_resized_heatmap[i]),
                              cmap='jet', alpha=0.3)
        
            if(parsed_args.bbox_threshold!=None):
                bboxes = get_bounding_boxes_m(np.squeeze(predicted_resized_heatmap[i]),threshold=parsed_args.bbox_threshold,otsu=False)
            else:
                bboxes = get_bounding_boxes_m(np.squeeze(predicted_resized_heatmap[i]),threshold=.60,otsu=True)
            patches = get_bbox_patches(bboxes)
        
            for patch in patches:
                img1.axes.add_patch(patch)
            
            ax.axis('off')
            
            plt.savefig(heatmap_path+f'{last_conv_layer_name}_{i}.png',bbox_inches='tight')
            plt.close()
            heatmap_image_paths_single_layer.append(heatmap_path+f'{last_conv_layer_name}_{i}.png')
        
        heatmap_images_paths[last_conv_layer_name]=heatmap_image_paths_single_layer
        for i in np.linspace(0,gcam_input.shape[1]-1,6):
            i=int(i)
            original_image_path=f'./inferenceHeatmaps/{folder_name}/original/'
            if not os.path.exists(original_image_path):
                os.makedirs(original_image_path)
            fig,ax = plt.subplots(figsize=(3,4))
            plt.imshow(np.squeeze(gcam_input[0][i]))
            ax.axis('off')
            plt.savefig(original_image_path+f'original_{i}.png',bbox_inches='tight')
            plt.close()
        #Video Generation
        #-------------------
        animation_file_path=create_animation(gcam_input[0],f'{last_conv_layer_name}_heatmap_video',saveat=heatmap_path,heatmap=predicted_resized_heatmap)
        animation_paths[layer]=animation_file_path
    
    
    return render_template('result.html',layer_names=conv3d_layers,image_paths=True,animation_paths=True,inference_heatmap_paths=True)

@app.route('/get_images',methods=['POST'])
def send_image_path():
    layer_name=request.form.get('layer_name')
    individual_layer_image_list=heatmap_images_paths[layer_name]
    return render_template('result.html',layer_names=conv3d_layers,image_paths=True,individual_layer_image_list=individual_layer_image_list,animation_paths=True,inference_heatmap_paths=True)

@app.route('/get_videos',methods=['POST'])
def send_video_path():
    individual_video_paths=animation_paths
    return render_template('result.html',layer_names=conv3d_layers,image_paths=True,animation_paths=True,individual_video_paths=individual_video_paths,inference_heatmap_paths=True)

@app.route('/get_inference',methods=['POST'])
def send_inference_path():
    individual_inference_paths=inference_heatmap_paths
    for name,path in individual_inference_paths.items():
        print(path)
    return render_template('result.html',layer_names=conv3d_layers,image_paths=True,animation_paths=True,inference_heatmap_paths=True,individual_inference_paths=individual_inference_paths)
    
def convert_to_mp4(original_filename,video_path,output_path):
    
    output_filename=original_filename.rsplit('.',1)[0]+'.mp4'
    output_path=os.path.join(output_path,output_filename)
    video=VideoFileClip(video_path)
    video.write_videofile(output_path,codec='libx264') #####codec libx264 important for displaying content using video tag
    return output_filename

def process_video(video_data):
    global gcam_input,model_path
    model=load_model(model_path)
    #model.summary()
    model_input_shape=model.input_shape
    gcam_input=read_frames_gcam(video_data,model_input_shape,each_nth=2)
    gcam_input=select_frames_gcam(gcam_input, model_input_shape[1])
    output=create_video_from_np_array(gcam_input)
    convert_to_mp4(f'preprocessed_{input_filename}.mp4',output,'./uploads/')
    result=gcam_input.shape
    return result,output

def create_video_from_np_array(gcam_input):
        width,height=gcam_input.shape[3],gcam_input.shape[2]
        #no_frames=24
        output_filename=f'./uploads/preprocessed_{input_filename}.mp4'
        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        fps=3
        out=cv2.VideoWriter(output_filename,fourcc,fps,(width,height))
             
        for frame in gcam_input[0]:
            frame=np.uint8(frame*255) #normalize input back in range of 0-255 earlier it was 0-1
            out.write(frame)
        out.release()
            
        return output_filename

def read_class_map(filename):
    try:
        print('in read_class_map function')
        class_map_filename=f'./classmap/{filename}'
        class_map_dict={}

        with open(class_map_filename,'r') as file:
            lines=file.readlines()
            for line in lines:
                parts=line.strip().split(':')
                if len(parts)==2:
                    numeric_index=int(parts[0].strip())
                    class_label=parts[1].strip()
                    class_map_dict[numeric_index]=class_label
        print("Loaded class map:")
        return class_map_dict
    except FileNotFoundError:
       return False
        #print(f"Error: File '{filename}' not found in classmap folder.Please place the file in the classmap foilder to get predictions")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="GradCam App")
    parser.add_argument('--pred_index',type=int,required=False,help='Prediction index value to run GradCAm app with')
    parser.add_argument('--bbox_threshold',type=float,required=False,help='User Defined Bounding Box threshold, ** Otsu threshold disabled')
    args=parser.parse_args()
    parsed_args=args
    app.run(debug=False)