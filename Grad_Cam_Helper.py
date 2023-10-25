# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:10:21 2023

@author: Allwin Noble
"""

import cv2
from skimage.transform import resize
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from scipy.ndimage import zoom
from matplotlib.patches import Rectangle
from matplotlib import animation
import matplotlib.pyplot as plt


#Helper functions for video preprocessing like resizing ,center croping and specific number of frame selection
#----------------------------------------------------------------
def read_frames_gcam(filename,model_input_shape,each_nth=2):
    videos=[]
    vcap=cv2.VideoCapture(filename)
    sucess=True

    frames=[]
    cnt=0
    while sucess:
        try:
            sucess,image=vcap.read()
            cnt+=1
            if cnt%each_nth==0:
                #image=crop_center_square(image)
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=resize(image,(model_input_shape[2],model_input_shape[3]))
                frames.append(image)
        except Exception as e:
            print(e)
    videos.append(frames)
    
    return videos

def crop_center_square(frame):
  y, x = frame.shape[0:2] #used to extract the dimensions (height and width) of a frame
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2) #// integer division
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def select_frames_gcam(frames_arr,n):
    videos_n_frames=[]


    frames=[]
    for i in range(len(frames_arr)):
        frames=[]
        for t in np.linspace(0,len(frames_arr[i])-1,num=n):
            frames.append(frames_arr[i][int(t)])
        videos_n_frames.append(frames)

    videos_n_frames=np.array(videos_n_frames)
    print(videos_n_frames.shape)
    return videos_n_frames

#---------------------------------------------------------------------------

#Helper functions for predictions and gradcam heatmaps

#Modified function Grad cam ++
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Generate class activation heatmap"""
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads,axis=(0,1,2,3))
    
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel (equivalent to global average pooling)
    #pooled_grads = tf.reduce_mean(grads, axis=(0,1,2,3))

    # We multiply each channel in the feature map array
    # by 'how important this channel is' with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output@pooled_grads[...,tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    print(heatmap.shape)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # Notice that we clip the heatmap values, which is equivalent to applying ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_resized_heatmap(heatmap, shape):
    """Resize heatmap to shape"""
    # Rescale heatmap to a range 0-255
    upscaled_heatmap = np.uint8(255 * heatmap)

    upscaled_heatmap = zoom(
        upscaled_heatmap,
        (
            shape[0] / upscaled_heatmap.shape[0],
            shape[1] / upscaled_heatmap.shape[1],
            shape[2] / upscaled_heatmap.shape[2],
            #shape[3] / upscaled_heatmap.shape[3],
        ),
    )
    
    
    return upscaled_heatmap

def get_bounding_boxes_m(heatmap, threshold=0.8, otsu=False):
    """Get bounding boxes from heatmap"""
    p_heatmap = np.copy(heatmap)

    if otsu:
        # Otsu's thresholding method to find the bounding boxes
        #print(cv2.THRESH_OTSU)
        threshold, p_heatmap = cv2.threshold(
            heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # Using a fixed threshold
        p_heatmap[p_heatmap < threshold * 255] = 0
        p_heatmap[p_heatmap >= threshold * 255] = 1

    # find the contours in the thresholded heatmap
    contours = cv2.findContours(p_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # get the bounding boxes from the contours
    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append([x, y, x + w, y + h])

    return bboxes


def get_bbox_patches(bboxes, color='r', linewidth=2):
    """Get patches for bounding boxes"""
    patches = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        patches.append(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                edgecolor=color,
                facecolor='none',
                linewidth=linewidth,
            )
        )
    return patches


def create_animation(array, case, saveat,heatmap=None, alpha=0.3):
    """Create an animation of a volume"""
    #array = np.transpose(array, (2, 0, 1))
    #if heatmap is not None:
        #heatmap = np.transpose(heatmap, (2, 0, 1))
    
    fig = plt.figure(figsize=(4, 3))
    images = []
    for idx, image in enumerate(array):
        # plot image without notifying animation
        image_plot = plt.imshow(image, animated=True, cmap='gray',aspect='auto')
        aux = [image_plot]
        if heatmap is not None:
            image_plot2 = plt.imshow(
                heatmap[idx], animated=True, cmap='jet', alpha=alpha, extent=image_plot.get_extent(),aspect='auto')
            aux.append(image_plot2)

            # add bounding boxes to the heatmap image as animated patches
            bboxes = get_bounding_boxes_m(heatmap[idx],threshold=.40,otsu=True)
            patches = get_bbox_patches(bboxes)
            aux.extend(image_plot2.axes.add_patch(patch) for patch in patches)
        images.append(aux)
    print(heatmap.shape)
    print(len(images))
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    #plt.title(f'{case}', fontsize=16)
    
    ani = animation.ArtistAnimation(
        fig, images, interval=5000//len(array), blit=False, repeat_delay=2000)
        
        #save animation as mp4 file
    
        
        
    animation_filename=f'{saveat}{case}.mp4'
    print(animation_filename)
    #writer=PillowWriter(fps=8)
    writer=animation.FFMpegWriter(fps=4,extra_args=['-vcodec', 'libx264'])
    ani.save(animation_filename,writer=writer)
    print('animation saved')
    plt.close()
    
    return animation_filename
#X=read_frames_gcam('./uploads/v_ApplyEyeMakeup_g07_c01.avi.mp4',each_nth=4)
#X=select_frames_gcam(X, 24)



        