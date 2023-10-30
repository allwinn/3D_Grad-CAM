# 3D_Grad-CAM
The 3D Grad-CAM tool was designed with the main objective of helping understand 3D
neural networks better using visualization techniques like heatmaps. The 3D Grad-CAM
tool is one of the first tools designed for the analyses of 3D CNN networks. It extends the
Grad-CAM technique to handle 3D inputs and uses it for generating visual explanations
for a 3D CNN network using heatmaps. The tool is designed as an easy-to-use web
application with a simple and user-friendly interface.

### Running the 3D Grad-CAM app
The 3D Grad-CAM tool is a very simple python flask based web application that is dependent on basic python libraries like TensorFlow, OpenCV, NumPy,
flask ,matplotlib and few others along with the FFmpeg software installed and configured.

1. After the environment setup navigate to the root folder of the 3D Grad-CAM app where the app.py file is
present.
2. Run the command python app.py --pred_index 0
3. This will start the flask server on the localhost (127.0.0.1) on the default port 5000.
4. Open any web browser and type in the address http://127.0.0.1:5000
5. This should open up the 3D Grad-CAM app main page as shown below
   
   <img width="956" alt="page_initial" src="https://github.com/allwinn/3D_Grad-CAM/assets/51354802/c89468a3-82a8-4f63-a61b-91820509a904">

### Predicting using the 3D Grad-CAM app
1. Upload the sample 3D CNN model file.
   Note:The model file must be a TensorFlow .h5 file containing atleast one instance of a Conv3D layer. PyTorch models are currently not supported.
2. Upload the sample video to be tested. the video can be played after it is sucessfully uploaded
3. Click on the ‘Preprocess video’ button.
4. Click on the ‘Make Predictions’ button.
5. After getting the predictions click on the ‘Analyze’ button to generate the heatmaps for interpretation of the predictions made by the model.

    <img width="906" alt="wrong_1_a" src="https://github.com/allwinn/3D_Grad-CAM/assets/51354802/203e806b-0543-4e33-a22d-6ff1284f4a47">

### Analysing the predictions using the 3D-Grad-CAM app

The heatmap videos can be viewed under **Heatmap Videos** tab.


<img width="969" alt="wrong_1_b" src="https://github.com/allwinn/3D_Grad-CAM/assets/51354802/c00b3265-7e47-415f-ab44-c83f396c2f79">



The heatmap images can be viewed under the **Heatmap Images** tab.


<img width="953" alt="wrong_1_c" src="https://github.com/allwinn/3D_Grad-CAM/assets/51354802/e5e7a85f-dfe9-45cd-8015-cf6335cd9d90">



For wrong predictions the inference maps can be viewed under the **Inference Heatmaps** tab.


<img width="959" alt="wrong_1_d" src="https://github.com/allwinn/3D_Grad-CAM/assets/51354802/c5bfb3c9-6332-49ca-b3c1-523db8276478">


