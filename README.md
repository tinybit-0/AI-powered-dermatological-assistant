AI-Powered Dermatological Assistant:
      This repository contains the code for a machine learning project that uses a custom-trained model to classify skin lesions. It is built for educational use to demonstrate a complete machine learning workflow from dataset to a local desktop application.

Key Features:
      Custom Model: A ResNet-18 model trained to classify seven common types of skin lesions.
      Local Application: A desktop app built with Tkinter for analyzing images on your own computer.
      Minimal Setup: All necessary files, including the dataset, are included.
      
How to Use:
    1) Clone the repository:
        Open the terminal in VS Code (Ctrl + ~) and run:
           git clone https://github.com/tinybit-0/AI-powered-dermatological-assistant.git
           
    2) Open the folder:
        Go to File > Open Folder... and select the repository you just cloned.
        
    3) Install dependencies:
        pip install torch torchvision Pillow
        pip install scikit-learn
        Check the model.py file for the import commands and use 'pip-install' command to install all the required libraries if any are remaining after running the above two commands.
        
    4) Train the model:
        Run the file Model.py to create the trained model file (skin_lesion_classifier.pth).
        
    5) Run the application:
        Once the training is complete, open the app.py file. 
        Right-click inside the editor and select "Run Python File in Terminal" to launch the desktop application.

If you have an NVIDIA GPU, try to make use of CUDA, it's faster. To get it set up, you can check out a YouTube video or some website.
