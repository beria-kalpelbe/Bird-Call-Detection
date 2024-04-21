# Deep Learning in Bio-acoustic: case of Bird call detection
<a href="https://doi.org/10.5281/zenodo.10675498">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10675498.svg" alt="DOI"></a>
<a href="https://colab.research.google.com/drive/1jQNUamyTX5-k_JV7f7WF34VAr16VnmGS?usp=sharing"> 
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
 </a>

Here, we developed three Machine Learning Models to detect Bird calls in the audio sequences. The models explored are: Support Vector Machine (SVM), CNN binary classification and the pre-trained ResNet50V2 model.

<img src="https://ml-rds-wordpress-prod-s3.s3.amazonaws.com/uploads/2021/06/Sound-ID-ML-Prothonotary-Warbler.png">


## Data collection
The data used in this work were recorded directly in the field at <a href="https://www.google.com/maps/place/Intaka+Island/@-33.888283,18.5040338,16z/data=!4m10!1m2!2m1!1sIntaka+Island!3m6!1s0x1dcc5c0500e09a03:0x6849fe1bc7618fc5!8m2!3d-33.888283!4d18.513561!15sCg1JbnRha2EgSXNsYW5kkgEQbmF0aW9uYWxfcmVzZXJ2ZeABAA!16s%2Fg%2F1hm292vgp?entry=ttu">Intaka Island</a> by a group of 26 students at AIMS South Africa evenly distributed over the collection area using Rasberry Pi. 

<iframe src="https://www.google.com/maps/embed?pb=!1m14!1m8!1m3!1d6624.106453261446!2d18.5040338!3d-33.888283!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x1dcc5c0500e09a03%3A0x6849fe1bc7618fc5!2sIntaka%20Island!5e0!3m2!1sen!2sza!4v1713710906302!5m2!1sen!2sza" width="100%" height="450" frameborder="0" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>

To keep the Raspberry Pi running smoothly, a background job has been set up so that as soon as the device is switched on, the sound recording task begins. Sequences of 30s of audio are then recorded.

<div style="text-align: center;">
    <img src="rasberry.jpg" width="100%" alt="Raspberry Image">
</div>




## Setting up
### 1. Load data file

To keep the code running, you need to download the dataset from zenodo:

    wget https://zenodo.org/record/10675498/files/Data.zip
    unzip Data.zip
    
   

### 2. Create virtual environment in your workspace

    python -m venv venv
    source venv/bin/activate

### 3. Clone the github repository

    git clone https://github.com/beria-kalpelbe/Bird-Call-Detection.git
    cd Bird-Call-Detection

### 4. Install requirements

    pip install -r requirements.txt

### 5. Pre-process Training data

    python preprocess-train.py

### 5. Pre-process Testing data

    python Ppreprocess-test.py

### 6. Train the CNN from scratch

    python cnn.py

### 7. Train the model using transfer learning with ResNet50V2

    python resnet50.py

### 8. Train the model using SVM

    python svm.py

### 9. Access to the results of the models

All the results are stored in the Plots folder. 
