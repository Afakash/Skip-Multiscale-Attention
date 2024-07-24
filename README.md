# Skip-Encoder and Skip-Decoder for Optical Remote Sensing Object Detection
* The origin name is ‘Skip-Multiscale-Attention for Detection Transformer in Remote Sensing Imagery’.
This repository is dedicated to open-sourcing the code from our paper.
The paper is currently under review, and the specific code has been made public now.

## Outline
1. Environment setup
2. Get Started
3. Inference
4. Training your datasets
5. Reference

## Environment setup
### Dependency
You should install the origin [DINO's code (Github)](https://github.com/IDEA-Research/DINO).

### Code Instruction
After downloading the original DINO, you can use the code files provided on this page for replacement. 
Here are some introductions about the code provided on this page:
1. NWPU VHR-10 folder: [download NWPU VHR-10.v2 dataset](https://www.kaggle.com/datasets/feifanyang6755/nwpu-vhr-400v2/data)
    - config: This folder includes the hyperparameter settings used in our paper when training the model: “DINO_4scale.py”.

       - If you want to further use other stronger skeletons provided by the original DINO, you can also refer to some of the settings in it.
               
       - If you need to use these codes to train your dataset, be sure to modify the num_classes in the hyperparameters to the number of categories in your dataset + 1.
     
    - datasets: This folder also only contains a “coco.py”, which mainly involves some settings about data augmentation and dataset reading.
       -  In the NWPU VHR-10 dataset, our dataset uses the label format of the coco dataset. The specific organization method is as follows:
    ```
     NWPU VHR-10.v2
        -test_cocoformat.json
        -train_cocoformat.json
        -trainval_cocoformat.json
        -val_cocoformat.json
        -JPEGImages/
           -00001.jpg
           -00002.jpg
           ...
    ```
     - models: This folder includes 3 files:
         - __skip_encoder_decoder.py__:  __This file includes our main work, the Skip-Encoder module, and the Skip-Decoder module__.
           
         - deformable_transformer.py: This file is the building of the Transformer part of the original DINO, in which we modified some of the code for adjusting the use of SE and SD modules.
           
         - dino.py: Compared to the file of the original DINO, we have made some minor modifications to adapt the SE and SD modules.
   
     - engine.py: This file adds a function for evaluating the mAP of each category.
     - main.py: Some interpreter parameters have been added for reading the dataset, and the parallel code in the original DINO has been changed to use the accelerator parallel from Hugging Face.
     - requirements.txt: The specific version numbers of each library have been specified to prevent excessive performance fluctuations.


2. DIOR folder: [download DIOR dataset](https://www.kaggle.com/datasets/feifanyang6755/dior-coco/data)
   - config: This folder includes the hyperparameter settings used in our paper when training the model: “DINO_4scale.py”.
    
       - If you want to further use other stronger skeletons provided by the original DINO, you can also refer to some of the settings in it.
         
       - If you need to use these codes to train your dataset, be sure to modify the num_classes in the hyperparameters to the number of categories in your dataset + 1.
         
   - datasets: This folder also only contains a “coco.py”, which mainly involves some settings about data augmentation and dataset reading.
       -  In the DIOR dataset, our dataset uses the label format of the coco dataset. The specific organization method is as follows:
    ```
     DIORcoco
       -Annotations/
         -DIOR_test.json
         -DIOR_trainval.json
       -JPEGImages/
         -00001.jpg
         -00002.jpg
         ...
    ```
   - models: This folder includes 3 files:
       - __skip_encoder_decoder.py__:  __This file includes our main work, the Skip-Encoder module, and the Skip-Decoder module__.
           
       - deformable_transformer.py: This file is the building of the Transformer part of the original DINO, in which we modified some of the code for adjusting the use of SE and SD modules.
           
       - dino.py: Compared to the file of the original DINO, we have made some minor modifications to adapt the SE and SD modules.
   
   - engine.py: This file adds a function for evaluating the mAP of each category.
   - main.py: Some interpreter parameters have been added for reading the dataset, and the parallel code in the original DINO has been changed to use the accelerator parallel from Hugging Face.
   - requirements.txt: The specific version numbers of each library have been specified to prevent excessive performance fluctuations.
     
## Get Started
After replacing the files according to the dataset you need to use, you can refer to the installation requirements of the [DINO](https://github.com/IDEA-Research/DINO) to install the corresponding requirements and compile the CUDA operator of deformable attention.
If the installation is correct, you can try to adjust some hyperparameters in the config file or try to use the default parameters for preliminary estimations. 
The following is the training code:
```
accelerate launch --multi_gpu --num_processes=your_gpu_nums /.../DINO-main/main.py
              \ --config_file '/.../DINO-main/config/DINO/DINO_4scale.py'
              \ --coco_path '/.../nwpu-vhr-400v2'
              \ --image_path '/.../nwpu-vhr-400v2'
              \ --train_path '/.../nwpu-vhr-400v2/trainval_cocoformat.json'
              \ --val_path '/.../nwpu-vhr-400v2/test_cocoformat.json'
              \ --output_dir '/.../DINO-main/output_dir'
              \ --backbone_dir '/.../DINO-main/backbone_dir'        
```

## Inference
If you just want to verify the effect of specific model weights, you can add ' -eval' after the training command to start the evaluation mode, and you also need to add ' -resume your_checkpoints.pth 'to load the weights.

## Training your datasets
If you want to use these modules or models to train your own dataset, you need to ensure:
1. The format of your target detection dataset is in coco format.
2. Modify the settings about the number of dataset categories in the config folder.
3. Modify the organization structure of your dataset or modify the part of the code for dataset loading.

## Reference
1. [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)
2. [Gong Cheng, Junwei Han, Peicheng Zhou, Lei Guo. Multi-class geospatial object detection and geographic image classification based on collection of part detectors. ISPRS Journal of Photogrammetry and Remote Sensing, 98: 119-132, 2014.](http://dx.doi.org/10.1016/j.isprsjprs.2014.10.002).
3. [Object Detection in Optical Remote Sensing Images: A Survey and A New Benchmark](https://arxiv.org/abs/1909.00133)
