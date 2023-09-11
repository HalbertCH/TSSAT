# TSSAT: Two-Stage Statistics-Aware Transformation for Artistic Style Transfer
This is the official PyTorch implementation of our paper: "TSSAT: Two-Stage Statistics-Aware Transformation for Artistic Style Transfer". (**ACM MM 2023**) 

Artistic style transfer aims to create new artistic images by rendering a given photograph with the target artistic style. Existing methods learn styles simply based on global statistics or local patches, lacking careful consideration of the drawing process in practice. Consequently, the stylization results either fail to capture abundant and diversified local style patterns, or contain undesired semantic information of the style image and deviate from the global style distribution. To address this issue, we imitate the drawing process of humans and propose a Two-Stage Statistics-Aware Transformation (TSSAT) module, which first builds the global style foundation by aligning the global statistics of content and style features and then further enriches local style details by swapping the local statistics (instead of local features) in a patch-wise manner, significantly improving the stylization effects. Moreover, to further enhance both content and style representations, we introduce two novel losses: an attention-based content loss and a patch-based style loss, where the former enables better content preservation by enforcing the semantic relation in the content image to be retained during stylization, and the latter focuses on increasing the local style similarity between the style and stylized images. Extensive experiments verify the effectiveness of our method.

<div align=center>
<img src="https://github.com/HalbertCH/TSSAT/blob/main/figures/overview.jpg" width="900" alt="Pipeline"/><br/>
</div>

## Requirements  
We recommend the following configurations:  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1

## Model Training  
- Download the content dataset: [MS-COCO](https://cocodataset.org/#download).
- Download the style dataset: [WikiArt](https://www.kaggle.com/c/painter-by-numbers).
- Download the pre-trained [VGG-19](https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view?usp=sharing) model.
- Run the following command:
```
python train.py --content_dir /data/train2014 --style_dir /data/WikiArt/train
```

## Model Testing
- Put your trained model to *model/* folder.
- Put some sample photographs to *content/* folder.
- Put some artistic style images to *style/* folder.
- Run the following command:
```
python Eval.py --content content/1.jpg --style style/1.jpg
```
We provide the pre-trained model in [link](https://drive.google.com/file/d/1r3T-oA7yN-pLT-M-DpQ2XO_-Y2bbJ92e/view?usp=sharing). 

## Comparison Results
We compare our model with some existing artistic style transfer methods, including [AdaIN](https://github.com/naoto0804/pytorch-AdaIN), [WCT](https://github.com/eridgd/WCT-TF), [Avatar-Net](https://github.com/LucasSheng/avatar-net), [SANet](https://github.com/GlebBrykin/SANET), [ArtFlow](https://github.com/pkuanjie/ArtFlow), [IEST](https://github.com/HalbertCH/IEContraAST), [AdaAttN](https://github.com/Huage001/AdaAttN), and [StyTr<sup>2](https://github.com/diyiiyiii/StyTR-2).  

![image](https://github.com/HalbertCH/TSSAT/blob/main/figures/comparison.jpg) 

 ## Acknowledgments
The code in this repository is based on [SANet](https://github.com/GlebBrykin/SANET). Thanks for both their paper and code.
