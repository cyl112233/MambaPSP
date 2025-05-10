**Enhanced Semantic Segmentation in Road Scenes via MambaPSP: Integrating Mamba Architecture with PSPNet**
Official implementation of the paper:
_Published in The Visual Computer_.


Features
⚡ Fast inference with the Mamba architecture

🎯 High segmentation accuracy, especially on small objects

🧪 Pretrained models and reproducible experiments

🗂️ Full documentation and usage guide included


**Installation**
**Requirements**
Python ≥ 3.10
PyTorch ≥ 1.13
CUDA ≥ 11.8 (for GPU acceleration)
Ubuntu22.4
Additional dependencies listed in requirements.txt

_git clone https://github.com/your_username/MambaPSP.git_
cd MambaPSP
pip install -r requirements.txt

🚀 Usage
**1. Dataset Preparation**
We use the Cityscapes dataset. Please follow the official instructions to download and extract it, then update the dataset path in config.yaml.
markdown
复制
编辑
dataset/
└── cityscapes/
    ├── leftImg8bit/
    └── gtFine/

**2. Training**
bash
复制
编辑
python main.py

**3. Evaluation**

python evaluate.py 



📊 Results
Model	mIoU (%)	Pixel Accuracy (%)	
PSPNet	70.6	95.2	
MambaPSP	76.2	95.8	

_We will continue to improve and refine the code repository after the official publication of the paper, aiming to provide more comprehensive documentation, a better user experience, and enhanced reproducibility and scalability of the experiments._


