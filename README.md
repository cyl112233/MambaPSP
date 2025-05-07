# MambaPSP
This study addresses the critical challenge of achieving both high accuracy and real-time performance in semantic segmentation for autonomous driving applications. 



Project Background
With the rapid advancement of autonomous driving technology, environmental perception has become a core component of autonomous systems, tasked with acquiring and interpreting real-time surrounding environment data to provide critical support for subsequent decision-making and motion control. However, existing deep learning-based semantic segmentation models often face a trade-off between inference speed and accuracy.

To address this challenge, MambaPSP integrates state-space models (Mamba) with global receptive fields and dynamic weighting mechanisms, significantly enhancing the model’s capability to capture fine-grained features while improving global semantic understanding. This innovation demonstrates superior stability and precision, particularly in segmenting small objects and refining boundary delineation in complex urban driving scenarios.


MambaPSP/
├── datasets/           # 数据集相关处理
├── models/             # 模型定义，包括 Mamba 模块和 PSPNet 架构
├── utils/              # 辅助函数和工具
├── main.py            # 脚本         
├── requirements.txt    # 项目依赖
└── README.md           # 项目说明文件


