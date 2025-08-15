# Spotting the Fakes: CNN-Based Real vs. AI-Generated Face Classification

## Overview
With the rapid advancement of AI-generated image technology, distinguishing synthetic faces from real ones is becoming increasingly difficult. This project develops a deep learning–based system to **classify whether a face is real or AI-generated**, leveraging **Convolutional Neural Networks (CNNs)** and modern architectures.  

We experimented with multiple models across datasets of varying complexity to identify architectures that perform well both in controlled and challenging real-world scenarios.

---

## Goals
- Detect and classify real vs. AI-generated faces.  
- Compare performance of different CNN architectures.  
- Evaluate how dataset complexity impacts model performance.  
- Optimize training strategies to improve robustness and generalization.

---

## Datasets
We used two datasets with contrasting levels of complexity:

1. **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** (Kaggle)  
   - Equal number of real and AI-generated faces.  
   - High-resolution, clean, frontal images.  
   - Balanced and relatively easy to classify.  

2. **[WildDeepfake](https://arxiv.org/abs/2101.01456)**  
   - Real-world deepfakes from online videos.  
   - Varied lighting, angles, resolutions.  
   - Includes blur, artifacts, and background noise.  
   - Far more challenging for classification.

---

## Research Process
We iteratively tested different architectures and training strategies:

| Step | Model           | Dataset                | Result |
|------|----------------|------------------------|--------|
| 1    | ResNet18       | 140k Faces (Simple)    | Average accuracy ✗ |
| 2    | ResNet18       | WildDeepfake (Complex) | Low accuracy ✗ |
| 3    | ResNet50       | 140k Faces             | High accuracy ✓ |
| 4    | ResNet50       | WildDeepfake           | Average accuracy ✗ |
| 5    | ConvNeXt-Tiny  | WildDeepfake           | High accuracy ✓ |

---

## Best Model: ConvNeXt-Tiny
We achieved our best performance using **ConvNeXt-Tiny**, a modern CNN architecture inspired by vision transformers but optimized for efficiency.

### Why ConvNeXt-Tiny?
- **Next-gen design**: Modern architecture with strong feature extraction.  
- **Lightweight**: Fast and GPU-friendly for large datasets.  
- **Robust**: Performs well on complex, noisy, real-world images.

---

## Training Strategy
We used **progressive unfreezing** for fine-tuning:
1. **Phase 1:** Train only the final fully connected (FC) layer → **74.7%** validation accuracy.  
2. **Phase 2:** Unfreeze final stage + FC → **89.4%** validation accuracy.  
3. **Phase 3:** Unfreeze all layers → **92.2%** validation accuracy.  

**Final WildDeepfake Accuracy:** ~**95%**

---

## Repository Structure

```
Spotting_the_Fakes/
│
├── README.md                              # Project overview and documentation
├── resnet18.ipynb                         # Jupyter notebook: implements and evaluates ResNet-18 model
├── resnet50_on_kaggle.ipynb               # Jupyter notebook: ResNet-50 model trained/evaluated on the Kaggle dataset
├── resnet50_on_wilddeepfake.ipynb         # Jupyter notebook: ResNet-50 model trained/evaluated on WildDeepfake dataset
└── cinvnext_tiny_wilddeepfake.ipynb       # Jupyter notebook: ConvNeXt-Tiny model applied to WildDeepfake dataset```
```

File Descriptions:
| File | Description | 
|------|----------------|
|README.md| The main documentation file. Describes the project goal of detecting AI-generated or fake faces, outlines datasets used, model architectures, experimental results, and training insights.|
|resnet18.ipynb| A notebook running experiments with the ResNet-18 architecture. It likely includes data loading, training, evaluation, and performance metrics (e.g., accuracy on real vs. fake face classification).|
|resnet50_on_kaggle.ipynb| Contains experiments using ResNet-50 on the cleaner, Kaggle-sourced dataset (likely the "140k Real and Fake Faces"). Demonstrates how this model performs on high-resolution, balanced images.|
|resnet50_on_wilddeepfake.ipynb| Applies ResNet-50 to the more challenging WildDeepfake dataset. Designed to show how the model handles real-world deepfake videos, with varying quality and noise conditions.|
|cinvnext_tiny_wilddeepfake.ipynb | Focuses on ConvNeXt-Tiny (a modern, efficient CNN architecture) applied to the WildDeepfake dataset. Likely documents progressive unfreezing strategies, training performance, and any resulting accuracy improvements.|---


##  Setup & Usage

1. *Clone the repository*
    bash
    git clone https://github.com/hzeriene/Spotting_the_Fakes.git
    cd Spotting_the_Fakes
    

2. *Install Dependencies*
    bash
    pip install -r requirements.txt
   

3. *Token Setup*
   The code requires access to datasets from Kaggle and Hugging Face that need authentication tokens:

      Kaggle Token:
      - Go to your Kaggle account settings and create a new API token
      - Download the `kaggle.json` file and place it in `~/.kaggle/` directory
      - Or set up Kaggle API credentials as environment variables
      
      Hugging Face Token:
      - Sign up at [Hugging Face](https://huggingface.co)
      - Generate an access token at: https://huggingface.co/settings/tokens
      - Create a `.env` file in the root of the project:
        ```
        HF_TOKEN=your_huggingface_token_here
        ```
      
      In each notebook file, update this section:
      ```python
      # Replace "your_own_huggingface_token" with your actual Hugging Face access token
      # Get one at: https://huggingface.co/settings/tokens
      myTtoken = "your_own_huggingface_token"
      ```
5. *Run Training or Evaluation*
    - Open the appropriate notebook (e.g., cinvnext_tiny_wilddeepfake.ipynb) and execute the cells.
    - Ensure any dataset paths or configurations are correctly set—typically, there will be variables like DATA_DIR to point to downloaded dataset location.

6. *Hyperparameters*

     batch_size = 32
     learning_rate = 1e-4
     epochs = 10
     weight_decay = 1e-5
     max_samples_training = 4000
     max_smaples_test = 2000
     

## Key Insights
- **Architecture matters**: Modern, well-designed smaller models can outperform larger ones on complex datasets.  
- **Dataset complexity is critical**: A model that excels on clean datasets may fail on real-world noisy data.  
- **Iterative experimentation wins**: Progressive testing and tuning led to consistent improvements.

---

## Future Work
- Test **cross-dataset generalization** to ensure robustness beyond training data.  
- Develop **lighter versions** for deployment on mobile and edge devices.  

---

## References
- Dataset: [WildDeepfake](https://arxiv.org/abs/2101.01456)  
- Dataset: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  
- Architecture: [ConvNeXt](https://arxiv.org/abs/2201.03545)  






## Citing

If you use this work or datasets in your research, please cite the following:

**WildDeepfake dataset:**
```bibtex
@inproceedings{zi2020wilddeepfake,
  title     = {WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection},
  author    = {Zi, Bojia and Chang, Minghao and Chen, Jingjing and Ma, Xingjun and Jiang, Yu-Gang},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  pages     = {2382--2390},
  year      = {2020}
}
```

**140k Real and Fake Faces dataset:**
```bibtex
@misc{xhlulu140k,
  title        = {140k Real and Fake Faces},
  author       = {Xingjun Ma},
  howpublished = {\url{https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces}},
  year         = {2019}
}
```

**ConvNeXt architecture:**
```bibtex
@article{liu2022convnet,
  title   = {A ConvNet for the 2020s},
  author  = {Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022}
}
```










  
