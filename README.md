### Authors
Annalisa Gallina, Enrico Gottardis, Elisa Silene Panozzo

This project explores the Car Make Verification (CMV) task - determining whether two car images belong to the same manufacturer - using two different deep learning strategies:

1. **Classification + SVM**  
   A fine-tuned ResNet18 model is used as a feature extractor, followed by a Support Vector Machine trained on feature differences.

2. **Siamese Network**  
   A twin-branch neural network trained with Contrastive Loss to learn similarity between image pairs.

The study compares both methods on the **CompCars dataset**, evaluating accuracy and performance trade-offs.

## Classification + SVM
- **Backbone:** ResNet18 pre-trained on ImageNet  
- **Optimizer:** SGD (lr = 1e-4)  
- **Loss:** Cross Entropy  
- **Verification:** SVM with RBF kernel on feature differences  
- **Result:** 78.57% verification accuracy

## Siamese Network
- **Architecture:** Dual ResNet18 branches with shared weights  
- **Embedding dimension:** 128, ℓ2-normalized  
- **Loss:** Contrastive Loss (margin = 1.5)  
- **Result:** 68.03% test accuracy (color images)

## Dataset

- **Dataset:** [CompCars (Yang et al., CVPR 2015)](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/)
- **Images:** 136,727 web images  
- **Classes:** 163 makes, 1,716 models  
- **Preprocessing:** resize to 256×256, center crop 224×224, random augmentations

## Results Summary

| Method | Accuracy (Test) | Positive Acc. | Negative Acc. |
|:-------|:---------------:|:--------------:|:--------------:|
| Classification + SVM | **78.57%** | 80% | 78% |
| Siamese (B&W) | 63.89% | 59.41% | 69.39% |
| Siamese (Color) | 68.03% | 64.40% | 71.93% |

## Conclusions

The **Classification + SVM** approach outperformed the Siamese Network, achieving over 10% higher accuracy.  
This demonstrates that leveraging pre-trained feature extractors combined with traditional classifiers remains a strong baseline for car make verification tasks.

## Technologies
- Python 3.10  
- PyTorch  
- scikit-learn  
- NumPy, Matplotlib, UMAP  
