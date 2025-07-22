
![HCNN Photovoltaic Fault Detection Banner](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024102_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZ2l0aHViX2Jhbm5lcg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12WjJsMGFIVmlYMkpoYm01bGNnLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=X1zCXXXMxqGdfc3vJptbbmBrJHDficCW-r6k41JGVVV55Dr36dQQUUUaUBAp3mOgAMNujO1RmN8jRui4XrUhrCNf-aZLO5ylF7J5UZ57RhT-~9xwZkNjuV5xsWK8T8J84pPaFK8qaiO-TLCKHfHoed2tyIHcTIrJ4VCh~B2HAhmox-lCVf8jErzjkW28wulfNhhIvsxu9F3FmgqghON2MbFqKAFYxUsXiY2FlhMK9mPRaJL7zDaQEzMbpCCJW8QiF0q~YoP1IhqAj4jCzGkWn7p3GTUa5T3dLEFIA78rvPUy9JBMMoJSyW3fLX1DzvlTtce5ldMpe3zf9BlxrfKGJw__)

## Hybrid Convolutional Neural Network for Fault Detection in Electroluminescence Images of Photovoltaic Cells

**Authors:** Alan Marques da Rocha, Marcelo Marques Simões de Souza, Carlos Alexandre Rolim Fernandes

**Affiliation:** Graduate Program in Electrical and Computer Engineering, Federal University of Ceará, Sobral, Ceará, Brazil

**Correspondence:** eng.alanmarquesrocha@gmail.com

---

## Abstract

The expansion of installed capacity in photovoltaic generation systems demands automated methods for fault detection in their constituent cells. This paper proposes a hybrid convolutional neural network model for fault detection in electroluminescence images of photovoltaic panels. The model leverages the convolutional neural networks ResNet50 and VGG16 for feature extraction and the support vector machine classifier to detect faulty cells. Adjusting the model's settings using a genetic algorithm achieved accuracy rates of 98.17% and 99.67% in tests with two public datasets. The challenges that this dataset's heterogeneity imposed on training the model were addressed by data augmentation and contrast enhancement techniques. These results support that hybrid convolutional neural networks are promising for automatic defect detection in photovoltaic cells, which is important for ensuring energy conversion efficiency and extending the lifespan of photovoltaic systems.

**Keywords:** Electroluminescence, Hybrid Convolutional Neural Network, Evolutionary Genetic Algorithms, Fault Detection

---

## Key Contributions

*   **Novel HCNN Architecture:** A robust hybrid model combining CNNs (ResNet50, VGG16) for feature extraction and SVM for precise fault classification in EL images.
*   **Genetic Algorithm Optimization:** Implementation of genetic algorithms for efficient hyperparameter tuning, leading to superior model performance and generalization.
*   **Enhanced Data Handling:** Utilization of advanced preprocessing (CLAHE) and data augmentation techniques to address dataset heterogeneity and prevent overfitting.
*   **High Accuracy and Reliability:** Achieved state-of-the-art accuracy (up to 99.7%) and a high Kappa index (80.2%), demonstrating the model's effectiveness and consistency in defect detection.
*   **Comprehensive Evaluation:** Thorough validation using 5-fold cross-validation and detailed analysis of performance metrics, including confusion matrices, to ensure robust and reliable results.

This project offers a significant advancement in automated PV cell inspection, providing a powerful tool for maintaining the efficiency and longevity of solar energy infrastructure.

# Introduction

The increasing demand for clean and renewable energy has driven the installation of large-scale photovoltaic systems. However, improper maintenance of photovoltaic cells can lead to failures that compromise the efficiency and longevity of these systems. Early detection and preventive correction of defects are crucial to avoid significant economic and environmental losses. This work addresses fault detection in photovoltaic cells using advanced machine learning techniques, specifically hybrid convolutional neural networks (HCNNs).

---

# Proposed Methodology

This study proposes a Hybrid Convolutional Neural Network (HCNN) model for fault detection in electroluminescence (EL) images of photovoltaic cells. The methodology integrates the robustness of CNNs in feature extraction with the precision of Support Vector Machines (SVM) in classification. The process is divided into preprocessing, data augmentation, and HCNN construction with genetic hyperparameter tuning.

## Datasets

The work uses images from the ELPV dataset, which consists of 2,624 EL images of photovoltaic cells, with a resolution of 300x300 pixels, extracted from 44 photovoltaic modules. Two subsets were derived: DS1 (1,074 images of Si-m cells) and DS2 (1,550 images of Si-p cells). The images include functional and non-functional cells, presenting various types of defects such as cracks, microcracks, fractures, and delaminations.

![Electroluminescence Images of Photovoltaic Cells](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024103_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZmlnMDI.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12Wm1sbk1ESS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=ho~2S8FYwDc~HiFdnDSMInAyV8EumC89ten3k3lsFQgxWynJ3F3JmeDY4Ts58ydjXkYSSHE~cCpmxbJrWTCoqM911UHFUZAcB~ZZL1eU0rXkfx6jqoDR908JBX-8bdIXrDzqT8nC3IW5DoSC-kmM~CQBrpkOc5fiKvUFozXsDpkS-ds~1YXgZTxSnZuVSiI4mcykHFkNZY82-AYjtvUqZ3jBjYNfQKZSswDxnO-U8tEyzzNenBkhZwfTogB5f3kFBUD9dHL7iol-L6sG1CCuBlNWX6-zN5s2QCnfi3ihBph-sEO9yEYsU~IsGLtJAuZm5LY8ygtu8RsK4wvLZFie1A__) <br>
*Figure 1: Electroluminescence images of silicon photovoltaic cells: (a) functional monocrystalline; (b) monocrystalline with cracks and microcracks defects; (c) functional polycrystalline; (d) polycrystalline with crack, fractures, and delamination defects.*

## Preprocessing

Images are initially normalized for contrast, perspective, and standardized in grayscale levels and dimension. The Contrast Limited Adaptive Histogram Equalization (CLAHE) technique was applied to enhance the contrast of EL images, which often exhibit low visibility of subtle details due to uneven luminescence distribution. CLAHE divides the image into small regions (tiles) and performs histogram equalization within each, adjusting the distribution of grayscale levels and increasing local contrast in a controlled manner. A contrast limit is set to prevent excessive noise amplification. The block size was configured to 8x8 pixels, with a clip limit factor of 2 for noise.

![Effect of CLAHE Technique Application](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024104_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZmlnMDM.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12Wm1sbk1ETS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=BoahfyAG76h4-l7N84xoPuWp64aWbcYeKeUDWPWEj-5VMlWbal4cOTld3SNuYCd8ANilR0WLiKU4htC5susJoUkBx8Fb2xgryyxYFSmNHF~jEy7mEHPq3gNl9f8qQP2jRfxWYZBHO7SreNhDxCD-0wgS72Vmc1Q4CVb1f8VHuKmLJcQfdCMi1PlBV~T11Ohmyz8eDab4m-e8NEsRUrIYvKzNhmTmxKJ8PhIeKsfn300xrwkwBmk85gM22QSh2FCTRSVvpnfCqAJ11EhzZjyyoSHpvv7sqv4FaveJVqXn-yyzKLqAW2150GmoeA-RvToH0EvCA-ToFc9PlB38cUXa2w__) <br>
*Figure 2: (a) Original image of the Si-p photovoltaic cell. (b) Image resulting from the application of the CLAHE technique. (c) Histogram of the pixel intensity of the original image. (d) Histogram of the pixel intensity after applying the CLAHE technique.*

## Data Augmentation

To increase model robustness and prevent overfitting, four data augmentation methods were employed: image rotation (90° clockwise and counterclockwise), flipping, Gaussian blur application (3x3 kernel, $\sigma=1.0$ standard deviation), and a 20% brightness increase. The resulting dataset, named DS2, totaled 13,120 photovoltaic cell images, maintaining the original proportion of functional and non-functional cells.

<img width="900" height="509" alt="image" src="https://github.com/user-attachments/assets/d87c3c86-60f7-4d89-bf2b-974b06453949" /><br>
*Figure 3: (Result of the data augmentation techniques applied to the image dataset. (a) Original. (b) 90° clockwise rotation. (c) 90° counterclockwise rotation. (d) Flipping. (e) Blurring. (f) Brightness increased by 20%.*


## Hybrid Convolutional Neural Network (HCNN)

The proposed HCNN combines the feature extraction capabilities of CNNs with the classification precision of SVMs. The construction process involves three main steps: genetic fine-tuning, defining the hyperparameter search space, and configuring SVM parameters. ResNet50 and VGG16 architectures, pre-trained on the ImageNet dataset, were chosen as the basis for the CNNs due to their recognized performance in image classification tasks. VGG16, for example, uses convolutional, max-pooling, and ReLU layers for feature extraction, replacing fully connected layers with an SVM for final classification.

![VGG16 Architecture for Feature Extraction](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024104_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZmlnMDQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDRfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12Wm1sbk1EUS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=OOCKgknyVyKQLLOdkeY9SNAOnFh7VvLihatC7SA9VCE-wLzjV8~Vjni84C8Gwseo5YbiIQw-RLr43r6PQEpqMQbKESqHAaoyIgDdqAbKct6W~cL-0pDkcjDgEsKtucTLrNHFLUiiWyx~HwcHGiptPEwyKcZW0WYdx5GPJJ1TgKL~jpfPCpZpiOBJJGeeDxkXmU0-0QMetsC~tuHiEtaNxvOK20Tf~w4U6C9oe7fem6sG87avxog7CHwTUybJz0RBBfEEMNjyhwxW0eZFc~TcsahOxOVqJOj7abhqu8QPxy44CJJ-AwBcZnw1TAC2D7uYkb5EpsLirHeFcJivVxVIIw__)
*Figure 4: Architecture of VGG16 used for feature extraction.*

### Genetic Fine-tuning

Genetic Fine-tuning uses Genetic Algorithms (GAs) to optimize CNN hyperparameters. Inspired by natural evolution, GAs employ selection, crossover, and mutation operations to evolve solutions over generations. Hyperparameters such as network depth, number of filters, filter size, learning rate, and activation function are encoded as genes. The process follows steps such as creating an initial population, selecting individuals by stochastic roulette wheel sampling, generating new individuals through crossover and mutation, and selecting the best individuals for the next generation based on the Kappa index ($\kappa$). Stopping conditions include reaching 100% Kappa, stagnation of average accuracy, or reaching 100 generations. The hyperparameter search space includes CNNs (ResNet50, VGG16), number of layers (1, 2), neurons per layer, activation functions (tanh, relu, selu, elu, exponential), optimizer types (adam, sgd, rmsprop, adadelta), and dropout rates (30%, 40%, 50%, 60%).

### Feature Extraction

After genetic fine-tuning, the pre-trained ResNet50 and VGG16 networks are applied to the EL images of PV cells. Each convolutional layer extracts different levels of features, from simple edges to more abstract patterns specific to defect detection. The advantage of using pre-trained networks lies in their ability to reuse knowledge acquired from previous tasks, accelerating the training process and improving the accuracy of feature extraction.

### Support Vector Machine (SVM)

SVM is a supervised classification algorithm effective for high-dimensional data, such as features extracted by CNNs. SVM helps prevent overfitting, especially in scenarios with more features than samples. The SVM classifier was configured with a Radial Basis Function (RBF) kernel to transform features into a higher-dimensional space, facilitating class separation. Parameters $C$ and $\gamma$ were tuned using the GridSearchCV class from the Scikit-Learn library, testing values for $C \in \{0.1, 1, 10, 100\}$ and $\gamma \in \{0.001, 0.01, 0.1, 1\}$.

---

## Results and Discussion

Classification experiments were conducted using DS1 and DS2 datasets (with CLAHE preprocessing). Each model generated by genetic fine-tuning was trained for 100 epochs, with 80% of images for training, 10% for testing, and 10% for validation. All performance metrics were calculated using five-fold cross-validation.

### Genetic Fine-tuning Without Data Augmentation (DS1)

The genetic fine-tuning results for the top ten ResNet50+SVM and VGG16+SVM topologies on the DS1 dataset showed that the VGG16+SVM combination with a 1,024-neuron layer, selu activation function, rmsprop optimizer, and 50% dropout achieved the best results. This model reached a $\kappa$ index of 78.2% and an accuracy of 95.2%. High accuracy and $\kappa$ index indicate good classification capability and agreement with true labels. Specificity of 96.3% and sensitivity of 94.5% confirm the model's ability to correctly distinguish classes, minimizing false positives and false negatives.

ResNet50+SVM, with 768 neurons in the first layer, 512 in the second, relu activation function, adam optimizer, and 50% dropout, also showed robust performance, with 94.4% accuracy and a $\kappa$ index of 74.2%. Although VGG16+SVM proved slightly superior in prediction robustness, both models are effective for the classification task.

### Genetic Fine-tuning With Data Augmentation (DS2)

With the DS2 dataset (with data augmentation), the best results were obtained with the VGG16+SVM topology, using a 768-neuron layer, elu activation function, adadelta optimizer, and 50% dropout. This model achieved a $\kappa$ index of 80.2% and an average accuracy of 99.7%. These values indicate strong agreement between model predictions and labels, as well as high classification capability. Specificity of 98.4% and sensitivity of 97.1% reinforce the model's effectiveness in correctly distinguishing classes.

ResNet50+SVM, with a 1,024-neuron layer, relu activation function, adam optimizer, and 40% dropout, achieved 98.2% accuracy and a $\kappa$ index of 85.3%, demonstrating competitive performance. Comparison with DS1 results reveals significant improvement in performance metrics, attributed to data augmentation techniques, which provided a broader and more varied training base, resulting in more robust and generalizable models.

### Confusion Matrices

The confusion matrices for the best results obtained by HCNN VGG16+SVM and HCNN ResNet50+SVM models (with data augmentation) are presented below. The matrix for VGG16+SVM shows 1,303 TNs and 1,295 TPs, with only 13 FPs and FNs, reflecting 99.7% accuracy. For ResNet50+SVM, the matrix shows 1,292 TNs and 1,280 TPs, with 26 FPs and 26 FNs, consistent with 98.2% accuracy. Both models demonstrate high specificity and sensitivity, with VGG16+SVM showing fewer classification errors.

![Confusion Matrices of Best Results](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024105_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZmlnMDU.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12Wm1sbk1EVS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=lersIr7bSsHxvGwaHlbwFI83pl9rRrfdKlfi5n37GiJ3pMoVu7Z-4Stn~nSACtHZSiQXIYQ4Dv2gBH~VPsEn77XCzdhgcoSkcXINDQofjNN78M0US6XSR7eqJlCg7m0eH8~1PHC31kVoxoHl0W-JwdvIDkirnQ0p~Djvpg1P8U2RdkVF~1kU09qcD1Vjrt~2tCYsaJ5PmayzqJjy1XsMAh-mV9tPOOLBDK2W9vzyfiFOMbtkf-be82RwOTBnt2vEhVT6iip-ZDb-CcPFAR9SVbIjQwsApsZ34fiA-lqUgzAh--BWf~wIM8xw0DhUkJmi3Frl22VtBR1PYnVeXrW2nA__)
*Figure 5: (a) Confusion matrix of the best result obtained by HCNN VGG16+SVM. (b) Confusion matrix of the best result obtained by HCNN ResNet50+SVM.*

### Visual Validation

Visual validation of EL images of photovoltaic cells confirmed the effectiveness of the proposed technique. The model successfully detected all functional and non-functional cells. An example of a misclassified cell, with a small delamination defect, highlights the need for future improvements to classify subtle defects.

![Examples of Classifications Performed by the VGG16+SVM Model](https://private-us-east-1.manuscdn.com/sessionFile/hVVZH7u8JEL2vL7urTQxZh/sandbox/Ca7lGdn8WpWOL7x40mICAe-images_1753143024105_na1fn_L2hvbWUvdWJ1bnR1L3JlYWRtZV9pbWFnZXMvZmlnMDY.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvaFZWWkg3dThKRUwydkw3dXJUUXhaaC9zYW5kYm94L0NhN2xHZG44V3BXT0w3eDQwbUlDQWUtaW1hZ2VzXzE3NTMxNDMwMjQxMDVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzSmxZV1J0WlY5cGJXRm5aWE12Wm1sbk1EWS5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=bh-6BGAQLH-VJpyjzpITSj5Hha8T-xwsvRb-Iy~og4xgUxBsNHy7YkIIFo5SmUmbhV-NJEzLAx0HBFFm92qFoXo5svG7avppqHEanD07S65o7XREUCbQ6qOgsBym1uMH5ubVIw60AJdla4k1wbR0DlsMD5VS8QFQYsvrsNxLq3HB0dgNXH8GpjWXvN4UqDC8iO5uCWwYzlYHig7cME-aKZA866UHiS3fpOPl9W-2CTZo-cqA~b2gPF2zVmO-wQ~E1aViEBfxBIPuIR7ftNOuysGEGE4zLJE6MA35IBlfk1H0FrxHCclebCdKwmwOlmBbbtyfG66~ZOp8KT5lZIEZIw__)
*Figure 6: Examples of classifications performed by the VGG16+SVM Model with data augmentation.*

### Comparison with State-of-the-Art

This work stands out for its use of advanced ML techniques and for achieving superior results in almost all evaluated metrics. The average accuracy of 99.7% is the highest among compared studies. Sensitivity, precision, specificity, and F-score obtained are also superior or comparable to the best results in the literature. Furthermore, this approach is the only one to include the $\kappa$ metric (80.2%) and to implement evolutionary genetic algorithms for hyperparameter optimization, confirming the consistency and reliability of the results.

---

## Conclusions

We propose a new HCNN model for detecting functional and non-functional photovoltaic cells based on EL images. The methodology includes genetic hyperparameter tuning to optimize CNN models. The combination of CNN and SVM algorithms formed a new HCNN topology, where the CNN extracts features from EL images and the SVM performs classification. The model was trained and evaluated using DS1 and DS2 datasets (with data augmentation).

Results showed that the proposed model accurately detected functional and non-functional panels, achieving 99.7% accuracy and a $\kappa$ index of 80.2%, values superior to similar recent works. A limitation identified is the difficulty in correctly classifying certain defects located in cell corners. Future work will focus on analyzing unrecognized defects to improve efficiency and expanding datasets to cover all types of defects in photovoltaic cells.

---

## How to Cite

If you use this work in your research, please cite the original article:

Rocha, A. M., Souza, M. M. S., & Fernandes, C. A. R. (YYYY). Hybrid Convolutional Neural Network for Fault Detection in Electroluminescence Images of Photovoltaic Cells. *Revista de Informática Teórica e Aplicada - RITA*, Vol. XX, Num. XX, pp. 11-XX. DOI: http://dx.doi.org/10.22456/2175-2745.XXXX

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please contact the authors via email: eng.alanmarquesrocha@gmail.com

---

## Acknowledgments

The authors thank the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) for financial support and FUNCAP/Brazil (Grant BP5-0197-00183.01.06/23) for partial funding.


