# EMG Gesture Recognition Project

This project focuses on classifying EMG (electromyography) signals into hand gesture categories using both classical machine learning and deep learning models. It utilizes a publicly available dataset containing multi-channel EMG signals recorded with the MYO Thalmic armband.

## Project Pipeline

![Project Pipeline Diagram](pipeline_diagram.png)

The project pipeline is modularized across three Jupyter notebooks:

### 1. Data Exploration (`Data_Exploration.ipynb`)

This notebook handles data loading, cleaning, segmentation, and visualization of the raw EMG signals. It prepares windowed datasets that are subsequently used for training both Machine Learning (ML) and Deep Learning (DL) models.

**Key Steps:**

* **Data Loading:** Loads and merges all `.txt` files from the dataset.

* **Cleaning:** Drops uninformative classes, specifically "unmarked" (0) and "extended palm" (7).

* **Segmentation:** Segments the data into continuous gesture chunks and splits it into training, validation, and testing sets, ensuring no overlap between gestures.

* **Windowing:** Extracts fixed-length sliding windows (250 samples) and normalizes EMG channels across all sets using training set statistics.

* **Visualization:** Visualizes raw signals and their class transitions.

### 2. Classical Machine Learning (`ML-GestureRecognition.ipynb`)

This notebook implements classical ML models (KNN, Random Forest, SVM) using engineered time-domain features extracted from the EMG windows. It provides a baseline performance with explainable models.

**Key Features & Techniques:**

* **Feature Extraction:** Extracts handcrafted features per channel per window: Mean, Std, Min, Max, Skewness, Kurtosis, RMS, Energy, and Zero-crossings.

* **Dimensionality Reduction:** Applies PCA while retaining 95% of the variance.


* **Validation:** Uses `GroupShuffleSplit` to ensure person-wise generalization.


* **Hyperparameter Tuning:**

    * **KNN:** Tuning `k`.
    
    * **Random Forest:** Tuning `n_estimators`, `max_depth`.
    
    * **SVM:** Tuning `C`, `kernel`.
    
* **Evaluation:** Evaluates models on test data using confusion matrices and classification reports.

### 3. Deep Learning (`DL-GestureRecognition.ipynb`)

This notebook acts as the advanced model, training a 1D Convolutional Neural Network (CNN) to learn features and classify EMG windows directly from raw signal patterns.

**Architecture & Training:**


* **Input:** Reuses the windowed, normalized data prepared in the Data Exploration step.


* **CNN Architecture:** A sequential model consisting of `Conv1D` + `ReLU`, `BatchNorm`, `Dropout`, `MaxPooling`, and final `Dense` layers with softmax for classification.


* **Callbacks:** Utilizes `EarlyStopping` and `ReduceLROnPlateau` for efficient training.



**Hyperparameter Tuning:**
The notebook performs two types of grid search tuning:


1.  **Segmentation Tuning:** Optimizing `window_size` and `step`.

2.  **Model Architecture Tuning:** Optimizing filter count, kernel size, dropout, weight decay, learning rate, and dense units.

**Evaluation:**

* Evaluates both window-level and segment-level accuracy using majority voting.


* Visualizes training performance and confusion matrices.

---

## Contact

This project was done by Laia Colomé and Joana Ros, if you have any questions or feedback regarding this project, please feel free to reach out:


* **Laia Colomé Xicoy**
  * Email: [lcolxic@gmail.com](mailto:lcolxic@gmail.com)
  * LinkedIn: [Profile](https://www.linkedin.com/in/laia-colom%C3%A9-xicoy-983788243/)
  * GitHub: [github.com/lacxy05](https://github.com/lacxy05)

* **Joana Ros Alonso**
  * Email: [joana.ros.alonso@gmail.com](mailto:joana.ros.alonso@gmail.com)
  * LinkedIn: [Profile](https://www.linkedin.com/in/joana-ros-alonso-414a5629b/)
  * GitHub: [github.com/traspami](https://github.com/traspami)


