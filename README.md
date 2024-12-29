## Data Mining course at Queen’s University Master’s program.
comptition link : https://www.kaggle.com/competitions/cisc-873-dm-w24-a4
# Airbnb-price-category-prediction
This project predicts Airbnb listing price categories (beginner, plus, premium) based on listing characteristics to assist new hosts. The dataset contains listings from Montreal in 2019 and includes various features like location and amenities. The goal is to classify listings into three pricing categories to optimize user experience.

### Problem Formulation:

#### Define the problem:
The problem is to predict the listing price category (beginner, plus, premium)(0,1,2) based on the listing characteristics. 

#### Input:
- Input includes various features of the listing such as text descriptions (summaries), images, and other attributes like location, amenities, etc.

#### Output:
- The output is the predicted price category, which is one of the three classes: beginner, plus, or premium.

#### Data Mining Function Required:
- Classification is the data mining function required to predict the price category.

## **Challenges:**
- **Handling Missing Values**: Missing or corrupt data in any column, including the 'summary' column, may disrupt the analysis. Proper handling is required to maintain data integrity.
- **Dealing with Different Languages in Summary**: Summaries in different languages require language detection or translation for consistency in analysis. Language-specific analysis may also be necessary.
- **Memory Management**: Efficient memory management is essential, especially when dealing with large datasets or when loading images into memory. Techniques such as batch loading and generator functions can help prevent memory overflow.
- **Preprocessing Images**: Image preprocessing tasks such as resizing, normalization, and augmentation can be computationally intensive. Efficient techniques need to be implemented to manage computational resources effectively.
- **Handling Missing or Corrupt Images**: Missing or corrupt image files may pose challenges during image loading. Proper error handling is required to ensure that missing images do not disrupt the preprocessing pipeline.
- **Data Augmentation for Images**: Techniques such as rotation, flipping, and cropping may be necessary to increase the diversity of the image dataset. However, applying these techniques while maintaining data integrity and avoiding overfitting can be challenging.
- **Compatibility and Dependency Issues**: Preprocessing libraries and functions may have dependencies or compatibility issues across different environments. Ensuring compatibility and proper installation of dependencies is crucial for smooth preprocessing.

### Preprocessing Steps for Image and Text Data

#### Image Data Preprocessing:
- **Loading Images:**
    - Images are loaded from the specified directory.
    - Each image is converted to grayscale and resized to 64x64 pixels.
    - If an error occurs during loading (file not found), a blank image (zeros) of size 64x64x2 is returned.
    - Images are loaded for both the training and test datasets.

#### Text Data Preprocessing:
- **Drop Missing Data:**
    - Rows with missing data (NaN values) are dropped from the dataset.
- **Tokenization:**
    - Text data (summaries) are tokenized.
    - The tokenizer is fit on the training data [summary].
- **Translation (Non-English to English):**
    - Non-English summaries are translated to English using a translation service.
    - Translation is applied to each summary after tokenization.
- **Padding:**
    - Summaries are padded to the maximum sequence length to ensure uniform length for model input.

## Experimental Protocol

1. **Data Loading:**
    - Load the image data for training and testing from the 'img_train' and 'img_test' directories.
    - Load the training and test datasets from the 'train_xy.csv' and 'test_x.csv' files respectively.
2. **Data Cleaning:**
    - Perform initial data exploration and identify any missing values or anomalies in the dataset.

3. **Preprocessing:**
    - Apply preprocessing steps to handle missing values, scale numerical features, and encode categorical features using pipelines.
    - Preprocess image data by resizing, normalizing, and augmenting the images.
    - Preprocess text data by tokenizing, padding, and translating non-English summaries to English.

4. **Model Selection:**
    - Experiment with various classification models such as Logistic Regression, GRU, BiDirectional LSTM, Convolutional Neural Network (CNN), and multi-modality models.

5. **Hyperparameter Tuning:**
    - Utilize different hyperparameter tuning techniques including Grid Search, Random Search, and Bayesian Optimization to optimize model performance.
    - Tune hyperparameters such as learning rate, batch size, dropout rate, and layer configurations.

6. **Model Evaluation:**
    - Evaluate each model's performance using accuracy as the metric on the validation set.
    - Analyze the model's performance to identify underfitting or overfitting issues.
  
### **Impact:**
- The model's predictions can optimize user experience by recommending appropriate pricing ranges to new hosts, potentially increasing the likelihood of successful listings.

### **Ideal Solution:**
- Ideal Solution: An ideal solution would involve leveraging both neural network-based models and traditional machine learning algorithms to build a multi-modal model that incorporates both text and image data for predicting the listing price category accurately. 

- For neural network-based models, such as Bidirectional LSTM with CNN (Trail 2), optimizing the architecture by experimenting with different numbers of LSTM units, convolutional layers, and dropout rates is crucial. Techniques like early stopping and batch normalization can help prevent overfitting and improve generalization.

- For other trails, such as Random Forest with Random Search (Trail 10), fine-tuning hyperparameters and exploring different feature engineering techniques can enhance performance. Techniques like oversampling for imbalanced classes, as demonstrated in Trail 8, can also improve learning from the data.

- The ideal solution should handle missing values, different languages, memory constraints, image preprocessing, and compatibility issues efficiently, regardless of the modeling approach. Additionally, the model should be robust to diverse listing characteristics and provide interpretable predictions for better understanding and decision-making.

## Model Tuning and Documentation:
| Trail | Model | Reason | Expected Outcome | Observations |
|-------|-------|--------|------------------|--------------|
| 1     | LSTM with CNN for Text Input | Combining LSTM and CNN for text processing | Enhanced understanding of text data | Overfitting observed, validation accuracy around 62.5% |
| 2     | Bidirectional LSTM with CNN for Text Input | Bidirectional LSTM captures bidirectional context | Improved context understanding, reduced overfitting | Validation accuracy around 67%, moderate performance |
| 3     | Bidirectional GRU for Text Classification | Utilizing Bidirectional GRU for context understanding | Better understanding of context compared to LSTM | Validation accuracy around 62%, moderate performance |
| 4     | CNN Model with Image Input | Simple CNN model for image processing | Moderate performance, potential overfitting | Training accuracy consistently higher than validation accuracy |
| 5     | CNN with Dropout (Image Input) | Dropout regularization added to CNN model | Reduced overfitting, improved generalization | Validation accuracy slightly improved to around 63-64% |
| 6     | Multi-modality Learning with Image and Text | Combining image and text data for learning | More information for predictions, potential overfitting | Validation accuracy around 61%, moderate performance |
| 7     | Multi-objective Learning for Price and Type Prediction | Predicting both price and type simultaneously | Improved overall performance, balanced learning | Some overfitting observed, higher accuracy for type prediction |
| 8     | Logistic Regression with RandomizedSearchCV | Utilizing RandomizedSearchCV for hyperparameter tuning to find optimal parameters. Including n-grams to capture more context from the text data. | Improved model performance by optimizing hyperparameters and considering different n-gram combinations. | RandomizedSearchCV helps in finding better hyperparameters compared to manual tuning. N-grams allow capturing multiple word sequences, potentially enhancing the model's understanding of the text. Validation Accuracy: 0.6712 |
| 9     | SVM with Randomized Search | Support Vector Machines with kernel trick can capture complex decision boundaries. Random search helps find optimal hyperparameters. | Expected improvement in performance with optimized hyperparameters. | Achieved higher validation accuracy compared to previous trials, suggesting better generalization. |
| 10    | Random Forest with Random Search | Random Forest is an ensemble method that can handle non-linear data and is robust to overfitting. Random search helps find optimal hyperparameters. | Expected improvement in overall accuracy and handling of complex decision boundaries. | Achieved moderate improvement in validation accuracy compared to previous trials, indicating better generalization. |
| 11    | XGBoost with Random Search | XGBoost is an efficient implementation of gradient boosting that can handle large datasets and non-linear data. Random search helps find optimal hyperparameters. | Expected improvement in overall accuracy and handling of complex decision boundaries. | Achieved moderate improvement in validation accuracy compared to previous trials, suggesting better generalization. |

Based on the observations and performance of each trail, the best trails are:
### Trails with Neural Networks:
**Trail 2: Bidirectional LSTM with CNN for Text Input:**
- **Reason:** Bidirectional LSTM captures both past and future context in text data, combined with CNN for image processing.
- **Expected Outcome:** Enhanced understanding of context in text data leading to better classification.
- **Observations:** Reduced overfitting compared to Trail 1, with similar validation accuracy.

**Trail 7: Multi-objective Learning for Price and Type Prediction:**
- **Reason:** Predicting both price and type simultaneously allows for balanced learning for both objectives.
- **Expected Outcome:** Improved overall performance by leveraging multi-task learning.
- **Observations:** Some degree of overfitting observed, especially for price prediction, but achieved higher accuracy for type prediction compared to price.

These trails showed promising results in terms of performance and addressing overfitting issues. However, further tuning and optimization may still be needed to achieve even better results.
### Other Best Trails:

2. **Trail 9: SVM with Randomized Search**
   - Validation Accuracy: 70.6%
   - Reason: SVM with Randomized Search achieved the highest validation accuracy.
   - Expected Outcome: Improved performance compared to previous trials.
   - Observations: Improved accuracy with optimized hyperparameters.

3. **Trail 11: XGBoost with Random Search**
   - Validation Accuracy: 69.3%
   - Reason: Utilizes XGBoost algorithm with optimized hyperparameters.
   - Expected Outcome: Moderate improvement in accuracy compared to other trials.
   - Observations: Achieved moderate performance with optimized parameters.

4. **Trail 8: Logistic Regression with RandomizedSearchCV**
   - Validation Accuracy: 67.1%
   - Reason: Utilized RandomizedSearchCV for hyperparameter tuning.
   - Expected Outcome: Improved model performance with n-grams.
   - Observations: Showed improved accuracy with optimized hyperparameters.
  
| Model                                      | acc Score | Best Parameters                                  | Observation                                                                                                                                                           |
|--------------------------------------------|-----------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Trail 1: LSTM with CNN for Text Input      | 0.625     | LSTM units: 64, Conv2D filters: 32, 64          | Overfitting observed, fluctuating validation accuracy.                                                                                                               |
| Trail 2: Bidirectional LSTM with CNN      | 0.625     | LSTM units: 64, Conv2D filters: 32, 64          | Reduced overfitting compared to Trail 1, similar validation accuracy.                                                                                                 |
| Trail 3: Bidirectional GRU                 | 0.62      | GRU units: 64                                    | Moderate performance, stable validation accuracy, overfitting not significant.                                                                                       |
| Trail 4: CNN Model with Image Input        | 0.61-0.62 | Conv2D filters: 32, 64, Dense units: 64         | Overfitting observed, slight improvement in validation accuracy with Dropout.                                                                                        |
| Trail 5: CNN with Dropout (Image Input)    | 0.63-0.64 | Conv2D filters: 32, 64, Dropout rate: 0.25      | Dropout helps reduce overfitting, improved validation accuracy.                                                                                                       |
| Trail 6: Multi-modality Learning           | 0.61      | LSTM units: 64, Conv2D filters: 32, 64          | Overfitting observed, moderate performance, no significant improvement.                                                                                               |
| Trail 7: Multi-objective Learning          | 0.63      | LSTM units: 64, Conv2D filters: 32, 64          | Some degree of overfitting observed, especially for price prediction, higher accuracy for type.                                                                     |
| Trail 8 : Logistic Regression with RandomizedSearchCV | 0.671 | max_features: 714, ngram_range: (1, 2), C: 0.1 | Improved model performance with n-grams, moderate accuracy |
| Trail 9 :SVM with Randomized Search | 0.706 | 'C': 3.845, 'gamma': 'scale' | Achieved higher validation accuracy compared to previous trials |
| Trail 10 :Random Forest with Random Search | 0.686 | n_estimators: 200, min_samples_split: 2, min_samples_leaf: 2, max_depth: None, bootstrap: False, ngram_range: (1, 1), max_features: 1000 | Achieved moderate improvement in validation accuracy |
| Trail 11 : XGBoost with Random Search | 0.693 | max_depth: 9, n_estimators: 200, learning_rate: 0.3 | Achieved moderate improvement in validation accuracy |

