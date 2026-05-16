# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Task 1: Dataset Understanding

The customer churn dataset was used for this part. It contains 2,000 rows and 17 columns, where each row represents one customer.

The target variable was identified as `churn`. A value of `0` represents a retained customer, while a value of `1` represents a churned customer.

No missing values were found in the dataset. The target variable was highly imbalanced, with 1,969 retained customers and 31 churned customers.


## Task 2: Data Preprocessing

The data was prepared before training the neural network. The `customer_id` column was removed because it was only an identifier, and the target variable `churn` was separated from the input features.

Categorical columns were encoded using one-hot encoding, and numerical columns were scaled using standard scaling. The dataset was split into training and testing sets using an 80:20 split.

After preprocessing, the training set contained 1,600 records, the testing set contained 400 records, and both sets had 28 input features.


## Task 3: Neural Network Model Building

A feed-forward neural network was built using TensorFlow/Keras. The model used 28 input features from the preprocessed dataset.

Two hidden layers were added with 32 and 16 neurons, and ReLU activation was used in both layers. The output layer used one neuron with sigmoid activation because the target variable was binary.

Binary crossentropy was used as the loss function, and Adam was used as the optimizer. The model had 1,473 trainable parameters.


## Task 4: Training and Evaluation

The model was trained for 20 epochs with a batch size of 32. The final training accuracy was 98.44%, and the testing accuracy was 98.50%.

The confusion matrix showed that 394 non-churn customers were correctly classified, but all 6 churn customers were missed.

Although the accuracy was high, the model was biased toward the majority class. This shows that accuracy alone was not enough for evaluation.


## Task 5: Hyperparameter Experimentation

Three model configurations were tested by changing the number of neurons, learning rate, batch size, and activation function.

Model 1 used 32 and 16 neurons with ReLU activation and gave the best testing accuracy of 98.75%. Model 2 used more neurons, but the testing accuracy slightly decreased to 98.50%. Model 3 used a lower learning rate, larger batch size, and tanh activation, with a testing accuracy of 98.50%.

Overall, the first model performed slightly better. However, the results were still affected by the class imbalance in the dataset.

| Model | Hidden Units | Second Layer Units | Learning Rate | Batch Size | Activation | Test Loss | Test Accuracy |
|---|---:|---:|---:|---:|---|---:|---:|
| Model 1 | 32 | 16 | 0.0010 | 32 | ReLU | 0.0601 | 98.75% |
| Model 2 | 64 | 32 | 0.0010 | 32 | ReLU | 0.0631 | 98.50% |
| Model 3 | 32 | 16 | 0.0005 | 64 | tanh | 0.0688 | 98.50% |


## Task 6: Final Reflection

Weights and biases were used by the model to learn patterns between the input features and the churn target. During training, these values were updated to reduce prediction error.

Activation functions were required to help the neural network learn non-linear relationships. Without activation functions, the model would behave like a simple linear model.

If the learning rate is too high, the model may learn too fast and miss the best solution. If it is too low, training may become slow and may not reach a good result.

The model did not show strong overfitting because the training and testing accuracy were close. However, it showed a limitation because it failed to correctly identify churn customers due to the highly imbalanced dataset.
