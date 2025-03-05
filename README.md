Industrial machines generate vast amounts of sensor data, including temperature, pressure, and vibration readings. Unexpected failures can lead to significant downtime and costly repairs, making predictive maintenance a critical challenge. Our solution leverages machine learning and deep learning techniques to analyze sensor data, detect anomalies, and predict failures before they occur. By implementing supervised and unsupervised models, we develop an AI-driven system that enhances maintenance planning and reduces operational disruptions. This presentation will cover our approach, including the models used, data preprocessing steps, visualization techniques, and an interactive dashboard for real-time monitoring..

Data sets used:
1)FD001 (C-MAPSS Dataset - Jet Engine RUL Prediction)
A subset of the C-MAPSS dataset used to predict Remaining Useful Life (RUL) of jet engines.
Contains 100 training engines and 100 test engines, all operating under a single condition (Sea Level) with one fault mode (HPC Degradation).
HPC Degradation: The High-Pressure Compressor (HPC) loses efficiency over time, leading to higher temperatures, reduced pressure ratios, and increased fuel consumption.

2)MVTec AD - Metal Nut Anomaly Detection
A real-world dataset for visual anomaly detection in metal nuts.
Includes defective and normal samples, used to train models for detecting irregularities in manufacturing.
This dataset helps in automating quality inspection by enabling machine learning models to identify defects with high precision, reducing manual inspection efforts and improving production efficiency.

3)Predictive Maintenance Dataset (AI4I 2020) 
10,000 data points with 14 features (temperature, speed, torque, tool wear).
5 Failure Modes: Tool wear, heat dissipation, power, overstrain, and random failures.
Used to train predictive maintenance models for failure detection & anomaly detection.
Helps in preventive maintenance by predicting failures before they occur, reducing downtime. 


The nasa folder contains anomaly detection on FD001
In our predictive maintenance system, we tested multiple models to find the best balance between accuracy, training speed, and real-time feasibility.
We compared traditional ML models (KNN, SVR, Random Forest) and deep learning (LSTM) to predict Remaining Useful Life (RUL) of machines based on sensor data.

Best Model: Random Forest (best accuracy-speed balance).
Key Insights:
RUL predictions improve as failure nears (<50 cycles).
Noisy sensor data enhances accuracy (smoothing reduces it).
Final cycles may hold crucial missed information.
KNN & RF are robust; LSTM needs fine-tuning & more data.
Our model successfully predicts engine failures before they happen.
 Random Forest is the best ML model for this dataset.
 LSTM is promising for long-term predictions but needs optimization.
 Deploying these models in a real-time dashboard can improve maintenance planning.

R² :  Measures how well the model explains variance in the data.
RMSE: Measures the average error between predicted and actual values.
Training Time: Measures how long it takes to train the model.
Prediction Time: Measures how long the model takes to make a single prediction.

Anomaly detection on MVtec ad
Detect anomalies in metal nut images using a deep learning-based autoencoder.
Trained a Convolutional Autoencoder on normal images.
Augmented data with cropping, flipping, and rotations to improve robustness.
The model learns normal patterns but fails to reconstruct anomalies.
Higher reconstruction error indicates a potential defect.
SSIM loss helps retain structural features better than MSE alone.
Successfully detects anomalous metal nuts.
 Autoencoders effectively differentiate normal vs. defective samples.
Can be deployed for automated defect detection in manufacturing.

Anomaly detection on Ai4I
Supervised Approach:
 Used Random Forest & XGBoost, with XGBoost performing best.
XGBoost Results:
Overall Accuracy: 98%
Anomaly Detection (Class 1): Precision: 63%, Recall: 78%, F1-score: 70%
Confusion Matrix: Detected 53 anomalies, with 15 false negatives.
Unsupervised Approach:
Used for detecting anomalies without labeled data.
Implemented Isolation Forest for anomaly detection.
Findings: XGBoost effectively classifies anomalies, while Isolation Forest helps detect outliers without labels.
