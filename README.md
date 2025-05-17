# Sentiment-Analysis-with-BiLSTM-CNN-Model
This project performs multiclass sentiment analysis on user-generated comments using a hybrid deep learning model combining Bidirectional LSTM and CNN layers. The dataset is sourced from Kaggle and includes labeled comments categorized into three sentiment classes: Positive, Neutral, and Negative.

🔍 Overview
This pipeline aims to extract emotional context from text data and classify it accurately using a deep learning-based model architecture. The project includes:
1. Dataset download and preprocessing
2. Tokenization and padding
3. Model building with BiLSTM and Conv1D
4. Evaluation with metrics and visualization

🗂 Dataset
Source: KaggleHub
Format: CSV
Features:
-Comment: The text input
-Sentiment: Integer class (0 = Negative, 1 = Neutral, 2 = Positive)

🧰 Libraries Used
python
Copy
Edit
pandas, numpy, seaborn, matplotlib
sklearn 
tensorflow.keras 
kagglehub 

🧪 Model Architecture
The model uses the following architecture:

Embedding Layer
↓
Bidirectional LSTM (64 units, return_sequences=True)
↓
Conv1D Layer (64 filters, kernel size=5, ReLU)
↓
Global Max Pooling
↓
Dropout (0.5)
↓
Dense (64 units, ReLU)
↓
Dense (3 units, softmax)

🛠️ Steps Followed
1. Data Loading
-Downloaded directly using kagglehub and filtered for relevant .csv files.
2. Preprocessing
-Cleaned and dropped nulls
-Converted comments to strings
-One-hot encoded the sentiment labels
-Tokenized and padded sequences to a fixed max_len
3. Model Building
-Compiled using Adam optimizer and categorical crossentropy loss.
4. Training
-Trained for 5 epochs with a batch size of 256, using a 10% validation split.
5. Evaluation
-Classification Report
-Confusion Matrix
-Accuracy & Loss plots

📊 Results
Metrics Used: Accuracy, Precision, Recall, F1-Score
Tools: sklearn.metrics, seaborn, matplotlib
Observations: The BiLSTM-CNN combination captured both temporal and spatial features effectively, yielding high accuracy on the validation set.
