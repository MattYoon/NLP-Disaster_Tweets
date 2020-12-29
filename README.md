# NLP-Disaster_Tweets
I test out multiple NLP models, and their effectiveness against the Kaggle Disaster Tweets dataset.
- Kaggle Disaster Tweets: https://www.kaggle.com/c/nlp-getting-started/data?select=train.csv

### 1. LSTM without pre-trained weights
- Trained completely from scratch, using Tensorflow Embedding Layers and LSTM layers.
- accuracy score: 0.79282

### 2. LSTM with pre-trained GLoVe Embeddings
- LSTM trained from scratch, pretrained GLoVe : https://nlp.stanford.edu/projects/glove/
- accuracy score: 0.81366
