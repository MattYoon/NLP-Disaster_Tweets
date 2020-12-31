# NLP-Disaster_Tweets
I test out multiple NLP models, and their effectiveness against the Kaggle Disaster Tweets dataset.
- Kaggle Disaster Tweets: https://www.kaggle.com/c/nlp-getting-started/data?select=train.csv

### 1. LSTM without pre-trained weights
- LSTM and embeddings trained completely from scratch
- accuracy score: 0.79282

### 2. LSTM with pre-trained GLoVe Embeddings
- LSTM trained from scratch, pre-trained GLoVe embeddings 
- GLoVe: https://nlp.stanford.edu/projects/glove/
- accuracy score: 0.81366

### 3. FFNN with pre-trained ELMo Embeddings
- FFNN with one hidden layer, pre-trained ELMo embeddings
- ELMo: http://vectors.nlpl.eu/repository/
- accuracy score: 0.81703

### 4. CNN with pre-trained ELMo Embeddings
- CNN using three different filter sizes and average pooling, pre-trained ELMo embeddings
- ELMo: http://vectors.nlpl.eu/repository/
- accuracy score: 0.82531

### 5. Multi-Head Self-Attention without pre-trained weights
- Single Encoder layer trained completely from scratch
- Reference: https://blog.tensorflow.org/2019/05/transformer-chatbot-tutorial-with-tensorflow-2.html
- accuracy score: 0.78639
