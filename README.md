# Identifying-the-Language-of-a-Document
Naive Baysian and 5_layers_Neural Network models were applied on the balanced Papluca dataset to train the model for identifying the language of a text. 

Models
-
This project is aimed at training models to identify the langage of a given text. The repository includes three different models including Naive Baysian and multi_layers_Neural Network (character and word based).

For all models TfidfVectorizer() used to get the most important features within the text. For Naive Basian TfidfVectorizer() is studied on char-based features, while for multi-layer Neural Network both char and word -based features have been investigated.

Dataset
-
The Papluca dataset which is available at https://huggingface.co/datasets/papluca/language-identification. The datset consists of a balaced datasets for 20 languages.

