import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score,confusion_matrix

# Parameters
n_feat = 200  # number of features
epochs = 4
batch_size = 100

# Read train data
with open('train.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = []
    for row in reader:
      data.append(row)
train_data = pd.DataFrame(data).sample(frac = 1)   #create the dataframe and shuffle it

# Read valid data
with open('valid.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = []
    for row in reader:
      data.append(row)
valid_data = pd.DataFrame(data).sample(frac = 1)

# Read test data
with open('test.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = []
    for row in reader:
      data.append(row)
test_data = pd.DataFrame(data).sample(frac = 1)

print(train_data.head())
print(train_data.columns.values)

# Get most important words
def get_important_words(texts,n_feat=200):
    #output: a list of the N most important words from a list of sentences

    #fit the n-gram model
    vectorizer = TfidfVectorizer(analyzer='word',
                            max_features=n_feat)

    X = vectorizer.fit_transform(texts)

    #Get model feature names
    feature_names = vectorizer.get_feature_names_out()

    return feature_names

# Obtain moset important words from each language
features = {}
features_set = set()

languages = train_data['labels'].unique()
print(len(languages))

for l in languages:

    #get texts for each language language
    texts = train_data[train_data.labels==l]['text']

    #get 200 most important words
    important_words = get_important_words(texts, n_feat)

    #add to dict and set
    features[l] = important_words
    features_set.update(important_words) #the set of important words across all languages


#create vocabulary list using feature set of important words
vocab_dic = dict()
for i,word in enumerate(features_set):
    vocab_dic[word]=i

# Create feaures for each train sapmle using vocabulary
vectorizer = TfidfVectorizer(analyzer='word',
                            max_features= n_feat,
                            vocabulary=vocab_dic)

# Create featur matrrix for train_texts
texts = train_data['text']
X = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

train_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)

#Scale train feature matrix
train_min = train_feat.min()
train_max = train_feat.max()
train_feat = (train_feat - train_min)/(train_max-train_min)

# Add target variable to train_feature matrix
train_feat['labels'] = list(train_data['labels'])

# Create feature matrix for validation_texts
texts = valid_data['text']
X = vectorizer.fit_transform(texts)

valid_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)
valid_feat = (valid_feat - train_min)/(train_max-train_min)
valid_feat['labels'] = list(valid_data['labels'])

# Create feature matrix for test_texts
texts = test_data['text']
X = vectorizer.fit_transform(texts)

test_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)
test_feat = (test_feat - train_min)/(train_max-train_min)
test_feat['labels'] = list(test_data['labels'])

#num_lists = len(lists_of_words)
common_word_matrix = [[0] * len(languages) for _ in range(len(languages))]

# Convert each list of lang-words into a set
sets_of_words = [set(features[l]) for l in languages]

# Calculate the common words and fill in the matrix
for i in range(len(languages)):
    for j in range(0 , i+1):  # Avoid comparing the same list to itself and avoid duplicate comparisons
        common_words = sets_of_words[i].intersection(sets_of_words[j])
        common_word_count = len(common_words)
        common_word_matrix[i][j] = common_word_count
        #common_word_matrix[j][i] = common_word_count  # Since it's symmetric

# The common_word_matrix now contains the counts of common words between each pair of lists

fig, ax = plt.subplots(figsize=(len(languages), len(languages)))

# Create a heatmap of the common word matrix
cax = ax.matshow(common_word_matrix, cmap="YlGnBu")

# Set the labels for the x and y axes
ax.set_xticks(np.arange(len(languages)))
ax.set_yticks(np.arange(len(languages)))
ax.set_xticklabels(languages)
ax.set_yticklabels(languages)
ax.xaxis.set_ticks_position("bottom")

# Add count annotations to the cells
for i in range(len(languages)):
    for j in range(0 , i+1):
        count = common_word_matrix[i][j]
        ax.text(j, i, str(count), ha="center", va="center", color="black")


# Add a title
plt.title("Common Words Between Languages")

# Display the plot
plt.show()

#Fit encoder
encoder = LabelEncoder()
encoder.fit(languages)

#converts the labels to one hot encodings
def encode(y):
    y_encoded = encoder.transform(y)
    y_dummy = utils.to_categorical(y_encoded)

    return y_dummy

print(len(features_set))

# Training data
x = train_feat.drop('labels',axis=1)
y = encode(train_feat['labels'])

x_val = valid_feat.drop('labels',axis=1)
y_val = encode(valid_feat['labels'])

# Define model
model = Sequential()
model.add(Dense(2000, input_dim=len(features_set), activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(len(languages), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
model.fit(x, y, epochs=4, batch_size=100, validation_data=(x_val, y_val))

x_test = test_feat.drop('labels', axis=1)
y_test = test_feat['labels']

#Get predictions on test set
predict_x = model.predict(x_test)
predict_labels = np.argmax(predict_x, axis=1)

#labels = model.predict_classes(x_test)
predictions = encoder.inverse_transform(predict_labels)

#Accuracy on test set
accuracy = accuracy_score(y_test, predictions)
print("Accuracy =", accuracy)

#Create confusion matrix
conf_matrix = confusion_matrix(y_test, predictions, labels=languages)
conf_matrix_df = pd.DataFrame(conf_matrix, columns=languages ,index=languages)

#Plot confusion matrix heatmap
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.heatmap(conf_matrix_df,cmap='coolwarm', annot=True, fmt='.5g', cbar=False)
plt.xlabel('Predicted',fontsize=22)
plt.ylabel('Actual',fontsize=22)

new_text = ["Αυτό θα συμβεί γιατί η αίτηση που έχουν καταθέσει, είτε με τη γνωστή διαδικασία έως τις 25 Σεπτεμβρίου αν είναι νέοι δικαιούχους, είτε αυτόματα από το σύστημα για τους παλιούς, θα παραμένει ακόμα υπό επεξεργασία. Η ΑΑΔΕ προχωρά στη διασταύρωση στοιχείων με βάση τις τελευταίες φορολογικές δηλώσεις. Αν η φορολογική δήλωση δεν έχει εκκαθαριστεί έως τις 25 Σεπτεμβρίου, δεν προχωράει και η διαδικασία πληρωμής για το Market Pass."]
# Create feature matrix for test_texts
X = vectorizer.fit_transform(new_text)
new_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)
#new_feat = (new_feat - train_min)/(train_max-train_min)
#Get prediction for a text
predict_x = model.predict(new_feat)
predict_labels = np.argmax(predict_x, axis=1)
prediction = encoder.inverse_transform(predict_labels)
print(prediction)

new_text = ["女性は頭や顔などに大けがをして病院に搬送されましたが、命に別状はないということです。体長およそ1メートルと体長およそ50センチのあわせて2頭のクマがいたということです。"]
#true label for new tex is 'ja' (japanese)

# Create feature matrix for test_texts
X = vectorizer.fit_transform(new_text)
new_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)
#new_feat = (new_feat - train_min)/(train_max-train_min)
#Get prediction for a text
predict_new = model.predict(new_feat)
predict_labels = np.argmax(predict_new, axis=1)
prediction = encoder.inverse_transform(predict_labels)
print(prediction)

