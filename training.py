import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def is_valid_sample(sample):
    try:
        x = np.asarray(sample)
        return True
    except ValueError:
        return False


data_dict = pickle.load(open('./data.pickle', 'rb'))
data1 = data_dict['data']
labels = data_dict['labels']


# Check the number of data samples
num_samples = len(data1)
print("Number of data samples:", num_samples)

# Check the shape of each data sample


# Define a function to check if a data sample is valid

# Filter out the invalid data samples

for sample in data1:
    print("Sample shape:", is_valid_sample(sample))
index = 0
for sample in labels:
    print(sample)
    index = index + 1
    print(index)

data_filtered = []
labels_filtered = []

for sample, label in zip(data1, labels):
    try:
        x = np.asarray(sample)
        data_filtered.append(x)
        labels_filtered.append(label)
    except ValueError:
        continue

data = np.asarray(data_filtered)
labels = np.asarray(labels_filtered)


data_filtered = [sample for sample in data1 if is_valid_sample(sample)]
labels_filtered = [label for i, label in enumerate(labels) if is_valid_sample(data1[i])]

# Convert the filtered data and labels to NumPy arrays
data = np.asarray(data_filtered)
labels_filtered = np.asarray(labels_filtered)



labels = np.asarray(data_dict['labels'])

x_train,x_test,y_train,y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels) 
#what is this? 
#we are splitting all the data into a train set and test set, to create x train to x test
# we are splitting all the labels into a train set and test set, to create y train to y test
#we are taking all the info from the data and label, into 2 different sets.
#Test size 0.2 means we are keeping 20% of data as test sets
#ALWAYS shuffle data when training classifier
#It essentially means we shuffle the data to remove biases
#Stratify labels, meaning that we are keeping the same proportion of the labels in train set & test

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print(str(score*100) + "% were correctly predicted!")
#print(y_predict)
f = open('model.p','wb')
pickle.dump({'model':model},f)
f.close()