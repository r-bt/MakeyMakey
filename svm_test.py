import numpy as np

positive_data = [
    "training/exp_1_always_on_bystander_behind.npy",
    "training/exp1_always_on_person_working_between_machine_bystander_behind.npy",
    "training/exp1_always_on_person_working_at_machine_bystander_behind.npy",
]
negative_data = [
    "training/exp_1_always_off_bystander_behind.npy",
    "training/exp1_always_off_person_working_between_machine_bystander_behind.npy",
    "training/exp1_always_off_person_working_at_machine_bystander_behind.npy",
]

# Load all the data into X and the labels into Y

X = []
Y = []

for data in positive_data:
    data = np.load(data, allow_pickle=True)
    for obj in data:
        X.append(obj[0])
        Y.append(1)

for data in negative_data:
    data = np.load(data, allow_pickle=True)
    for obj in data:
        X.append(obj[0])
        Y.append(0)

# Shuffle the data
from sklearn.utils import shuffle

X, Y = shuffle(X, Y)

# Divide into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the SVM
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), svm.SVC(kernel="linear"))
model.fit(X_train, Y_train)

# Test the SVM
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

# Test the SVM on the negative data
for name in negative_data:
    data = np.load(name, allow_pickle=True)

    data = data.reshape(data.shape[0], -1)

    pred = model.predict(data)

    print("{} accuracy: {}".format(name, 1 - (sum(pred) / len(pred))))

for name in positive_data:
    data = np.load(name, allow_pickle=True)

    data = data.reshape(data.shape[0], -1)

    pred = model.predict(data)

    print("{} accuracy: {}".format(name, sum(pred) / len(pred)))
