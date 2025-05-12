import numpy as np

positive_data = [
    "training/exp2_always_on_persons_on_other_tables_2m.npy",
    "training/exp2_always_on_persons_on_other_tables_4m.npy",
    "training/exp2_always_on_person_working_at_machine_persons_on_other_tables_2m.npy",
    "training/exp2_both_machines_on_person_working_at_machine_persons_on_other_tables.npy",
]
negative_data = [
    "training/exp2_always_off_persons_on_other_tables_2m_4m.npy",
    "training/exp2_always_off_person_working_at_machine_persons_on_other_tables.npy",
]

# Load all the data into X and the labels into Y

X = []
Y = []

for data in positive_data:
    data = np.load(data, allow_pickle=True)
    for obj in data:
        X.append(obj)
        Y.append(1)

for data in negative_data:
    data = np.load(data, allow_pickle=True)
    for obj in data:
        X.append(obj)
        Y.append(0)

# # Shuffle the data
from sklearn.utils import shuffle

X, Y = shuffle(X, Y)

# # Divide into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the SVM
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), svm.SVC(kernel="linear"))
model.fit(X_train, Y_train)

# # Test the SVM
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

# Save the model
from joblib import dump, load

# Save the model
dump(model, "svm_model.joblib")

# Load the model later
model = load("svm_model.joblib")
predictions = model.predict(X_test)
