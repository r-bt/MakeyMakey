positive_data = [
    "training/exp2_always_on_persons_on_other_tables_2m.npy",
    "training/exp2_always_on_persons_on_other_tables_4m.npy",
    "training/exp2_always_on_person_working_at_machine_persons_on_other_tables_2m.npy",
    "training/exp2_both_machines_on_person_working_at_machine_persons_on_other_tables.npy",
]
negative_data = [
    "training/exp2_always_off_persons_on_other_tables_2m_4m.npy",
    "training/exp2_always_off_person_working_at_machine_persons_on_other_tables.npy",
    "training/exp2_lots_of_people_no_machines.npy",
]

pretty_names = {
    "exp2_always_on_persons_on_other_tables_2m": "Always on, 2m",
    "exp2_always_on_persons_on_other_tables_4m": "Always on, 4m",
    "exp2_always_off_persons_on_other_tables_2m_4m": "Always off, 2m + 4m",
    "exp2_always_off_person_working_at_machine_persons_on_other_tables": "Always off, Win Win at machine",
    "exp2_always_on_person_working_at_machine_persons_on_other_tables_2m": "Always on, 2m, Win Win at machine",
    "exp2_both_machines_on_person_working_at_machine_persons_on_other_tables": "Both machines on, Win Win at machine",
}

# Import the model
import joblib
from sklearn.metrics import classification_report, confusion_matrix

model = joblib.load("svm_model.joblib")
[
    X_train,
    Y_train,
    X_test,
    Y_test,
] = joblib.load("svm_data.joblib")

# # Test the SVM
Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))

# Show the accuracies across the scenarios
import numpy as np
import matplotlib.pyplot as plt

file_names, accuracies, errors = [], [], []

# for file in positive_data + negative_data:
#     data = np.load(file, allow_pickle=True)
#     n = len(data)
#     true_label = 1 if file in positive_data else 0
#     labels = np.full(n, true_label)

#     preds = model.predict(data)
#     p = np.mean(preds == labels)
#     se = np.sqrt(p * (1 - p) / n)

#     short = file.split("/")[-1].replace(".npy", "")
#     file_names.append(short)
#     accuracies.append(p)
#     errors.append(se)

subset = [
    "training/exp2_always_on_person_working_at_machine_persons_on_other_tables_2m.npy",
    "training/exp2_both_machines_on_person_working_at_machine_persons_on_other_tables.npy",
    "training/exp2_always_off_person_working_at_machine_persons_on_other_tables.npy",
]

for file in subset:
    data = np.load(file, allow_pickle=True)
    n = len(data)
    true_label = 1 if file in positive_data else 0
    labels = np.full(n, true_label)

    preds = model.predict(data)
    p = np.mean(preds == labels)
    se = np.sqrt(p * (1 - p) / n)

    short = file.split("/")[-1].replace(".npy", "")
    file_names.append(short)
    accuracies.append(p)
    errors.append(se)

names = [pretty_names.get(name, name) for name in file_names]

plt.figure(figsize=(12, 6))
bars = plt.bar(names, accuracies, yerr=errors, capsize=5)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Per-Scenario Accuracy with 1-SE Error Bars")

for bar, p in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        p + 0.02,
        f"{p:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()
