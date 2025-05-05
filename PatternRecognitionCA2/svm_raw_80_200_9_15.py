import images_prep
import pca_from_scratch
import lda_from_scratch
import numpy as np
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# ======= Data preparation =======
random.seed(71)
selected_indices = random.sample(range(1, 69), 25)
selected_indices.append(69)

PIE_path = "./PIE"
training_images, training_labels, test_images, test_labels = images_prep.get_images(PIE_path, selected_indices)

X_train_raw = np.array([img.flatten() for img in training_images])
X_test_raw = np.array([img.flatten() for img in test_images])
Y_train = np.array(training_labels)
Y_test = np.array(test_labels)

mean_80, proj_80, X_train_pca_80 = pca_from_scratch.mannual_pca(X_train_raw, 80)
mean_200, proj_200, X_train_pca_200 = pca_from_scratch.mannual_pca(X_train_raw, 200)
X_test_pca_80 = (X_test_raw - mean_80) @ proj_80
X_test_pca_200 = (X_test_raw - mean_200) @ proj_200

proj_lda_9, X_train_lda_9 = lda_from_scratch.mannual_lda(X_train_raw, Y_train, 9)
proj_lda_15, X_train_lda_15 = lda_from_scratch.mannual_lda(X_train_raw, Y_train, 15)
X_test_lda_9 = X_test_raw @ proj_lda_9
X_test_lda_15 = X_test_raw @ proj_lda_15

# ======= Define a general SVM training evaluation function =======
def train_and_evaluate_svm(X_train, Y_train, X_test, Y_test, C_value):
    ovr_svm = OneVsRestClassifier(svm.SVC(kernel='linear', C=C_value, random_state=71))
    ovr_svm.fit(X_train, Y_train)
    y_train_pred = ovr_svm.predict(X_train)
    y_test_pred = ovr_svm.predict(X_test)
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    test_accuracy = accuracy_score(Y_test, y_test_pred)
    return train_accuracy, test_accuracy

# ======= Evaluating SVM classification performance under different dimensionality reduction methods =======
C_values = [1e-2, 1e-1, 1]

print("====== Raw Features ======")
for C in C_values:
    train_acc, test_acc = train_and_evaluate_svm(X_train_raw, Y_train, X_test_raw, Y_test, C)
    print(f"[Raw] C={C} | Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

print("\n====== PCA Features ======")
for C in C_values:
    train_acc_80, test_acc_80 = train_and_evaluate_svm(X_train_pca_80, Y_train, X_test_pca_80, Y_test, C)
    print(f"[PCA-80] C={C} | Train Acc: {train_acc_80:.2%} | Test Acc: {test_acc_80:.2%}")
    
    train_acc_200, test_acc_200 = train_and_evaluate_svm(X_train_pca_200, Y_train, X_test_pca_200, Y_test, C)
    print(f"[PCA-200] C={C} | Train Acc: {train_acc_200:.2%} | Test Acc: {test_acc_200:.2%}")

print("\n====== LDA Features ======")
for C in C_values:
    train_acc_9, test_acc_9 = train_and_evaluate_svm(X_train_lda_9, Y_train, X_test_lda_9, Y_test, C)
    print(f"[LDA-9] C={C} | Train Acc: {train_acc_9:.2%} | Test Acc: {test_acc_9:.2%}")
    
    train_acc_15, test_acc_15 = train_and_evaluate_svm(X_train_lda_15, Y_train, X_test_lda_15, Y_test, C)
    print(f"[LDA-15] C={C} | Train Acc: {train_acc_15:.2%} | Test Acc: {test_acc_15:.2%}")
