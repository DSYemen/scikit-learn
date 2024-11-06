"""
================================
تمرين تصنيف الأرقام
================================

تمرين تعليمي يتعلق باستخدام تقنيات التصنيف على
مجموعة بيانات الأرقام.

يتم استخدام هذا التمرين في جزء :ref:`clf_tut` من
قسم :ref:`supervised_learning_tut` من
:ref:`stat_learn_tut_index`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn import datasets, linear_model, neighbors

X_digits, y_digits = datasets.load_digits(return_X_y=True)
X_digits = X_digits / X_digits.max()

n_samples = len(X_digits)

X_train = X_digits[: int(0.9 * n_samples)]
y_train = y_digits[: int(0.9 * n_samples)]
X_test = X_digits[int(0.9 * n_samples) :]
y_test = y_digits[int(0.9 * n_samples) :]

knn = neighbors.KNeighborsClassifier()
logistic = linear_model.LogisticRegression(max_iter=1000)

print("نتيجة KNN: %f" % knn.fit(X_train, y_train).score(X_test, y_test))
print(
    "نتيجة LogisticRegression: %f"
    % logistic.fit(X_train, y_train).score(X_test, y_test)
)


