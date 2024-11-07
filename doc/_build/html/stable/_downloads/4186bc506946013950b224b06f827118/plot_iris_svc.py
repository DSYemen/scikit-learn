"""
==================================================
رسم مخططات لمصنفات SVM المختلفة في مجموعة بيانات الزهرة
==================================================

مقارنة بين مصنفات SVM الخطية المختلفة على إسقاط ثنائي الأبعاد لمجموعة بيانات الزهرة. نأخذ في الاعتبار فقط أول ميزتين لهذه المجموعة:

- طول السبلة
- عرض السبلة

يوضح هذا المثال كيفية رسم سطح القرار لأربعة مصنفات SVM
باستخدام نوى مختلفة.

تنتج النماذج الخطية "LinearSVC()" و "SVC(kernel='linear')" حدود قرار مختلفة قليلاً. قد يكون ذلك نتيجة للاختلافات التالية:

- يقلل "LinearSVC" من خسارة المفصلة المربعة بينما يقلل "SVC" من خسارة المفصلة العادية.

- يستخدم "LinearSVC" التخفيض متعدد الفئات One-vs-All (المعروف أيضًا باسم One-vs-Rest) بينما يستخدم "SVC" التخفيض متعدد الفئات One-vs-One.

لدى كلا النموذجين الخطيين حدود قرار خطية (مستويّات متقاطعة)
في حين أن النماذج غير الخطية (البولينومية أو Gaussian RBF) لها حدود قرار غير خطية أكثر مرونة بأشكال تعتمد على نوع النواة ومعاملاتها.

.. NOTE:: أثناء رسم دالة القرار للمصنفات لمجموعات البيانات ثنائية الأبعاد، يمكن أن يساعد ذلك في الحصول على فهم حدسي لقوتها التعبيرية، ولكن كن على دراية بأن هذه الحدوس لا تعمم دائمًا على المشكلات الواقعية عالية الأبعاد.

"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# استيراد بعض البيانات للتجربة
iris = datasets.load_iris()
# أخذ أول ميزتين. يمكننا تجنب ذلك باستخدام مجموعة بيانات ثنائية الأبعاد
X = iris.data[:, :2]
y = iris.target

# ننشئ مثالاً من SVM ونناسب البيانات. لا نقوم بضبط بياناتنا
# لأننا نريد رسم المتجهات الداعمة
C = 1.0  # معامل ضبط SVM
models = (
    svm.SVC(kernel="linear", C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)

# عنوان للمخططات
titles = (
    "SVC مع نواة خطية",
    "LinearSVC (نواة خطية)",
    "SVC مع نواة RBF",
    "SVC مع نواة بولينومية (درجة 3)",
)

# إعداد شبكة 2x2 للرسم
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()