"""
==================
مصفوفة الارتباك
==================

مثال على استخدام مصفوفة الارتباك لتقييم جودة
مخرجات مصنف على مجموعة بيانات الزهرة Iris. تمثل العناصر القطرية
عدد النقاط التي يكون فيها التصنيف المتوقع مساويًا للتصنيف الحقيقي،
بينما العناصر خارج القطرية هي تلك التي يخطئ المصنف في تصنيفها.
كلما كانت قيم القطرية لمصفوفة الارتباك أعلى، كان ذلك أفضل، مما يشير إلى
الكثير من التوقعات الصحيحة.

توضح الأشكال مصفوفة الارتباك مع وبدون
التطبيع حسب حجم دعم الفئة (عدد العناصر
في كل فئة). يمكن أن يكون هذا النوع من التطبيع
مثيرًا للاهتمام في حالة عدم توازن الفئات للحصول على تفسير مرئي أكثر
للفئة التي يتم تصنيفها بشكل خاطئ.

هنا النتائج ليست جيدة كما يمكن أن تكون لأن
اختيارنا لمعامل الانتظام C لم يكن الأفضل.
في التطبيقات الواقعية، عادة ما يتم اختيار هذا المعامل
باستخدام البحث الشبكي.
"""
# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# استيراد بعض البيانات للتجربة
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# تشغيل المصنف، باستخدام نموذج مفرط في الانتظام (C منخفض جدًا) لمشاهدة
# التأثير على النتائج
classifier = svm.SVC(kernel="linear", C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# رسم مصفوفة الارتباك غير المطبعنة
titles_options = [
    ("مصفوفة الارتباك، بدون تطبيع", None),
    ("مصفوفة الارتباك المطبعنة", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()