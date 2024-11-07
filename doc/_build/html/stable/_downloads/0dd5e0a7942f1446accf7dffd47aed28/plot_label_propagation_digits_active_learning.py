"""
========================================
نشر التسمية للأرقام باستخدام التعلم النشط
========================================

يوضح هذا المثال تقنية التعلم النشط لتعلم التعرف على الأرقام المكتوبة بخط اليد
باستخدام نشر التسمية.

نبدأ بتدريب نموذج نشر التسمية باستخدام 10 نقاط فقط ذات تسميات،
ثم نقوم باختيار أكثر 5 نقاط غير مؤكدة لنقوم بتسميتها. بعد ذلك، نقوم بتدريب
النموذج باستخدام 15 نقطة ذات تسميات (10 نقاط أصلية + 5 نقاط جديدة). ونكرر هذه العملية
خمس مرات لنحصل على نموذج مدرب على 30 مثالًا ذا تسميات. يمكنك زيادة عدد التكرارات لتسمية أكثر من 30 مثالًا من خلال تغيير `max_iterations`. يمكن أن يكون تسمية أكثر من 30 مثالًا مفيدًا للحصول على فكرة عن سرعة تقارب
هذه التقنية للتعلم النشط.

سيظهر رسم بياني يوضح أكثر 5 أرقام غير مؤكدة في كل تكرار
للتدريب. قد تحتوي هذه الأمثلة على أخطاء أو لا، ولكننا سنقوم بتدريب النموذج التالي
باستخدام التسميات الصحيحة لها.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.semi_supervised import LabelSpreading

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 40
max_iterations = 5

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()

for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1

    lp_model = LabelSpreading(gamma=0.25, max_iter=20)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print(
        "Label Spreading model: %d labeled & %d unlabeled (%d total)"
        % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
    )

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # حساب أنتروبيا التوزيعات المسماة المتوقعة
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # اختيار 5 أمثلة للأرقام التي يكون النموذج أكثر عدم تأكد بشأنها
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.isin(uncertainty_index, unlabeled_indices)
    ][:5]

    # تتبع المؤشرات التي نحصل على التسميات لها
    delete_indices = np.array([], dtype=int)

    # للعدد الأكبر من 5 تكرارات، يتم تصور المكسب فقط على أول 5
    if i < 5:
        f.text(
            0.05,
            (1 - (i + 1) * 0.183),
            "model %d\n\nfit with\n%d labels" % ((i + 1), i * 5 + 10),
            size=10,
        )
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        # للعدد الأكبر من 5 تكرارات، يتم تصور المكسب فقط على أول 5
        if i < 5:
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation="none")
            sub.set_title(
                "predict: %i\ntrue: %i"
                % (lp_model.transduction_[image_index], y[image_index]),
                size=10,
            )
            sub.axis("off")

        # تسمية 5 نقاط، وإزالتها من مجموعة البيانات المسماة
        (delete_index,) = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)

f.suptitle(
    (
        "Active learning with Label Propagation.\nRows show 5 most "
        "uncertain labels to learn with the next model."
    ),
    y=1.15,
)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2, hspace=0.85)
plt.show()