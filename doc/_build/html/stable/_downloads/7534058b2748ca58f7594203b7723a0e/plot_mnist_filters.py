"""
=====================================
تصور أوزان الشبكة العصبية متعددة الطبقات على مجموعة بيانات MNIST
=====================================

في بعض الأحيان، يمكن أن يوفر فحص المعاملات المُدربة لشبكة عصبية نظرة ثاقبة لسلوك التعلم. على سبيل المثال، إذا بدت الأوزان غير منظمة، فقد يكون بعضها غير مستخدم على الإطلاق، أو إذا كانت هناك معاملات كبيرة جدًا، فقد يكون معدل التعلم مرتفعًا جدًا.

يوضح هذا المثال كيفية رسم بعض أوزان الطبقة الأولى في نموذج MLPClassifier المدرب على مجموعة بيانات MNIST.

تتكون بيانات الإدخال من أرقام مكتوبة بخط اليد بحجم 28x28 بكسل، مما يؤدي إلى 784 خاصية في مجموعة البيانات. وبالتالي، فإن مصفوفة أوزان الطبقة الأولى لها الشكل (784، hidden_layer_sizes[0]). يمكننا بالتالي تصور عمود واحد من مصفوفة الأوزان كصورة 28x28 بكسل.

لجعل المثال يعمل بشكل أسرع، نستخدم عددًا قليلًا جدًا من الوحدات المخفية، ونقوم بالتدريب لفترة قصيرة جدًا. سيؤدي التدريب لفترة أطول إلى الحصول على أوزان ذات مظهر مكاني أكثر سلاسة. سيقوم المثال بإظهار تحذير لأنه لا يتقارب، وفي هذه الحالة، هذا ما نريده بسبب قيود استخدام الموارد على بنيتنا التحتية للتكامل المستمر التي تُستخدم لبناء هذه الوثائق بشكل منتظم.
"""

# المؤلفون: مطوري مكتبة ساي كيت ليرن
# معرف الترخيص: BSD-3-Clause

import warnings

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# تحميل البيانات من https://www.openml.org/d/554
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0

# تقسيم البيانات إلى قسم التدريب وقسم الاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)

mlp = MLPClassifier(
    hidden_layer_sizes=(40,),
    max_iter=8,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.2,
)

# هذا المثال لن يتقارب بسبب قيود استخدام الموارد على
# بنيتنا التحتية للتكامل المستمر، لذلك نلتقط التحذير و
# نتجاهله هنا
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X_train, y_train)

print("درجة مجموعة التدريب: %f" % mlp.score(X_train, y_train))
print("درجة مجموعة الاختبار: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# استخدام الحد الأدنى/الحد الأقصى العالمي لضمان عرض جميع الأوزان على نفس المقياس
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()