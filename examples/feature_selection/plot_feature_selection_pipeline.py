"""
==================
Pipeline ANOVA SVM
==================

يوضح هذا المثال كيف يمكن دمج اختيار الميزات بسهولة ضمن
مجرى تعلم الآلة.

نعرض أيضًا أنه يمكنك بسهولة فحص جزء من المجرى.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# سنبدأ بتوليد مجموعة بيانات تصنيف ثنائي. بعد ذلك، سنقوم
# بتقسيم مجموعة البيانات إلى مجموعتين فرعيتين.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
# خطأ شائع يرتكب مع اختيار الميزات هو البحث عن مجموعة فرعية من
# الميزات التمييزية على مجموعة البيانات الكاملة، بدلاً من استخدام
# مجموعة التدريب فقط. استخدام scikit-learn :func:`~sklearn.pipeline.Pipeline`
# يمنع ارتكاب مثل هذا الخطأ.
#
# هنا، سنوضح كيفية بناء مجرى حيث تكون الخطوة الأولى
# هي اختيار الميزات.
#
# عند استدعاء `fit` على بيانات التدريب، سيتم تحديد مجموعة فرعية من الميزات
# وسيتم تخزين فهرس هذه الميزات المحددة. سيقوم محدد الميزات
# لاحقًا بتقليل عدد الميزات، وتمرير هذه المجموعة الفرعية إلى
# المصنف الذي سيتم تدريبه.

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

anova_filter = SelectKBest(f_classif, k=3)
clf = LinearSVC()
anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X_train, y_train)

# %%
# بمجرد اكتمال التدريب، يمكننا التنبؤ بعينات جديدة غير مرئية. في هذه
# الحالة، سيقوم محدد الميزات بتحديد الميزات الأكثر تمييزًا فقط
# بناءً على المعلومات المخزنة أثناء التدريب. بعد ذلك، سيتم
# تمرير البيانات إلى المصنف الذي سيجري التنبؤ.
#
# هنا، نعرض المقاييس النهائية عبر تقرير تصنيف.

from sklearn.metrics import classification_report

y_pred = anova_svm.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
# كن على علم بأنه يمكنك فحص خطوة في المجرى. على سبيل المثال، قد
# نكون مهتمين بمعلمات المصنف. نظرًا لأننا اخترنا
# ثلاث ميزات، فإننا نتوقع أن يكون لدينا ثلاثة معاملات.

anova_svm[-1].coef_

# %%
# ومع ذلك، لا نعرف الميزات التي تم تحديدها من مجموعة البيانات الأصلية.
# يمكننا المتابعة بعدة طرق. هنا، سنقوم بعكس
# تحويل هذه المعاملات للحصول على معلومات حول المساحة الأصلية.

anova_svm[:-1].inverse_transform(anova_svm[-1].coef_)

# %%
# يمكننا أن نرى أن الميزات ذات المعاملات غير الصفرية هي الميزات المحددة
# بواسطة الخطوة الأولى.
