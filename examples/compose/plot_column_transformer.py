"""
==================================================
محول الأعمدة مع مصادر بيانات غير متجانسة
==================================================

يمكن أن تحتوي مجموعات البيانات غالبًا على مكونات تتطلب معالجة واستخراج مميزات مختلفة. قد يحدث هذا السيناريو عندما:

1. تتكون مجموعة البيانات الخاصة بك من أنواع بيانات غير متجانسة (مثل صور نقطية ونصوص توضيحية)،
2. يتم تخزين مجموعة البيانات الخاصة بك في :class:`pandas.DataFrame` وتتطلب أعمدة مختلفة خطوط أنابيب معالجة مختلفة.

يوضح هذا المثال كيفية استخدام
:class:`~sklearn.compose.ColumnTransformer` على مجموعة بيانات تحتوي على
أنواع مختلفة من الميزات. اختيار الميزات ليس مفيدًا بشكل خاص، ولكنه يخدم لتوضيح التقنية.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

##############################################################################
# مجموعة بيانات 20 مجموعة أخبار
# ---------------------
#
# سنستخدم :ref:`مجموعة بيانات 20 مجموعة أخبار <20newsgroups_dataset>`، والتي
# تتألف من منشورات من مجموعات أخبار حول 20 موضوعًا. يتم تقسيم مجموعة البيانات هذه
# إلى مجموعات فرعية للتدريب والاختبار بناءً على الرسائل المنشورة قبل وبعد
# تاريخ محدد. سنستخدم فقط المنشورات من فئتين لتسريع وقت التشغيل.

categories = ["sci.med", "sci.space"]
X_train, y_train = fetch_20newsgroups(
    random_state=1,
    subset="train",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)
X_test, y_test = fetch_20newsgroups(
    random_state=1,
    subset="test",
    categories=categories,
    remove=("footers", "quotes"),
    return_X_y=True,
)

##############################################################################
# تتضمن كل ميزة معلومات وصفية حول هذا المنشور، مثل الموضوع،
# ونص منشور الأخبار.

print(X_train[0])

##############################################################################
# إنشاء المحولات
# ---------------------
#
# أولاً، نريد محولًا يستخرج الموضوع و
# نص كل منشور. نظرًا لأن هذا تحويل عديم الحالة (لا
# يتطلب معلومات الحالة من بيانات التدريب)، يمكننا تحديد دالة
# تقوم بتحويل البيانات ثم استخدام
# :class:`~sklearn.preprocessing.FunctionTransformer` لإنشاء محول scikit-learn.


def subject_body_extractor(posts):
    # بناء مصفوفة من نوع كائن مع عمودين
    # العمود الأول = 'subject' والعمود الثاني = 'body'
    features = np.empty(shape=(len(posts), 2), dtype=object)
    for i, text in enumerate(posts):
        # المتغير المؤقت `_` يخزن '\n\n'
        headers, _, body = text.partition("\n\n")
        # تخزين نص الجسم في العمود الثاني
        features[i, 1] = body

        prefix = "Subject:"
        sub = ""
        # حفظ النص بعد 'Subject:' في العمود الأول
        for line in headers.split("\n"):
            if line.startswith(prefix):
                sub = line[len(prefix):]
                break
        features[i, 0] = sub

    return features


subject_body_transformer = FunctionTransformer(subject_body_extractor)

##############################################################################
# سننشئ أيضًا محولًا يستخرج
# طول النص وعدد الجمل.


def text_stats(posts):
    return [{"length": len(text), "num_sentences": text.count(".")} for text in posts]


text_stats_transformer = FunctionTransformer(text_stats)

##############################################################################
# خط أنابيب التصنيف
# -----------------------
#
# يستخرج خط الأنابيب أدناه الموضوع والنص من كل منشور باستخدام
# ``SubjectBodyExtractor``، مما ينتج عنه مصفوفة (n_samples، 2). يتم استخدام هذه المصفوفة
# بعد ذلك لحساب ميزات حقيبة الكلمات القياسية للموضوع والنص
# بالإضافة إلى طول النص وعدد الجمل في النص، باستخدام
# ``ColumnTransformer``. نقوم بدمجها، مع أوزان، ثم ندرب
# مصنفًا على مجموعة الميزات المجمعة.

pipeline = Pipeline(
    [
        # استخراج الموضوع والنص
        ("subjectbody", subject_body_transformer),
        # استخدام ColumnTransformer لدمج ميزات الموضوع والنص
        (
            "union",
            ColumnTransformer(
                [
                    # حقيبة الكلمات للموضوع (العمود 0)
                    ("subject", TfidfVectorizer(min_df=50), 0),
                    # حقيبة الكلمات مع التحليل للنص (العمود 1)
                    (
                        "body_bow",
                        Pipeline(
                            [
                                ("tfidf", TfidfVectorizer()),
                                ("best", PCA(n_components=50, svd_solver="arpack")),
                            ]
                        ),
                        1,
                    ),
                    # خط أنابيب لسحب إحصائيات النص من نص المنشور
                    (
                        "body_stats",
                        Pipeline(
                            [
                                (
                                    "stats",
                                    text_stats_transformer,
                                ),  # يُرجع قائمة من القواميس
                                (
                                    "vect",
                                    DictVectorizer(),
                                ),  # قائمة القواميس -> مصفوفة الميزات
                            ]
                        ),
                        1,
                    ),
                ],
                # وزن ميزات ColumnTransformer أعلاه
                transformer_weights={
                    "subject": 0.8,
                    "body_bow": 0.5,
                    "body_stats": 1.0,
                },
            ),
        ),
        # استخدام مصنف SVC على الميزات المجمعة
        ("svc", LinearSVC(dual=False)),
    ],
    verbose=True,
)


##############################################################################
# أخيرًا، نقوم بملاءمة خط الأنابيب الخاص بنا على بيانات التدريب ونستخدمه للتنبؤ
# بالموضوعات لـ ``X_test``. ثم تتم طباعة مقاييس أداء خط الأنابيب الخاص بنا.

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Classification report:\n\n{}".format(
    classification_report(y_test, y_pred)))
