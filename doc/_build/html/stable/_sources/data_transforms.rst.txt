.. _data-transforms:

تحويلات مجموعات البيانات
-----------------------

يوفر Scikit-learn مكتبة من المحولات، والتي قد تنظف (انظر
:ref:`preprocessing`)، أو تقلل (انظر :ref:`data_reduction`)، أو توسع (انظر
:ref:`kernel_approximation`) أو تولد (انظر :ref:`feature_extraction`)
تمثيلات الميزات.

مثل المقدرات الأخرى، يتم تمثيلها بواسطة فئات باستخدام طريقة ``fit``،
والتي تتعلم معلمات النموذج (على سبيل المثال، المتوسط والانحراف المعياري لـ
التطبيع) من مجموعة التدريب، وطريقة ``transform`` التي تطبق
نموذج التحويل هذا على بيانات غير مرئية. قد يكون ``fit_transform`` أكثر
ملاءمة وفعالية لنمذجة وتحويل بيانات التدريب
في وقت واحد.

تمت تغطية الجمع بين هذه المحولات، إما بالتوازي أو بالتسلسل
:ref:`combining_estimators`. :ref:`metrics` يغطي تحويل الميزة
مسافات إلى مصفوفات تقارب، بينما :ref:`preprocessing_targets` يعتبر
تحويلات مساحة الهدف (على سبيل المثال، التصنيفات الفئوية) للاستخدام في
Scikit-learn.

.. toctree::
    :maxdepth: 2

    modules/compose
    modules/feature_extraction
    modules/preprocessing
    modules/impute
    modules/unsupervised_reduction
    modules/random_projection
    modules/kernel_approximation
    modules/metrics
    modules/preprocessing_targets

