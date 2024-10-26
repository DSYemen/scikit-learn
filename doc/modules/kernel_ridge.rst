
.. _kernel_ridge:

===========================
انحدار حافة النواة
===========================

.. currentmodule:: sklearn.kernel_ridge

انحدار حافة النواة (KRR) [M2012]_ يجمع بين :ref:`ridge_regression`
(المربعات الصغرى الخطية مع تنظيم قاعدة l2) مع `خدعة النواة
<https://en.wikipedia.org/wiki/Kernel_method>`_. وبالتالي يتعلم دالة
خطية في الفضاء الناتج عن النواة المعنية والبيانات. لـ
النوى غير الخطية، هذا يُقابل دالة غير خطية في الفضاء
الأصلي.

شكل النموذج الذي تعلمه :class:`KernelRidge` مطابق لانحدار متجه
الدعم (:class:`~sklearn.svm.SVR`). ومع ذلك، يتم استخدام دوال
خسارة مُختلفة: يستخدم KRR خسارة الخطأ التربيعي بينما يستخدم انحدار متجه
الدعم خسارة :math:`\epsilon` غير الحساسة، وكلاهما مُجتمع مع
تنظيم l2. على عكس :class:`~sklearn.svm.SVR`، يمكن إجراء ملاءمة
:class:`KernelRidge` في شكل مُغلق وعادةً ما يكون أسرع بالنسبة لـ
مجموعات البيانات متوسطة الحجم. من ناحية أخرى، فإن النموذج الذي تم تعلمه غير
متفرق وبالتالي أبطأ من :class:`~sklearn.svm.SVR`، الذي يتعلم نموذجًا متفرقًا
لـ :math:`\epsilon > 0`، في وقت التنبؤ.

تُقارن الصورة التالية :class:`KernelRidge` و
:class:`~sklearn.svm.SVR` على مجموعة بيانات اصطناعية، تتكون من
دالة هدف جيبية وضوضاء قوية تُضاف إلى كل نقطة بيانات خامسة.
يتم رسم النموذج الذي تعلمه :class:`KernelRidge` و
:class:`~sklearn.svm.SVR`، حيث تم تحسين كل من التعقيد / التنظيم
وعرض النطاق الترددي لنواة RBF باستخدام البحث الشبكي. الدوال التي تم
تعلمها متشابهة جدًا؛ ومع ذلك، فإن ملاءمة :class:`KernelRidge` أسرع بحوالي
سبع مرات من ملاءمة :class:`~sklearn.svm.SVR` (كلاهما مع البحث الشبكي).
ومع ذلك، فإن التنبؤ بـ 100000 قيمة هدف أسرع بأكثر من ثلاث مرات
باستخدام :class:`~sklearn.svm.SVR` لأنه تعلم نموذجًا متفرقًا باستخدام
1/3 فقط تقريبًا من نقاط بيانات التدريب البالغ عددها 100 كمتجهات دعم.

.. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_kernel_ridge_regression_001.png
   :target: ../auto_examples/miscellaneous/plot_kernel_ridge_regression.html
   :align: center

تُقارن الصورة التالية وقت ملاءمة وتنبؤ
:class:`KernelRidge` و :class:`~sklearn.svm.SVR` لأحجام مختلفة من
مجموعة التدريب. ملاءمة :class:`KernelRidge` أسرع من
:class:`~sklearn.svm.SVR` لمجموعات التدريب متوسطة الحجم (أقل من 1000
عينة)؛ ومع ذلك، بالنسبة لمجموعات التدريب الأكبر، يتناسب :class:`~sklearn.svm.SVR`
بشكل أفضل. فيما يتعلق بوقت التنبؤ، :class:`~sklearn.svm.SVR` أسرع
من :class:`KernelRidge` لجميع أحجام مجموعة التدريب بسبب
الحل المتفرق الذي تم تعلمه. لاحظ أن درجة التفرق، وبالتالي
وقت التنبؤ، يعتمد على المعلمتين :math:`\epsilon` و :math:`C`
لـ :class:`~sklearn.svm.SVR`؛ :math:`\epsilon = 0` سيُقابل
نموذجًا كثيفًا.

.. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_kernel_ridge_regression_002.png
   :target: ../auto_examples/miscellaneous/plot_kernel_ridge_regression.html
   :align: center

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_kernel_ridge_regression.py`

.. rubric:: المراجع

.. [M2012] "التعلم الآلي: منظور احتمالي"
   Murphy, K. P. - الفصل 14.4.3، الصفحات 492-493، The MIT Press، 2012


