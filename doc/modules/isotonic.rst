
.. _isotonic:

===================
الانحدار المتساوي التوتر
===================

.. currentmodule:: sklearn.isotonic

تُناسب فئة :class:`IsotonicRegression` دالة حقيقية غير متناقصة لـ
بيانات أحادية البعد. إنها تحل المشكلة التالية:

.. math::
    \min \sum_i w_i (y_i - \hat{y}_i)^2

بشرط :math:`\hat{y}_i \le \hat{y}_j` كلما كان :math:`X_i \le X_j`،
حيث تكون الأوزان :math:`w_i` موجبة تمامًا، و `X` و `y`
كميات حقيقية عشوائية.

تُغير معلمة `increasing` القيد إلى
:math:`\hat{y}_i \ge \hat{y}_j` كلما كان :math:`X_i \le X_j`. سيؤدي تعيينها إلى
'auto' إلى اختيار القيد تلقائيًا بناءً على `معامل ارتباط رتبة سبيرمان
<https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_.

تُنتج :class:`IsotonicRegression` سلسلة من التنبؤات
:math:`\hat{y}_i` لبيانات التدريب وهي الأقرب إلى الأهداف
:math:`y` من حيث متوسط الخطأ التربيعي. يتم استيفاء هذه التنبؤات
للتنبؤ ببيانات غير مرئية. وبالتالي تُشكِّل تنبؤات :class:`IsotonicRegression`
دالة خطية متعددة التعريف:

.. figure:: ../auto_examples/miscellaneous/images/sphx_glr_plot_isotonic_regression_001.png
   :target: ../auto_examples/miscellaneous/plot_isotonic_regression.html
   :align: center

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_isotonic_regression.py`


