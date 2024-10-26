
.. _cross_decomposition:

===================
التحليل المتقاطع
===================

.. currentmodule:: sklearn.cross_decomposition

تحتوي وحدة التحليل المتقاطع على مقدرات **خاضعة للإشراف** لتقليل الأبعاد والانحدار، تنتمي إلى عائلة "المربعات الصغرى الجزئية".

.. figure:: ../auto_examples/cross_decomposition/images/sphx_glr_plot_compare_cross_decomposition_001.png
   :target: ../auto_examples/cross_decomposition/plot_compare_cross_decomposition.html
   :scale: 75%
   :align: center


تجد خوارزميات التحليل المتقاطع العلاقات الأساسية بين مصفوفتين (X و Y).
إنها مناهج متغيرة كامنة لنمذجة هياكل التباين المشترك في هاتين المساحتين.
سيحاولون إيجاد الاتجاه متعدد الأبعاد في مساحة X الذي يفسر اتجاه التباين متعدد الأبعاد الأقصى في مساحة Y.
بمعنى آخر، يقوم PLS بإسقاط كل من `X` و `Y` في فضاء فرعي منخفض الأبعاد بحيث يكون التباين المشترك بين `transformed(X)` و `transformed(Y)` هو الحد الأقصى.

يرسم PLS أوجه تشابه مع `Principal Component Regression <https://en.wikipedia.org/wiki/Principal_component_regression>`_ (PCR)، حيث يتم أولاً إسقاط العينات في فضاء فرعي منخفض الأبعاد، ويتم توقع الأهداف `y` باستخدام `transformed(X)`.
إحدى المشكلات المتعلقة بـ PCR هي أن تقليل الأبعاد غير خاضع للإشراف، وقد يفقد بعض المتغيرات المهمة: سيحتفظ PCR بالميزات ذات التباين الأكبر، ولكن من الممكن أن تكون الميزات ذات التباينات الصغيرة ذات صلة من التنبؤ بالهدف.
بطريقة ما، يسمح PLS بنفس النوع من تقليل الأبعاد، ولكن من خلال مراعاة الأهداف `y`.
يتم توضيح هذه الحقيقة في المثال التالي:
* :ref:`sphx_glr_auto_examples_cross_decomposition/plot_pcr_vs_pls.py`.

بصرف النظر عن CCA، فإن مقدرات PLS مناسبة بشكل خاص عندما تحتوي مصفوفة المتنبئين على متغيرات أكثر من الملاحظات، وعندما يكون هناك توافق متعدد بين الميزات.
على النقيض من ذلك، سيفشل الانحدار الخطي القياسي في هذه الحالات ما لم يتم تنظيمه.

الفئات المضمنة في هذه الوحدة هي :class:`PLSRegression` و :class:`PLSCanonical` و :class:`CCA` و :class:`PLSSVD`

PLSCanonical
------------

نصف هنا الخوارزمية المستخدمة في :class:`PLSCanonical`.
تستخدم المقدرات الأخرى متغيرات من هذه الخوارزمية، ويتم تفصيلها أدناه.
نوصي بالقسم [1]_ لمزيد من التفاصيل والمقارنات بين هذه الخوارزميات.
في [1]_، يتوافق :class:`PLSCanonical` مع "PLSW2A".

بالنظر إلى مصفوفتين متمركزتين :math:`X \in \mathbb{R}^{n \times d}` و
:math:`Y \in \mathbb{R}^{n \times t}`، وعدد من المكونات :math:`K`،
:class:`PLSCanonical`  على النحو التالي:

عيّن :math:`X_1` إلى :math:`X` و :math:`Y_1` إلى :math:`Y`. ثم، لكل
:math:`k \in [1, K]`:

- أ) احسب :math:`u_k \in \mathbb{R}^d` و :math:`v_k \in \mathbb{R}^t`،
  أول متجهات مفردة يسارية ويمنى لمصفوفة التباين المشترك
  :math:`C = X_k^T Y_k`.
  :math:`u_k` و :math:`v_k` تسمى *الأوزان*.
  بحكم التعريف، يتم اختيار :math:`u_k` و :math:`v_k` بحيث يزيدان من التباين المشترك بين
  :math:`X_k` المسقط والهدف المسقط، أي :math:`\text{Cov}(X_k u_k,Y_k v_k)`.

- ب) إسقاط :math:`X_k` و :math:`Y_k` على المتجهات المفردة للحصول على
  *الدرجات*: :math:`\xi_k = X_k u_k` و :math:`\omega_k = Y_k v_k`

- ج) انحدار :math:`X_k` على :math:`\xi_k`، أي إيجاد متجه :math:`\gamma_k
  \in \mathbb{R}^d` بحيث تكون مصفوفة الرتبة 1 :math:`\xi_k \gamma_k^T`
  أقرب ما يمكن إلى :math:`X_k`. افعل الشيء نفسه على :math:`Y_k` مع
  :math:`\omega_k` للحصول على :math:`\delta_k`. المتجهات
  :math:`\gamma_k` و :math:`\delta_k` تسمى *الأحمال*.

- د) *إزالة* :math:`X_k` و :math:`Y_k`، أي طرح الرتبة 1
  التقريبات: :math:`X_{k+1} = X_k - \xi_k \gamma_k^T`، و
  :math:`Y_{k + 1} = Y_k - \omega_k \delta_k^T`.

في النهاية، قمنا بتقريب :math:`X` كمجموع مصفوفات الرتبة 1:
:math:`X = \Xi \Gamma^T` حيث :math:`\Xi \in \mathbb{R}^{n \times K}`
يحتوي على الدرجات في أعمدته، و :math:`\Gamma^T \in \mathbb{R}^{K\times d}` يحتوي على الأحمال في صفوفه.
وبالمثل بالنسبة لـ :math:`Y`، لدينا :math:`Y = \Omega \Delta^T`.

لاحظ أن مصفوفات الدرجات :math:`\Xi` و :math:`\Omega` تتوافق مع إسقاطات بيانات التدريب :math:`X` و :math:`Y`، على التوالي.

يمكن تنفيذ الخطوة *أ)* بطريقتين: إما عن طريق حساب SVD الكامل لـ :math:`C` والاحتفاظ فقط بالمتجهات المفردة ذات أكبر قيم مفردة، أو عن طريق حساب المتجهات المفردة مباشرةً باستخدام طريقة الطاقة (راجع القسم 11.3 في [1]_ )، والذي يتوافق مع خيار `'nipals'` لمعلمة `algorithm`.

.. dropdown:: تحويل البيانات

  لتحويل :math:`X` إلى :math:`\bar{X}`، نحتاج إلى إيجاد مصفوفة إسقاط :math:`P` بحيث تكون :math:`\bar{X} = XP`.
  نعلم أنه بالنسبة لبيانات التدريب، :math:`\Xi = XP`، و :math:`X = \Xi \Gamma^T`.
  بتعيين :math:`P = U(\Gamma^T U)^{-1}` حيث :math:`U` هي المصفوفة مع :math:`u_k` في الأعمدة، لدينا :math:`XP = X U(\Gamma^T U)^{-1} = \Xi (\Gamma^T U) (\Gamma^T U)^{-1} = \Xi` كما هو مطلوب.
  يمكن الوصول إلى مصفوفة الدوران :math:`P` من سمة `x_rotations_`.

  وبالمثل، يمكن تحويل :math:`Y` باستخدام مصفوفة الدوران :math:`V(\Delta^T V)^{-1}`، والتي يمكن الوصول إليها عبر سمة `y_rotations_`.

.. dropdown:: التنبؤ بالأهداف `Y`

  للتنبؤ بأهداف بعض البيانات :math:`X`، نحن نبحث عن مصفوفة معامل :math:`\beta \in R^{d \times t}` بحيث تكون :math:`Y =X\beta`.

  الفكرة هي محاولة التنبؤ بالأهداف المحولة :math:`\Omega` كدالة للعينات المحولة :math:`\Xi`، عن طريق حساب :math:`\alpha \in \mathbb{R}` بحيث تكون :math:`\Omega = \alpha \Xi`.

  ثم، لدينا :math:`Y = \Omega \Delta^T = \alpha \Xi \Delta^T`، وبما أن :math:`\Xi` هي بيانات التدريب المحولة، لدينا :math:`Y = X \alpha P \Delta^T`، ونتيجة لذلك مصفوفة المعامل :math:`\beta = \alpha P \Delta^T`.

  يمكن الوصول إلى :math:`\beta` من خلال سمة `coef_`.

PLSSVD
------

:class:`PLSSVD` هو إصدار مبسط من :class:`PLSCanonical` الموصوف سابقًا: بدلاً من إزالة المصفوفات :math:`X_k` و :math:`Y_k` بشكل متكرر، يحسب :class:`PLSSVD` SVD لـ :math:`C = X^TY` *مرة واحدة فقط*، ويخزن `n_components` المتجهات المفردة المقابلة لأكبر القيم المفردة في المصفوفات `U` و `V`، المقابلة لسمات `x_weights_` و `y_weights_`.
هنا، البيانات المحولة هي ببساطة `transformed(X) = XU` و `transformed(Y) = YV`.

إذا كان `n_components == 1`، فإن :class:`PLSSVD` و :class:`PLSCanonical` متكافئان تمامًا.

PLSRegression
-------------

مقدر :class:`PLSRegression` مشابه لـ :class:`PLSCanonical` مع `algorithm='nipals'`، مع اختلافين مهمين:

- في الخطوة أ) في طريقة الطاقة لحساب :math:`u_k` و :math:`v_k`، لا يتم تطبيع :math:`v_k` أبدًا.
- في الخطوة ج)، يتم تقريب الأهداف :math:`Y_k` باستخدام إسقاط :math:`X_k` (أي :math:`\xi_k`) بدلاً من إسقاط :math:`Y_k` (أي :math:`\omega_k`).
  بمعنى آخر، يختلف حساب الأحمال.
  ونتيجة لذلك، سيتأثر الانكماش في الخطوة د) أيضًا.

يؤثر هذان التعديلان على ناتج `predict` و `transform`، وهما ليسا نفس ناتج :class:`PLSCanonical`.
أيضًا، بينما يكون عدد المكونات محدودًا بـ `min(n_samples, n_features, n_targets)` في :class:`PLSCanonical`، هنا يكون الحد هو رتبة :math:`X^TX`، أي `min(n_samples, n_features)`.

يُعرف :class:`PLSRegression` أيضًا باسم PLS1 (أهداف فردية) و PLS2 (أهداف متعددة). مثل :class:`~sklearn.linear_model.Lasso`،
:class:`PLSRegression` هو شكل من أشكال الانحدار الخطي المنتظم حيث يتحكم عدد المكونات في قوة التنظيم.

تحليل الارتباط المتعارف عليه
------------------------------

تم تطوير تحليل الارتباط المتعارف عليه قبل PLS وبشكل مستقل.
لكن اتضح أن :class:`CCA` هو حالة خاصة من PLS، ويتوافق مع PLS في "الوضع B" في الأدبيات.

يختلف :class:`CCA` عن :class:`PLSCanonical` في طريقة حساب الأوزان :math:`u_k` و :math:`v_k` في طريقة الطاقة للخطوة أ).
يمكن العثور على التفاصيل في القسم 10 من [1]_.

نظرًا لأن :class:`CCA` يتضمن انعكاس :math:`X_k^TX_k` و :math:`Y_k^TY_k`، يمكن أن يكون هذا المقدر غير مستقر إذا كان عدد الميزات أو الأهداف أكبر من عدد العينات.

.. rubric:: المراجع

.. [1] `A survey of Partial Least Squares (PLS) methods, with emphasis on the two-block case <https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf>`_، JA Wegelin

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_cross_decomposition/plot_compare_cross_decomposition.py`
* :ref:`sphx_glr_auto_examples_cross_decomposition/plot_pcr_vs_pls.py`


