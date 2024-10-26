
.. _data_reduction:

=====================================
تخفيض الأبعاد غير الخاضع للإشراف
=====================================

إذا كان عدد ميزاتك مرتفعًا، فقد يكون من المفيد تقليله بخطوة غير خاضعة للإشراف قبل الخطوات الخاضعة للإشراف. تُطبق العديد من أساليب :ref:`unsupervised-learning` طريقة ``transform`` التي يمكن استخدامها لتقليل الأبعاد. نناقش أدناه مثالين مُحدّدين لهذا النمط يُستخدمان بكثرة.

.. topic:: **خطوط الأنابيب**

    يمكن ربط اختزال البيانات غير الخاضع للإشراف والمقدر الخاضع للإشراف في خطوة واحدة. انظر :ref:`pipeline`.


.. currentmodule:: sklearn

PCA: تحليل المكونات الرئيسية
----------------------------------

يبحث :class:`decomposition.PCA` عن توليفة من الميزات التي تلتقط تباين الميزات الأصلية بشكل جيد. انظر :ref:`decompositions`.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`



الإسقاطات العشوائية
-------------------

تُوفر الوحدة: :mod:`~sklearn.random_projection` العديد من الأدوات لاختزال البيانات بواسطة الإسقاطات العشوائية. انظر القسم ذي الصلة من الوثائق: :ref:`random_projection`.


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_miscellaneous_plot_johnson_lindenstrauss_bound.py`


تجميع الميزات
------------------------

يطبق :class:`cluster.FeatureAgglomeration` :ref:`hierarchical_clustering` لتجميع الميزات التي تتصرف بشكل مُماثل.


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`

* :ref:`sphx_glr_auto_examples_cluster_plot_digits_agglomeration.py`


.. topic:: **قياس الميزات**

   لاحظ أنه إذا كانت الميزات لها قياس أو خصائص إحصائية مختلفة جدًا، فقد لا يكون :class:`cluster.FeatureAgglomeration` قادرًا على التقاط الروابط بين الميزات ذات الصلة. قد يكون استخدام :class:`preprocessing.StandardScaler` مفيدًا في هذه الإعدادات.


