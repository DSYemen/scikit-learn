
.. _cython:

أفضل ممارسات Cython والاتفاقيات والمعرفة
================================================

يوثق هذا النصائح لتطوير كود Cython في scikit-learn.

نصائح للتطوير باستخدام Cython في scikit-learn
-----------------------------------------------

نصائح لتسهيل التطوير
^^^^^^^^^^^^^^^^^^^^^^^^

* الوقت الذي تقضيه في قراءة `وثائق Cython <https://cython.readthedocs.io/en/latest/>`_ ليس وقتًا ضائعًا.

* إذا كنت تنوي استخدام OpenMP: في نظام MacOS، لا يُنفِّذ توزيع النظام ``clang`` لـ OpenMP.
  يمكنك تثبيت حزمة ``compilers`` المتاحة على ``conda-forge`` التي تأتي مع تنفيذ OpenMP.

* قد يساعد تنشيط `الفحوصات <https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87>`_. على سبيل المثال، لتنشيط boundscheck، استخدم:

  .. code-block:: bash

         export SKLEARN_ENABLE_DEBUG_CYTHON_DIRECTIVES=1

* `ابدأ من الصفر في دفتر ملاحظات <https://cython.readthedocs.io/en/latest/src/quickstart/build.html#using-the-jupyter-notebook>`_ لفهم كيفية استخدام Cython والحصول على تعليقات على عملك بسرعة.
  إذا كنت تخطط لاستخدام OpenMP لعمليات التنفيذ الخاصة بك في Jupyter Notebook، فقم بإضافة وسيطات مترجم ورابط إضافية في Cython magic.

  .. code-block:: python

         # لـ GCC و clang
         %%cython --compile-args=-fopenmp --link-args=-fopenmp
         # لمترجمات Microsoft
         %%cython --compile-args=/openmp --link-args=/openmp

* لتصحيح أخطاء كود C (على سبيل المثال، segfault)، استخدم ``gdb`` مع:

  .. code-block:: bash

         gdb --ex r --args python ./entrypoint_to_bug_reproducer.py

* للوصول إلى بعض القيمة في مكانها لتصحيح الأخطاء في سياق ``cdef (nogil)``، استخدم:

  .. code-block:: cython

         with gil:
             print(state_to_print)

* لاحظ أن Cython لا يمكنه تحليل سلاسل f مع تعبيرات ``{var=}``، على سبيل المثال

  .. code-block:: bash

         print(f"{test_val=}")

* تحتوي قاعدة كود scikit-learn على الكثير من تعريفات (إعادة تعريفات) الأنواع غير الموحدة (المدمجة).
  هناك حاليًا `عمل جارٍ لتبسيط ذلك وتوحيده عبر قاعدة التعليمات البرمجية
  <https://github.com/scikit-learn/scikit-learn/issues/25572>`_.
  في الوقت الحالي، تأكد من فهمك للأنواع الملموسة التي يتم استخدامها في النهاية.

* قد تجد هذا الاسم المستعار لتجميع ملحق Cython الفردي مفيدًا:

  .. code-block::

      # قد ترغب في إضافة هذا الاسم المستعار إلى تكوين البرنامج النصي shell الخاص بك.
      alias cythonX="cython -X language_level=3 -X boundscheck=False -X wraparound=False -X initializedcheck=False -X nonecheck=False -X cdivision=True"

      # يقوم هذا بإنشاء `source.c` كما لو كنت قد قمت بإعادة تجميع scikit-learn بالكامل.
      cythonX --annotate source.pyx

* يسمح استخدام خيار ``--annotate`` مع هذا العلم بإنشاء تقرير HTML لتعليق توضيحي للتعليمات البرمجية.
  يشير هذا التقرير إلى التفاعلات مع مترجم CPython على أساس كل سطر على حدة.
  يجب تجنب التفاعلات مع مترجم CPython قدر الإمكان في
  الأقسام كثيفة الحساب للخوارزميات.
  لمزيد من المعلومات، يرجى الرجوع إلى `هذا القسم من برنامج Cython التعليمي <https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#primes>`_

  .. code-block::

      # يقوم هذا بإنشاء تقرير HTML (`source.html`) لـ `source.c`.
      cythonX --annotate source.pyx

نصائح للأداء
^^^^^^^^^^^^^

* افهم GIL في سياق CPython (المشكلات التي يحلها، وما هي حدوده)
  واحصل على فهم جيد لوقت تعيين Cython إلى كود C خالٍ من التفاعلات مع
  CPython، ومتى لن يتم ذلك، ومتى لا يمكن ذلك (على سبيل المثال، وجود تفاعلات مع كائنات Python،
  والتي تتضمن دوال). في هذا الصدد، يوفر `PEP073 <https://peps.python.org/pep-0703/>`_
  نظرة عامة جيدة وسياقًا ومسارات للإزالة.

* تأكد من أنك قمت بإلغاء تنشيط `الفحوصات <https://github.com/scikit-learn/scikit-learn/blob/62a017efa047e9581ae7df8bbaa62cf4c0544ee4/sklearn/_build_utils/__init__.py#L68-L87>`_.

* فضّل دائمًا عروض الذاكرة على ``cnp.ndarray`` كلما أمكن ذلك: عروض الذاكرة خفيفة الوزن.

* تجنب تقسيم عروض الذاكرة: قد يكون تقسيم عروض الذاكرة مكلفًا أو مضللًا في بعض الحالات
  ومن الأفضل عدم استخدامه، حتى لو كان التعامل مع أبعاد أقل في بعض السياقات أمرًا مفضلًا.

* زيِّن الفئات أو الأساليب النهائية بـ ``@final`` (يسمح هذا بإزالة الجداول الافتراضية عند الحاجة)

* دوال وأساليب مضمنة عندما يكون ذلك منطقيًا

* في حالة الشك، اقرأ كود C أو C++ الذي تم إنشاؤه إذا استطعت: "كلما قل عدد تعليمات C والتوجيهات غير المباشرة
  لسطر كود Cython، كان ذلك أفضل" هي قاعدة جيدة.

* إعلانات ``nogil`` هي مجرد تلميحات: عند الإعلان عن دوال ``cdef``
  على أنها nogil، فهذا يعني أنه يمكن استدعاؤها دون الاحتفاظ بـ GIL، لكنها لا تُطلِق
  GIL عند الدخول إليها. عليك أن تفعل ذلك بنفسك إما عن طريق تمرير ``nogil=True`` إلى
  ``cython.parallel.prange`` صراحةً، أو باستخدام مدير سياق صريح:

  .. code-block:: cython

      cdef inline void my_func(self) nogil:

          # بعض المنطق الذي يتفاعل مع CPython، على سبيل المثال تخصيص مصفوفات عبر NumPy.

          with nogil:
              # يتم تشغيل الكود هنا كما لو كان مكتوبًا بلغة C.

          return 0

  يعتمد هذا العنصر على `هذا التعليق من Stéfan Benhel <https://github.com/cython/cython/issues/2798#issuecomment-459971828>`_

* يمكن إجراء استدعاءات مباشرة لإجراءات BLAS عبر واجهات مُعرَّفة في ``sklearn.utils._cython_blas``.

استخدام OpenMP
^^^^^^^^^^^^^^^

نظرًا لأنه يمكن بناء scikit-learn بدون OpenMP، فمن الضروري حماية كل
استدعاء مباشر لـ OpenMP.

توفر وحدة `_openmp_helpers`، المتاحة في
`sklearn/utils/_openmp_helpers.pyx <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_openmp_helpers.pyx>`_
إصدارات محمية من إجراءات OpenMP. لاستخدام إجراءات OpenMP، يجب
``cimported`` من هذه الوحدة وليس من مكتبة OpenMP مباشرةً:

.. code-block:: cython

   from sklearn.utils._openmp_helpers cimport omp_get_max_threads
   max_threads = omp_get_max_threads()


حلقات التكرار المتوازية، `prange`، محمية بالفعل بواسطة cython ويمكن استخدامها مباشرةً
من `cython.parallel`.

الأنواع
~~~~~~~

يتطلب كود Cython استخدام أنواع صريحة. هذا أحد أسباب حصولك على
زيادة في الأداء. لتجنب ازدواجية التعليمات البرمجية، لدينا مكان مركزي
للأنواع الأكثر استخدامًا في
`sklearn/utils/_typedefs.pyd <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_typedefs.pyd>`_.
من الناحية المثالية، تبدأ بإلقاء نظرة هناك و `cimport` الأنواع التي تحتاجها، على سبيل المثال

.. code-block:: cython

    from sklear.utils._typedefs cimport float32, float64


