.. _installation-instructions:

=======================
تثبيت scikit-learn
=======================

هناك طرق مختلفة لتثبيت scikit-learn:

* :ref:`قم بتثبيت أحدث إصدار رسمي <install_official_release>`.
  هذا هو أفضل نهج لمعظم المستخدمين.
  سوف يوفر إصدارًا ثابتًا وحزمًا مُعدة مسبقًا متاحة لمعظم الأنظمة الأساسية.

* قم بتثبيت إصدار scikit-learn الذي يوفره
  :ref:`نظام التشغيل أو توزيع Python <install_by_distribution>`.
  هذا خيار سريع لأولئك الذين لديهم أنظمة تشغيل أو توزيعات Python تقوم بتوزيع scikit-learn.
  قد لا يوفر إصدار الإصدار الأخير.

* :ref:`بناء الحزمة من المصدر
  <install_bleeding_edge>`. هذا هو الأفضل للمستخدمين الذين يريدون أحدث الميزات وأروعها ولا يخشون تشغيل كود جديد تمامًا.
  هذا مطلوب أيضًا للمستخدمين الذين يرغبون في المساهمة في المشروع.


.. _install_official_release:

تثبيت أحدث إصدار
======================

.. raw:: html

  <style>
    /* إظهار التسمية التوضيحية على الشاشات الكبيرة */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "نظام التشغيل";
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "مدير الحزم";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-os

    .. tab-item:: Windows
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          قم بتثبيت إصدار 64 بت من Python 3، على سبيل المثال من
          `الموقع الرسمي <https://www.python.org/downloads/windows/>`__.

          الآن قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وقم بتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية اختيارية ولكن يوصى بها بشدة، من أجل تجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: powershell

            python -m venv sklearn-env
            sklearn-env\Scripts\activate  # تفعيل
            pip install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: powershell

            python -m pip show scikit-learn  # إظهار إصدار وموقع scikit-learn
            python -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: MacOS
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          قم بتثبيت Python 3 باستخدام `homebrew <https://brew.sh/>`_ (`brew install python`)
          أو عن طريق تثبيت الحزمة يدويًا من `الموقع الرسمي <https://www.python.org/downloads/macos/>`__.

          الآن قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وقم بتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية اختيارية ولكن يوصى بها بشدة، من أجل تجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: bash

            python -m venv sklearn-env
            source sklearn-env/bin/activate  # تفعيل
            pip install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: bash

            python -m pip show scikit-learn  # إظهار إصدار وموقع scikit-learn
            python -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst

    .. tab-item:: Linux
      :class-label: tab-4

      .. tab-set::
        :class: tabs-package-manager

        .. tab-item:: pip
          :class-label: tab-6
          :sync: package-manager-pip

          عادةً ما يتم تثبيت Python 3 افتراضيًا على معظم توزيعات Linux.
          للتحقق مما إذا كان مثبتًا لديك، جرب:

          .. prompt:: bash

            python3 --version
            pip3 --version

          إذا لم يكن Python 3 مثبتًا لديك، فيرجى تثبيت `python3` و `python3-pip` من مدير حزم التوزيع الخاص بك.

          الآن قم بإنشاء `بيئة افتراضية (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ وقم بتثبيت scikit-learn.
          لاحظ أن البيئة الافتراضية اختيارية ولكن يوصى بها بشدة، من أجل تجنب التعارضات المحتملة مع الحزم الأخرى.

          .. prompt:: bash

            python3 -m venv sklearn-env
            source sklearn-env/bin/activate  # تفعيل
            pip3 install -U scikit-learn

          للتحقق من التثبيت، يمكنك استخدام:

          .. prompt:: bash

            python3 -m pip show scikit-learn  # إظهار إصدار وموقع scikit-learn
            python3 -m pip freeze             # إظهار جميع الحزم المثبتة في البيئة
            python3 -c "import sklearn; sklearn.show_versions()"

        .. tab-item:: conda
          :class-label: tab-6
          :sync: package-manager-conda

          .. include:: ./install_instructions_conda.rst


يجعل استخدام بيئة معزولة مثل pip venv أو conda من الممكن تثبيت إصدار محدد من scikit-learn مع pip أو conda وتبعياتها بشكل مستقل عن أي حزم Python مثبتة مسبقًا.
لا سيما في Linux، لا يُنصح بتثبيت حزم pip جنبًا إلى جنب مع الحزم التي يديرها مدير حزم التوزيع (apt، dnf، pacman ...).

لاحظ أنه يجب عليك دائمًا تذكر تنشيط البيئة التي تختارها قبل تشغيل أي أمر Python كلما بدأت جلسة طرفية جديدة.

إذا لم تكن قد قمت بتثبيت NumPy أو SciPy حتى الآن، فيمكنك أيضًا تثبيتها باستخدام conda أو pip.
عند استخدام pip، يرجى التأكد من استخدام *العجلات الثنائية*، وعدم إعادة تجميع NumPy و SciPy من المصدر، وهو ما يمكن أن يحدث عند استخدام تكوينات معينة لنظام التشغيل والأجهزة (مثل Linux على Raspberry Pi).

تتطلب إمكانات التخطيط في Scikit-learn (أي الوظائف التي تبدأ بـ `plot\_` والفئات التي تنتهي بـ `Display`) Matplotlib.
تتطلب الأمثلة Matplotlib وتتطلب بعض الأمثلة scikit-image أو pandas أو seaborn.
يتم سرد الحد الأدنى من إصدار تبعيات scikit-learn أدناه جنبًا إلى جنب مع الغرض منه.

.. include:: min_dependency_table.rst

.. warning::

    كان Scikit-learn 0.20 هو الإصدار الأخير الذي يدعم Python 2.7 و Python 3.4.
    يدعم Scikit-learn 0.21 Python 3.5-3.7.
    يدعم Scikit-learn 0.22 Python 3.5-3.8.
    يتطلب Scikit-learn 0.23-0.24 Python 3.6 أو أحدث.
    يدعم Scikit-learn 1.0 Python 3.7-3.10.
    يدعم Scikit-learn 1.1 و 1.2 و 1.3 Python 3.8-3.12
    يتطلب Scikit-learn 1.4 Python 3.9 أو أحدث.

.. _install_by_distribution:

## توزيعات الطرف الثالث من scikit-learn
=========================================

توفر بعض توزيعات الجهات الخارجية إصدارات من scikit-learn مدمجة مع أنظمة إدارة الحزم الخاصة بها.

يمكن أن تجعل هذه التثبيت والترقية أسهل بكثير للمستخدمين نظرًا لأن التكامل يتضمن القدرة على تثبيت التبعيات تلقائيًا (numpy، scipy) التي تتطلبها scikit-learn.

فيما يلي قائمة غير كاملة بتوزيعات نظام التشغيل و python التي توفر إصدارها الخاص من scikit-learn.

### Alpine Linux
------------

يتم توفير حزمة Alpine Linux من خلال `المستودعات الرسمية <https://pkgs.alpinelinux.org/packages?name=py3-scikit-learn>`__ باسم ``py3-scikit-learn`` لـ Python.
يمكن تثبيته عن طريق كتابة الأمر التالي:

.. prompt:: bash

  sudo apk add py3-scikit-learn


### Arch Linux
----------

يتم توفير حزمة Arch Linux من خلال `المستودعات الرسمية <https://www.archlinux.org/packages/?q=scikit-learn>`_ باسم ``python-scikit-learn`` لـ Python.
يمكن تثبيته عن طريق كتابة الأمر التالي:

.. prompt:: bash

  sudo pacman -S python-scikit-learn


### Debian/Ubuntu
-------------

تنقسم حزمة Debian / Ubuntu إلى ثلاث حزم مختلفة تسمى ``python3-sklearn`` (وحدات python) ، ``python3-sklearn-lib`` (التنفيذات والارتباطات منخفضة المستوى) ، ``python3-sklearn-doc`` (التوثيق).
لاحظ أن scikit-learn يتطلب Python 3، ومن ثم الحاجة إلى استخدام أسماء الحزم التي تحمل لاحقة `python3-`.
يمكن تثبيت الحزم باستخدام ``apt-get``:

.. prompt:: bash

  sudo apt-get install python3-sklearn python3-sklearn-lib python3-sklearn-doc


### Fedora
------

تسمى حزمة Fedora ``python3-scikit-learn`` لإصدار python 3، وهو الإصدار الوحيد المتاح في Fedora.
يمكن تثبيته باستخدام ``dnf``:

.. prompt:: bash

  sudo dnf install python3-scikit-learn


### NetBSD
------

يتوفر scikit-learn عبر `pkgsrc-wip <http://pkgsrc-wip.sourceforge.net/>`_:
https://pkgsrc.se/math/py-scikit-learn


### MacPorts for Mac OSX
--------------------

تسمى حزمة MacPorts ``py<XY>-scikits-learn``، حيث يشير ``XY`` إلى إصدار Python.
يمكن تثبيته عن طريق كتابة الأمر التالي:

.. prompt:: bash

  sudo port install py39-scikit-learn


### Anaconda and Enthought Deployment Manager لجميع الأنظمة الأساسية المدعومة
---------------------------------------------------------------------

`Anaconda <https://www.anaconda.com/download>`_ و `Enthought Deployment Manager <https://assets.enthought.com/downloads/>`_ كلاهما مزود بـ scikit-learn بالإضافة إلى مجموعة كبيرة من مكتبة python العلمية لنظام التشغيل Windows و Mac OSX و Linux.

يقدم Anaconda scikit-learn كجزء من توزيعه المجاني.


### Intel Extension for Scikit-learn
--------------------------------

تحتفظ Intel بحزمة x86_64 محسّنة، متاحة في PyPI (عبر `pip`)، وفي قنوات conda `main` و `conda-forge` و `intel`:

.. prompt:: bash

  conda install scikit-learn-intelex

تحتوي هذه الحزمة على إصدار محسن من Intel للعديد من المقدرات.
كلما لم يكن هناك تنفيذ بديل، يتم استخدام تنفيذ scikit-learn كخيار احتياطي.
تأتي أدوات الحل المحسّنة هذه من مكتبة oneDAL C ++ وتم تحسينها لهندسة x86_64، وتم تحسينها لوحدات المعالجة المركزية Intel متعددة النواة.

لاحظ أن أدوات الحل هذه غير ممكّنة افتراضيًا، يرجى الرجوع إلى `scikit-learn-intelex <https://intel.github.io/scikit-learn-intelex/latest/what-is-patching.html>`_ الوثائق لمزيد من التفاصيل حول سيناريوهات الاستخدام.
مثال التصدير المباشر:

.. prompt:: python >>>

  from sklearnex.neighbors import NearestNeighbors

يتم التحقق من التوافق مع أدوات حل scikit-learn القياسية عن طريق تشغيل مجموعة اختبار scikit-learn الكاملة عبر التكامل المستمر الآلي كما هو مذكور في https://github.com/intel/scikit-learn-intelex.
إذا لاحظت أي مشكلة مع `scikit-learn-intelex`، فيرجى الإبلاغ عن المشكلة على `متعقب المشكلات <https://github.com/intel/scikit-learn-intelex/issues>`__.


### WinPython for Windows
---------------------

يقوم مشروع `WinPython <https://winpython.github.io/>`_ بتوزيع scikit-learn كملحق إضافي.


## استكشاف الأخطاء وإصلاحها
===============

إذا واجهت أخطاء غير متوقعة عند تثبيت scikit-learn، فيمكنك إرسال مشكلة إلى `متعقب المشكلات <https://github.com/scikit-learn/scikit-learn/issues>`_.
قبل ذلك، يرجى أيضًا التأكد من التحقق من المشكلات الشائعة التالية.

.. _windows_longpath:

### خطأ ناتج عن حد طول مسار الملف على Windows
-------------------------------------------------

يمكن أن يحدث أن يفشل pip في تثبيت الحزم عند الوصول إلى حد حجم المسار الافتراضي لـ Windows إذا تم تثبيت Python في موقع متداخل مثل بنية مجلد `AppData` ضمن دليل المستخدم الرئيسي، على سبيل المثال::

    C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install scikit-learn
    Collecting scikit-learn
    ...
    Installing collected packages: scikit-learn
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\sklearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'

في هذه الحالة، من الممكن رفع هذا الحد في سجل Windows باستخدام أداة ``regedit``:

#. اكتب "regedit" في قائمة ابدأ في Windows لبدء تشغيل ``regedit``.

#. انتقل إلى مفتاح ``Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``.

#. قم بتحرير قيمة خاصية ``LongPathsEnabled`` لهذا المفتاح وقم بتعيينها على 1.

#. أعد تثبيت scikit-learn (تجاهل التثبيت المعطل السابق):

   .. prompt:: powershell

      pip install --exists-action=i scikit-learn


