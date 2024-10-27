
``` bash
pip install sphinx sphinx-gallery numpydoc matplotlib Pillow pandas \
            polars scikit-image packaging seaborn sphinx-prompt \
            sphinxext-opengraph sphinx-copybutton plotly pooch \
            pydata-sphinx-theme sphinxcontrib-sass sphinx-design \
            sphinx-remove-toctrees pytest wheel numpy scipy cython \
            meson-python ninja jupyterlite_sphinx \
            pydata_sphinx_theme joblib threadpoolctl scikit-learn \
            --editable . \
            --verbose --no-build-isolation \
            --config-settings editable-verbose=true
```

```bash
cd doc

make html 

```