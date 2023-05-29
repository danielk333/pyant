{{ objname | escape }}
{{ (objname | escape | length)*"=" }}

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if modules %}
Modules
-------

.. autosummary::
    :toctree: .
    {% for module in modules %}
    {{ module }}
    {% endfor %}

{% endif %}

{% if classes %}
Classes
-------

.. autosummary::
    :toctree: .
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
Functions
---------

.. autosummary::
    :toctree: .
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}
