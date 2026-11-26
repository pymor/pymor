{#
  Replaces autosummary's default module template. Differences:
  - automodule only renders the module docstring (:no-members:). The members are documented
    via autoclass/autoexception/autofunction/autodata for the items collected by autosummary.
    Consequently, module-level data is only documented when it has a docstring (autosummary
    ignores undocumented data, whereas automodule with undoc-members would include it).
    Members of classes are selected by autodoc as usual (autodoc_default_options).
  - Submodules are listed directly after the module docstring instead of at the end.
  - Classes and exceptions share one summary table, summary tables have no signatures,
    and there is no summary table for module attributes.
  - The page of the pymor package is the top-level page of the API docs and titled 'API Reference'.
-#}
{% if fullname == 'pymor' -%}
API Reference
=============
{%- else -%}
{{ fullname | escape | underline }}
{%- endif %}

.. automodule:: {{ fullname }}
   :no-members:

{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

.. currentmodule:: {{ fullname }}

{% if classes or exceptions %}
.. rubric:: Classes

.. autosummary::
   :nosignatures:
{% for item in classes + exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
   :nosignatures:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% for item in attributes %}
.. autodata:: {{ item }}
{% endfor %}

{% for item in classes %}
.. autoclass:: {{ item }}
{% endfor %}

{% for item in exceptions %}
.. autoexception:: {{ item }}
{% endfor %}

{% for item in functions %}
.. autofunction:: {{ item }}
{% endfor %}
