{#
  Replaces autosummary's default module template. As in the default template, the members are
  documented by automodule (selected according to autodoc_default_options), and the summary
  tables are part of the automodule content, so that they are placed between the module
  docstring and the documentation of the members. Differences:
  - Submodules are listed directly after the module docstring instead of at the end.
  - Classes and exceptions share one summary table, summary tables have no signatures.
  - The summary table for module attributes (data and type aliases with docstrings) is titled
    'Attributes' and placed last, in accordance with the order of the documented members.
  - The page of the pymor package is the top-level page of the API docs and titled 'API Reference'.
-#}
{% if fullname == 'pymor' -%}
API Reference
=============
{%- else -%}
{{ fullname | escape | underline }}
{%- endif %}

.. automodule:: {{ fullname }}

{% if modules %}
   .. rubric:: Submodules

   .. autosummary::
      :toctree:
      :recursive:
{% for item in modules %}
      {{ item }}
{%- endfor %}
{% endif %}

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

{% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
{% for item in attributes %}
      {{ item }}
{%- endfor %}
{% endif %}
