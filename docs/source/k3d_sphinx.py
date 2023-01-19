import importlib
import shutil


def setup(app):
    to_copy_files = ['standalone.js', 'require.js']
    copied = []
    for fn in to_copy_files:
        for package_fn in importlib.metadata.files('k3d'):
            if fn in str(package_fn):
                shutil.copy(package_fn.locate(), './source/_static/')
                copied.append(fn)
                break
    assert copied == to_copy_files, copied

    app.add_js_file(filename=None, body='''
  <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
  <script>
    (function () {
      function addWidgetsRenderer() {
        var mimeElement = document.querySelector('script[type="application/vnd.jupyter.widget-view+json"]');
        var scriptElement = document.createElement('script');

        var widgetRendererSrc = 'https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js';

        var widgetState;

        // Fallback for older version:
        try {
          widgetState = mimeElement && JSON.parse(mimeElement.innerHTML);

          if (widgetState && (widgetState.version_major < 2 || !widgetState.version_major)) {

            var widgetRendererSrc = 'https://unpkg.com/@jupyter-js-widgets@*/dist/embed.js';

          }
        } catch (e) { }

        scriptElement.src = widgetRendererSrc;
        document.body.appendChild(scriptElement);
      }

      document.addEventListener('DOMContentLoaded', addWidgetsRenderer);
    }());
  </script>
''')


    return {
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
