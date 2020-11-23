from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


class binder_link_node(nodes.General, nodes.Element):
    pass


class TryOnBinder(Directive):

    def run(self):
        env = self.state.document.settings.env
        app = env.app
        branch = app.config.try_on_binder_branch
        source = Path(self.state.document.attributes['source'])
        generated_nb = f'{branch}/{source.stem}.ipynb'
        node = binder_link_node()
        node['target'] = f'https://mybinder.org/v2/gh/pymor/docs/{branch}?filepath={generated_nb}'
        node['badge'] = 'https://mybinder.org/badge_logo.svg'
        return [node]


def html_visit_binder_link_node(self, node):
    html = f'''
    <a href=\"{node['target']}\">
        <img src=\"{node['badge']}\" alt=\"try on mybinder.org\">
    </a>
    '''
    self.body.append(html)
    raise nodes.SkipNode


def setup(app):
    app.add_config_value('try_on_binder_branch', 'master', 'html')
    app.add_node(binder_link_node,
                 html=(html_visit_binder_link_node, None))
    app.add_directive("try_on_binder", TryOnBinder)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }