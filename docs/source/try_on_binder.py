from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive


class binder_link_node(nodes.General, nodes.Element):
    pass


class TryOnBinder(Directive):

    def run(self):
        env = self.state.document.settings.env
        app = env.app
        slug = app.config.try_on_binder_slug
        source = Path(self.state.document.attributes['source'])
        generated_nb = f'{source.stem}.ipynb'
        node = binder_link_node()
        # this is somewhat confusing, but the docs repository's branches are named after the directories
        # which are slugs to avoid slashes and such
        node['target'] = f'https://mybinder.org/v2/gh/pymor/docs/{slug}?filepath={generated_nb}'
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
    # since don;t want to replicate the slugify rules form gitlab we need to accept two params here
    app.add_config_value('try_on_binder_branch', 'master', 'html')
    app.add_config_value('try_on_binder_slug', 'master', 'html')
    app.add_node(binder_link_node,
                 html=(html_visit_binder_link_node, None))
    app.add_directive("try_on_binder", TryOnBinder)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
