

def skip(app, what, name, obj, skip, options):
    try:
        if ":noindex:" in obj.docstring:
            print(f'HERE DO_SKIP {name}')
            return True
    except AttributeError:
        pass
    return None


def setup(app):
    app.connect('autoapi-skip-member', skip)
    return {'parallel_read_safe': True}