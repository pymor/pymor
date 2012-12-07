if __name__ == "__main__":
    import nose
    import logging
    logging.basicConfig(level=logging.WARNING)
    nose.core.runmodule(name='__main__')