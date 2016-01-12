# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import sqlite3
import datetime
import base64
import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

from pymor.core.cache import CacheRegion
from pymor.core.interfaces import BasicInterface
from pymor.core.pickle import dump, load


class NetworkFilesystemRegion(CacheRegion):

    persistent = True

    def __init__(self, server_path, secret=''):
        self.server = xmlrpclib.ServerProxy(server_path)
        self.secret = secret

    def get(self, key):
        key = base64.b64encode(key)
        response = self.server.get(self.secret, key)
        assert len(response) == 2 and isinstance(response[0], bool) and isinstance(response[1], str)
        if response[0]:
            file_path = response[1]
            with open(file_path) as f:
                value = load(f)
            return True, value
        else:
            return False, None

    def set(self, key, value):
        key = base64.b64encode(key)
        response = self.server.set(self.secret, key)
        assert len(response) == 2 and isinstance(response[0], bool) and isinstance(response[1], str)
        if response[0]:
            with open(response[1], 'w') as f:
                dump(value, f)
                file_size = f.tell()
            response = self.server.set_finished(self.secret, key, file_size)
            assert isinstance(response, bool) and response
        else:
            from pymor.core.logger import getLogger
            getLogger('pymor.core.network_cache.NetworkFilesystemRegion')\
                .warn('Key already present in cache region, ignoring.')

    def clear(self):
        raise NotImplementedError


class NetworkFilesystemRegionServer(BasicInterface):

    def __init__(self, addr, path, secret=None):
        self.server = server = SimpleXMLRPCServer(addr)
        server.register_function(self._get, 'get')
        server.register_function(self._set, 'set')
        server.register_function(self._set_finished, 'set_finished')
        self.path = path
        self.secret = secret
        if not os.path.exists(path):
            os.mkdir(path)
            self.conn = conn = sqlite3.connect(os.path.join(path, 'pymor_cache.db'))
            c = conn.cursor()
            c.execute('''CREATE TABLE entries
                         (id INTEGER PRIMARY KEY, key TEXT UNIQUE, filename TEXT, size INT)''')
            conn.commit()
        else:
            self.conn = sqlite3.connect(os.path.join(path, 'pymor_cache.db'))
        now = datetime.datetime.now()
        self.prefix = now.isoformat()
        self.created = 0

    def serve_forever(self):
        self.server.serve_forever()

    def _get(self, secret, key):
        if self.secret and secret != self.secret:
            return
        c = self.conn.cursor()
        t = (key,)
        c.execute('SELECT filename FROM entries WHERE key=?', t)
        result = c.fetchall()
        if len(result) == 0:
            return False, ''
        elif len(result) == 1:
            file_path = os.path.join(self.path, result[0][0])
            return True, file_path
        else:
            raise RuntimeError('Cache is corrupt!')

    def _set(self, secret, key):
        if self.secret and secret != self.secret:
            return
        filename = '{}-{:0>6}.dat'.format(self.prefix, self.created + 1)
        file_path = os.path.join(self.path, filename)
        conn = self.conn
        c = conn.cursor()
        try:
            c.execute("INSERT INTO entries(key, filename, size) VALUES ('{}', '{}', {})"
                      .format(key, filename, -1))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.commit()
            return (False, '')
        self.created += 1
        return (True, file_path)

    def _set_finished(self, secret, key, file_size):
        if self.secret and secret != self.secret:
            return
        conn = self.conn
        c = conn.cursor()
        t = (key,)
        c.execute('SELECT filename, size FROM entries WHERE key=?', t)
        result = c.fetchall()
        if len(result) == 0:
            return False
        elif len(result) == 1:
            filename, size = result[0]
            if size != -1:
                return False
            c = conn.cursor()
            try:
                c.execute("UPDATE entries SET size='{}' WHERE key='{}'".format(file_size, key))
                conn.commit()
            except sqlite3.IntegrityError:
                raise RuntimeError('Cache is corrupt!')
            return True
        else:
            raise RuntimeError('Cache is corrupt!')
