# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import logging
import os
import sys
import re
import hashlib
from contextlib import closing  # for Python2.6 compatibility
import tarfile
import tempfile
from datetime import datetime
import json


#
# GLOBSTER
#

logger = logging.getLogger("globster")

"""Lazily compiled regex objects.

This module defines a class which creates proxy objects for regex
compilation.  This allows overriding re.compile() to return lazily compiled
objects.

We do this rather than just providing a new interface so that it will also
be used by existing Python modules that create regexs.
"""

class InvalidPattern(Exception):
    pass

class LazyRegex(object):
    """A proxy around a real regex, which won't be compiled until accessed."""


    # These are the parameters on a real _sre.SRE_Pattern object, which we
    # will map to local members so that we don't have the proxy overhead.
    _regex_attributes_to_copy = [
                 '__copy__', '__deepcopy__', 'findall', 'finditer', 'match',
                 'scanner', 'search', 'split', 'sub', 'subn'
                 ]

    # We use slots to keep the overhead low. But we need a slot entry for
    # all of the attributes we will copy
    __slots__ = ['_real_regex', '_regex_args', '_regex_kwargs',
                ] + _regex_attributes_to_copy

    def __init__(self, args=(), kwargs={}):
        """Create a new proxy object, passing in the args to pass to re.compile

        :param args: The `*args` to pass to re.compile
        :param kwargs: The `**kwargs` to pass to re.compile
        """
        self._real_regex = None
        self._regex_args = args
        self._regex_kwargs = kwargs

    def _compile_and_collapse(self):
        """Actually compile the requested regex"""
        self._real_regex = self._real_re_compile(*self._regex_args,
                                                 **self._regex_kwargs)
        for attr in self._regex_attributes_to_copy:
            setattr(self, attr, getattr(self._real_regex, attr))

    def _real_re_compile(self, *args, **kwargs):
        """Thunk over to the original re.compile"""
        try:
            return _real_re_compile(*args, **kwargs)
        except re.error as e:
            # raise InvalidPattern instead of re.error as this gives a
            # cleaner message to the user.
            raise InvalidPattern('"' + args[0] + '" ' +str(e))

    def __getstate__(self):
        """Return the state to use when pickling."""
        return {
            "args": self._regex_args,
            "kwargs": self._regex_kwargs,
            }

    def __setstate__(self, dict):
        """Restore from a pickled state."""
        self._real_regex = None
        setattr(self, "_regex_args", dict["args"])
        setattr(self, "_regex_kwargs", dict["kwargs"])

    def __getattr__(self, attr):
        """Return a member from the proxied regex object.

        If the regex hasn't been compiled yet, compile it
        """
        if self._real_regex is None:
            self._compile_and_collapse()
        # Once we have compiled, the only time we should come here
        # is actually if the attribute is missing.
        return getattr(self._real_regex, attr)


def lazy_compile(*args, **kwargs):
    """Create a proxy object which will compile the regex on demand.

    :return: a LazyRegex proxy object.
    """
    return LazyRegex(args, kwargs)


def install_lazy_compile():
    """Make lazy_compile the default compile mode for regex compilation.

    This overrides re.compile with lazy_compile. To restore the original
    functionality, call reset_compile().
    """
    re.compile = lazy_compile


def reset_compile():
    """Restore the original function to re.compile().

    It is safe to call reset_compile() multiple times, it will always
    restore re.compile() to the value that existed at import time.
    Though the first call will reset back to the original (it doesn't
    track nesting level)
    """
    re.compile = _real_re_compile


_real_re_compile = re.compile
if _real_re_compile is lazy_compile:
    raise AssertionError(
        "re.compile has already been overridden as lazy_compile, but this would" \
        " cause infinite recursion")


"""Tools for converting globs to regular expressions.

This module provides functions for converting shell-like globs to regular
expressions.
"""

# Dummy out some bzr internals
def mutter(x):
    logger.debug(x)

trace = mutter

class Replacer(object):
    """Do a multiple-pattern substitution.

    The patterns and substitutions are combined into one, so the result of
    one replacement is never substituted again. Add the patterns and
    replacements via the add method and then call the object. The patterns
    must not contain capturing groups.
    """

    _expand = lazy_compile(r'\\&')

    def __init__(self, source=None):
        self._pat = None
        if source:
            self._pats = list(source._pats)
            self._funs = list(source._funs)
        else:
            self._pats = []
            self._funs = []

    def add(self, pat, fun):
        r"""Add a pattern and replacement.

        The pattern must not contain capturing groups.
        The replacement might be either a string template in which \& will be
        replaced with the match, or a function that will get the matching text
        as argument. It does not get match object, because capturing is
        forbidden anyway.
        """
        self._pat = None
        self._pats.append(pat)
        self._funs.append(fun)

    def add_replacer(self, replacer):
        r"""Add all patterns from another replacer.

        All patterns and replacements from replacer are appended to the ones
        already defined.
        """
        self._pat = None
        self._pats.extend(replacer._pats)
        self._funs.extend(replacer._funs)

    def __call__(self, text):
        if not self._pat:
            self._pat = lazy_compile(
                    '|'.join(['(%s)' % p for p in self._pats]),
                    re.UNICODE)
        return self._pat.sub(self._do_sub, text)

    def _do_sub(self, m):
        fun = self._funs[m.lastindex - 1]
        if hasattr(fun, '__call__'):
            return fun(m.group(0))
        else:
            return self._expand.sub(m.group(0), fun)


_sub_named = Replacer()
_sub_named.add(r'\[:digit:\]', r'\d')
_sub_named.add(r'\[:space:\]', r'\s')
_sub_named.add(r'\[:alnum:\]', r'\w')
_sub_named.add(r'\[:ascii:\]', r'\0-\x7f')
_sub_named.add(r'\[:blank:\]', r' \t')
_sub_named.add(r'\[:cntrl:\]', r'\0-\x1f\x7f-\x9f')


def _sub_group(m):
    if m[1] in ('!', '^'):
        return '[^' + _sub_named(m[2:-1]) + ']'
    return '[' + _sub_named(m[1:-1]) + ']'


def _invalid_regex(repl):
    def _(m):
        warning("'%s' not allowed within a regular expression. "
                "Replacing with '%s'" % (m, repl))
        return repl
    return _


def _trailing_backslashes_regex(m):
    """Check trailing backslashes.

    Does a head count on trailing backslashes to ensure there isn't an odd
    one on the end that would escape the brackets we wrap the RE in.
    """
    if (len(m) % 2) != 0:
        warning("Regular expressions cannot end with an odd number of '\\'. "
                "Dropping the final '\\'.")
        return m[:-1]
    return m


_sub_re = Replacer()
_sub_re.add('^RE:', '')
_sub_re.add('\((?!\?)', '(?:')
_sub_re.add('\(\?P<.*>', _invalid_regex('(?:'))
_sub_re.add('\(\?P=[^)]*\)', _invalid_regex(''))
_sub_re.add(r'\\+$', _trailing_backslashes_regex)


_sub_fullpath = Replacer()
_sub_fullpath.add(r'^RE:.*', _sub_re) # RE:<anything> is a regex
_sub_fullpath.add(r'\[\^?\]?(?:[^][]|\[:[^]]+:\])+\]', _sub_group) # char group
_sub_fullpath.add(r'(?:(?<=/)|^)(?:\.?/)+', '') # canonicalize path
_sub_fullpath.add(r'\\.', r'\&') # keep anything backslashed
_sub_fullpath.add(r'[(){}|^$+.]', r'\\&') # escape specials
_sub_fullpath.add(r'(?:(?<=/)|^)\*\*+/', r'(?:.*/)?') # **/ after ^ or /
_sub_fullpath.add(r'\*+', r'[^/]*') # * elsewhere
_sub_fullpath.add(r'\?', r'[^/]') # ? everywhere


_sub_basename = Replacer()
_sub_basename.add(r'\[\^?\]?(?:[^][]|\[:[^]]+:\])+\]', _sub_group) # char group
_sub_basename.add(r'\\.', r'\&') # keep anything backslashed
_sub_basename.add(r'[(){}|^$+.]', r'\\&') # escape specials
_sub_basename.add(r'\*+', r'.*') # * everywhere
_sub_basename.add(r'\?', r'.') # ? everywhere


def _sub_extension(pattern):
    return _sub_basename(pattern[2:])


class Globster(object):
    """A simple wrapper for a set of glob patterns.

    Provides the capability to search the patterns to find a match for
    a given filename (including the full path).

    Patterns are translated to regular expressions to expidite matching.

    The regular expressions for multiple patterns are aggregated into
    a super-regex containing groups of up to 99 patterns.
    The 99 limitation is due to the grouping limit of the Python re module.
    The resulting super-regex and associated patterns are stored as a list of
    (regex,[patterns]) in _regex_patterns.

    For performance reasons the patterns are categorised as extension patterns
    (those that match against a file extension), basename patterns
    (those that match against the basename of the filename),
    and fullpath patterns (those that match against the full path).
    The translations used for extensions and basenames are relatively simpler
    and therefore faster to perform than the fullpath patterns.

    Also, the extension patterns are more likely to find a match and
    so are matched first, then the basename patterns, then the fullpath
    patterns.
    """
    # We want to _add_patterns in a specific order (as per type_list below)
    # starting with the shortest and going to the longest.
    # As some Python version don't support ordered dicts the list below is
    # used to select inputs for _add_pattern in a specific order.
    pattern_types = [ "extension", "basename", "fullpath" ]

    pattern_info = {
        "extension" : {
            "translator" : _sub_extension,
            "prefix" : r'(?:.*/)?(?!.*/)(?:.*\.)'
        },
        "basename" : {
            "translator" : _sub_basename,
            "prefix" : r'(?:.*/)?(?!.*/)'
        },
        "fullpath" : {
            "translator" : _sub_fullpath,
            "prefix" : r''
        },
    }

    def __init__(self, patterns, debug=False):
        self._regex_patterns = []
        self.debug = debug
        pattern_lists = {
            "extension" : [],
            "basename" : [],
            "fullpath" : [],
        }
        for pat in patterns:
            pat = normalize_pattern(pat)
            pattern_lists[Globster.identify(pat)].append(pat)
        pi = Globster.pattern_info
        for t in Globster.pattern_types:
            self._add_patterns(pattern_lists[t], pi[t]["translator"],
                pi[t]["prefix"])

    def _add_patterns(self, patterns, translator, prefix=''):
        while patterns:
            grouped_rules = [
                '(%s)' % translator(pat) for pat in patterns[:99]]
            joined_rule = '%s(?:%s)$' % (prefix, '|'.join(grouped_rules))
            # Explicitly use lazy_compile here, because we count on its
            # nicer error reporting.
            self._regex_patterns.append((
                lazy_compile(joined_rule, re.UNICODE),
                patterns[:99]))
            patterns = patterns[99:]

    def match(self, filename):
        """Searches for a pattern that matches the given filename.

        :return A matching pattern or None if there is no matching pattern.
        """

        try:
            for regex, patterns in self._regex_patterns:
                match = regex.match(filename)

                debug_template = "%s against %s: %%s" % (filename, regex._real_regex.pattern)

                if match:
                    if self.debug:
                        logger.info(debug_template % "hit")
                    return patterns[match.lastindex -1]

                if self.debug:
                    logger.info(debug_template % "miss")

        except Exception as e:
            # We can't show the default e.msg to the user as thats for
            # the combined pattern we sent to regex. Instead we indicate to
            # the user that an ignore file needs fixing.
            logger.error('Invalid pattern found in regex: %s.', e.msg)
            e.msg = "File ~/.bazaar/ignore or .bzrignore contains error(s)."
            bad_patterns = ''
            for _, patterns in self._regex_patterns:
                for p in patterns:
                    if not Globster.is_pattern_valid(p):
                        bad_patterns += ('\n  %s' % p)
            e.msg += bad_patterns
            raise e


        return None

    @staticmethod
    def identify(pattern):
        """Returns pattern category.

        :param pattern: normalized pattern.
        Identify if a pattern is fullpath, basename or extension
        and returns the appropriate type.
        """
        if pattern.startswith('RE:') or '/' in pattern:
            return "fullpath"
        elif pattern.startswith('*.'):
            return "extension"
        else:
            return "basename"

    @staticmethod
    def is_pattern_valid(pattern):
        """Returns True if pattern is valid.

        :param pattern: Normalized pattern.
        is_pattern_valid() assumes pattern to be normalized.
        see: globbing.normalize_pattern
        """
        result = True
        translator = Globster.pattern_info[Globster.identify(pattern)]["translator"]
        tpattern = '(%s)' % translator(pattern)
        try:
            re_obj = lazy_compile(tpattern, re.UNICODE)
            re_obj.search("") # force compile
        except Exception as e:
            result = False
        return result


class ExceptionGlobster(object):
    """A Globster that supports exception patterns.

    Exceptions are ignore patterns prefixed with '!'.  Exception
    patterns take precedence over regular patterns and cause a
    matching filename to return None from the match() function.
    Patterns using a '!!' prefix are highest precedence, and act
    as regular ignores. '!!' patterns are useful to establish ignores
    that apply under paths specified by '!' exception patterns.
    """

    def __init__(self,patterns, debug):
        ignores = [[], [], []]
        for p in patterns:
            if p.startswith('!!'):
                ignores[2].append(p[2:])
            elif p.startswith('!'):
                ignores[1].append(p[1:])
            else:
                ignores[0].append(p)
        self._ignores = [Globster(i, debug) for i in ignores]

    def match(self, filename):
        """Searches for a pattern that matches the given filename.

        :return A matching pattern or None if there is no matching pattern.
        """

        double_neg = self._ignores[2].match(filename)
        if double_neg:
            return "!!%s" % double_neg
        elif self._ignores[1].match(filename):
            #print("Ignores")
            return None
        else:
            #print("Normal match")
            return self._ignores[0].match(filename)

class _OrderedGlobster(Globster):
    """A Globster that keeps pattern order."""

    def __init__(self, patterns):
        """Constructor.

        :param patterns: sequence of glob patterns
        """
        # Note: This could be smarter by running like sequences together
        self._regex_patterns = []
        for pat in patterns:
            pat = normalize_pattern(pat)
            t = Globster.identify(pat)
            self._add_patterns([pat], Globster.pattern_info[t]["translator"],
                Globster.pattern_info[t]["prefix"])


_slashes = lazy_compile(r'[\\/]+')
def normalize_pattern(pattern):
    """Converts backslashes in path patterns to forward slashes.

    Doesn't normalize regular expressions - they may contain escapes.
    """
    if not (pattern.startswith('RE:') or pattern.startswith('!RE:')):
        pattern = _slashes.sub('/', pattern)
    if len(pattern) > 1:
        pattern = pattern.rstrip('/')
    return pattern


#
# DIRTOOLS
#


log = logging.getLogger("dirtools")

# TODO abs=True args for .files(), .subdirs() ?


def load_patterns(exclude_file=".exclude"):
    """ Load patterns to exclude file from `exclude_file',
    and return a list of pattern.

    :type exclude_file: str
    :param exclude_file: File containing exclude patterns

    :rtype: list
    :return: List a patterns

    """
    return filter(None, open(exclude_file).read().split("\n"))


def _filehash(filepath, blocksize=4096):
    """ Return the hash object for the file `filepath', processing the file
    by chunk of `blocksize'.

    :type filepath: str
    :param filepath: Path to file

    :type blocksize: int
    :param blocksize: Size of the chunk when processing the file

    """
    sha = hashlib.sha256()
    with open(filepath, 'rb') as fp:
        while 1:
            data = fp.read(blocksize)
            if data:
                sha.update(data)
            else:
                break
    return sha


def posix_path(fpath):
    """Convert a filesystem path to a posix path.

    Always use the forward slash as a separator. For instance,
    in windows the separator is the backslash.

    Args:
        fpath: The path to convert.
    """
    return fpath if os.altsep is None else fpath.replace(os.sep, os.altsep)



def native_path(fpath):
    """Convert a filesystem path to a native path.

    Use whatever separator is defined by the platform.

    Args:
        fpath: The path to convert.
    """
    return fpath if os.altsep is None else fpath.replace(os.altsep, os.sep)


def filehash(filepath, blocksize=4096):
    """ Return the hash hexdigest() for the file `filepath', processing the file
    by chunk of `blocksize'.

    :type filepath: str
    :param filepath: Path to file

    :type blocksize: int
    :param blocksize: Size of the chunk when processing the file

    """
    sha = _filehash(filepath, blocksize)
    return sha.hexdigest()


class File(object):
    def __init__(self, path):
        self.file = os.path.basename(path)
        self.path = os.path.abspath(path)

    def _hash(self):
        """ Return the hash object. """
        return _filehash(self.path)

    def hash(self):
        """ Return the hash hexdigest. """
        return filehash(self.path)

    def compress_to(self, archive_path=None):
        """ Compress the directory with gzip using tarlib.

        :type archive_path: str
        :param archive_path: Path to the archive, if None, a tempfile is created

        """
        if archive_path is None:
            archive = tempfile.NamedTemporaryFile(delete=False)
            tar_args = ()
            tar_kwargs = {'fileobj': archive}
            _return = archive.name
        else:
            tar_args = (archive_path)
            tar_kwargs = {}
            _return = archive_path
        tar_kwargs.update({'mode': 'w:gz'})
        with closing(tarfile.open(*tar_args, **tar_kwargs)) as tar:
            tar.add(self.path, arcname=self.file)

        return _return


class Dir(object):
    """ Wrapper for dirtools arround a path.

    Try to load a .exclude file, ready to compute hashdir,


    :type directory: str
    :param directory: Root directory for initialization

    :type exclude_file: str
    :param exclude_file: File containing exclusion pattern,
        .exclude by default, you can also load .gitignore files.

    :type excludes: list
    :param excludes: List of additionals patterns for exclusion,
        by default: ['.git/', '.hg/', '.svn/']

    """
    def __init__(self, directory=".", exclude_file=".exclude",
                 excludes=['.git/', '.hg/', '.svn/']):
        if not os.path.isdir(directory):
            raise TypeError("Directory must be a directory.")
        self.directory = os.path.basename(directory)
        self.path = os.path.abspath(directory)
        self.parent = os.path.dirname(self.path)
        self.exclude_file = os.path.join(self.path, exclude_file)
        if not os.path.isfile(self.exclude_file):
            self.exclude_file = os.path.join(__file__, exclude_file)
        if not os.path.isfile(self.exclude_file):
            self.exclude_file = exclude_file
        self.patterns = excludes
        if os.path.isfile(self.exclude_file):
            self.patterns.extend(load_patterns(self.exclude_file))
        self.globster = Globster(self.patterns)

    def hash(self, index_func=os.path.getmtime):
        """ Hash for the entire directory (except excluded files) recursively.

        Use mtime instead of sha256 by default for a faster hash.

        >>> dir.hash(index_func=dirtools.filehash)

        """
        # TODO alternative to filehash => mtime as a faster alternative
        shadir = hashlib.sha256()
        for f in self.files():
            try:
                shadir.update(str(index_func(os.path.join(self.path, f))))
            except (IOError, OSError):
                pass
        return shadir.hexdigest()

    def iterfiles(self, pattern=None, abspath=False):
        """ Generator for all the files not excluded recursively.

        Return relative path.

        :type pattern: str
        :param pattern: Unix style (glob like/gitignore like) pattern

        """
        if pattern is not None:
            globster = Globster([pattern])
        for root, dirs, files in self.walk():
            for f in files:
                if pattern is None or (pattern is not None and globster.match(f)):
                    if abspath:
                        yield os.path.join(root, f)
                    else:
                        yield self.relpath(os.path.join(root, f))

    def files(self, pattern=None, sort_key=lambda k: k, sort_reverse=False, abspath=False):
        """ Return a sorted list containing relative path of all files (recursively).

        :type pattern: str
        :param pattern: Unix style (glob like/gitignore like) pattern

        :param sort_key: key argument for sorted

        :param sort_reverse: reverse argument for sorted

        :rtype: list
        :return: List of all relative files paths.

        """
        return sorted(self.iterfiles(pattern, abspath=abspath), key=sort_key, reverse=sort_reverse)

    def get(self, pattern, sort_key=lambda k: k, sort_reverse=False, abspath=False):
        res = self.files(pattern, sort_key=sort_key, sort_reverse=sort_reverse, abspath=abspath)
        if res:
            return res[0]

    def itersubdirs(self, pattern=None, abspath=False):
        """ Generator for all subdirs (except excluded).

        :type pattern: str
        :param pattern: Unix style (glob like/gitignore like) pattern

        """
        if pattern is not None:
            globster = Globster([pattern])
        for root, dirs, files in self.walk():
            for d in dirs:
                if pattern is None or (pattern is not None and globster.match(d)):
                    if abspath:
                        yield os.path.join(root, d)
                    else:
                        yield self.relpath(os.path.join(root, d))

    def subdirs(self, pattern=None, sort_key=lambda k: k, sort_reverse=False, abspath=False):
        """ Return a sorted list containing relative path of all subdirs (recursively).

        :type pattern: str
        :param pattern: Unix style (glob like/gitignore like) pattern

        :param sort_key: key argument for sorted

        :param sort_reverse: reverse argument for sorted

        :rtype: list
        :return: List of all relative files paths.
        """
        return sorted(self.itersubdirs(pattern, abspath=abspath), key=sort_key, reverse=sort_reverse)

    def size(self):
        """ Return directory size in bytes.

        :rtype: int
        :return: Total directory size in bytes.
        """
        dir_size = 0
        for f in self.iterfiles(abspath=True):
            dir_size += os.path.getsize(f)
        return dir_size

    def is_excluded(self, path):
        """ Return True if `path' should be excluded
        given patterns in the `exclude_file'. """
        match = self.globster.match(self.relpath(path)) or self.globster.match(posix_path(self.relpath(path)))
        if match:
            log.debug("{0} matched {1} for exclusion".format(path, match))
            return True
        return False

    def walk(self):
        """ Walk the directory like os.path
        (yields a 3-tuple (dirpath, dirnames, filenames)
        except it exclude all files/directories on the fly. """
        for root, dirs, files in os.walk(self.path, topdown=True):
            # TODO relative walk, recursive call if root excluder found???
            #root_excluder = get_root_excluder(root)
            ndirs = []
            # First we exclude directories
            for d in list(dirs):
                if self.is_excluded(os.path.join(root, d)):
                    dirs.remove(d)
                elif not os.path.islink(os.path.join(root, d)):
                    ndirs.append(d)

            nfiles = []
            for fpath in (os.path.join(root, f) for f in files):
                if not self.is_excluded(fpath) and not os.path.islink(fpath):
                    nfiles.append(os.path.relpath(fpath, root))

            yield root, ndirs, nfiles

    def find_projects(self, file_identifier=".project"):
        """ Search all directory recursively for subdirs
        with `file_identifier' in it.

        :type file_identifier: str
        :param file_identifier: File identier, .project by default.

        :rtype: list
        :return: The list of subdirs with a `file_identifier' in it.

        """
        projects = []
        for d in self.subdirs():
            project_file = os.path.join(self.directory, d, file_identifier)
            if os.path.isfile(project_file):
                projects.append(d)
        return projects

    def relpath(self, path):
        """ Return a relative filepath to path from Dir path. """
        return os.path.relpath(path, start=self.path)

    def compress_to(self, archive_path=None):
        """ Compress the directory with gzip using tarlib.

        :type archive_path: str
        :param archive_path: Path to the archive, if None, a tempfile is created

        """
        if archive_path is None:
            archive = tempfile.NamedTemporaryFile(delete=False)
            tar_args = []
            tar_kwargs = {'fileobj': archive}
            _return = archive.name
        else:
            tar_args = [archive_path]
            tar_kwargs = {}
            _return = archive_path
        tar_kwargs.update({'mode': 'w:gz'})
        with closing(tarfile.open(*tar_args, **tar_kwargs)) as tar:
            tar.add(self.path, arcname='', exclude=self.is_excluded)

        return _return


class DirState(object):
    """ Hold a directory state / snapshot meta-data for later comparison. """
    def __init__(self, _dir=None, state=None, index_cmp=os.path.getmtime):
        self._dir = _dir
        self.index_cmp = index_cmp
        self.state = state or self.compute_state()

    def compute_state(self):
        """ Generate the index. """
        data = {}
        data['directory'] = self._dir.path
        data['files'] = list(self._dir.files())
        data['subdirs'] = list(self._dir.subdirs())
        data['index'] = self.index()
        return data

    def index(self):
        index = {}
        for f in self._dir.iterfiles():
            try:
                index[f] = self.index_cmp(os.path.join(self._dir.path, f))
            except Exception as exc:
                print(f, exc)
        return index

    def __sub__(self, other):
        """ Compute diff with operator overloading.

        >>> path = DirState(Dir('/path'))
        >>> path_copy = DirState(Dir('/path_copy'))
        >>> diff =  path_copy - path
        >>> # Equals to
        >>> diff = compute_diff(path_copy.state, path.state)

        """
        if self.index_cmp != other.index_cmp:
            raise Exception('Both DirState instance must have the same index_cmp.')
        return compute_diff(self.state, other.state)

    def to_json(self, base_path='.', dt=None, fmt=None):
        if fmt is None:
            fmt = '{0}@{1}.json'
        if dt is None:
            dt = datetime.utcnow()
        path = fmt.format(self._dir.path.strip('/').split('/')[-1],
                          dt.isoformat())
        path = os.path.join(base_path, path)

        with open(path, 'wb') as f:
            f.write(json.dumps(self.state))

        return path

    @classmethod
    def from_json(cls, path):
        with open(path, 'rb') as f:
            return cls(state=json.loads(f.read()))


def compute_diff(dir_base, dir_cmp):
    """ Compare `dir_base' and `dir_cmp' and returns a list with
    the following keys:
     - deleted files `deleted'
     - created files `created'
     - updated files `updated'
     - deleted directories `deleted_dirs'

    """
    data = {}
    data['deleted'] = list(set(dir_cmp['files']) - set(dir_base['files']))
    data['created'] = list(set(dir_base['files']) - set(dir_cmp['files']))
    data['updated'] = []
    data['deleted_dirs'] = list(set(dir_cmp['subdirs']) - set(dir_base['subdirs']))

    for f in set(dir_cmp['files']).intersection(set(dir_base['files'])):
        if dir_base['index'][f] != dir_cmp['index'][f]:
            data['updated'].append(f)

    return data


#
# DISPLAY UTILS
#

#: Storage size symbols
SYM_NAMES = ('Byte', 'Kb', 'Mb', 'Gb', 'Tb', 'Pb', 'Xb', 'Zb', 'Yb')


def human2bytes(value):
    """ Attempts to guess the string format based on default symbols set and
    return the corresponding bytes as an integer. When unable to recognize
    the format ValueError is raised.

    Function itself is case-insensitive means 'gb' = 'Gb' = 'GB' for gigabyte.
    It does not support bytes (as in 300b) and any numeric value will be
    considered as megabyte. Supported file sizes are:

    * Kb: Kilobyte
    * Mb: Megabyte
    * Gb: Gigabyte
    * Tb: Terabyte
    * Pb: Petabyte
    * Xb: Exabyte
    * Zb: Zettabyte
    * Yb: Yottabyte

      >>>  human2bytes('400') == human2bytes('400 byte') == 400
      True
      >>> human2bytes('2 Kb')
      2048
      >>> human2bytes('2.4 kb')
      2457
      >>> human2bytes('1.1 MB')
      1153433
      >>> human2bytes('1 Gb')
      1073741824
      >>> human2bytes('1 Tb')
      1099511627776

      >>> human2bytes('12 x')
      Traceback (most recent call last):
          ...
      ValueError: Cannot convert to float: '12 x'

    :param value: Human readable value to represent a size
    :type value: str
    :return: Integer value representation in bytes
    :rtype: int
    :raises TypeError: If other than string given
    :raises ValueError: If cannot parse the human-readable string
    """

    def _get_float(val):
        try:
            return float(val)
        except ValueError:
            raise ValueError('Cannot convert to float: {0}'.format(value))

    # Assume a 2-digit symbol was given
    try:
        sym = value[-2:].capitalize()
    except (TypeError, AttributeError):
        raise TypeError('Expected string, given: {0}.'.format(type(value)))

    if sym in SYM_NAMES:
        # size symbol is correct
        index = SYM_NAMES.index(sym)
        num = _get_float(value[:-2])
        return int(num * (1 << index * 10))

    # "Byte" special condition
    elif value[-4:].lower() == 'byte':
        return int(_get_float(value[:-4]))

    # incorrect or no symbol given, will try to parse float so will raise value error
    else:
        return int(_get_float(value))


def bytes2human(value, precision=2):
    """Converts integer byte values to human readable name. For example:

      >>> bytes2human(0.9 * 1024)
      '922 Byte'
      >>> bytes2human(0.99 * 1024)
      '1014 Byte'
      >>> bytes2human(0.999 * 1024)
      '1023 Byte'
      >>> bytes2human(1024)
      '1 Kb'
      >>> bytes2human(1024 + 512)
      '1.5 Kb'
      >>> bytes2human(85.70 * 1024 * 1024)
      '85.7 Mb'
      >>> bytes2human(28.926 * 1024 * 1024 * 1024)
      '28.93 Gb'

    This function does NOT check the argument type.
      >>> bytes2human('foo')
      Traceback (most recent call last):
          ...
      TypeError: type str doesn't define __round__ method

    :param value: Byte(s) value in integer.
    :type value: int or float
    :param precision: Floating precision of human-readable format (default 2).
    :type precision: int
    :return: Human representation of bytes
    :rtype: str
    :raises TypeError: If other than integer or float given
    :raises ValueError: If negative number is given.
    """
    try:
        byte_val = round(value)
    except TypeError as exc:
        raise exc
    else:
        if value < 0:
            raise ValueError('Given value cannot be negative: {0.real}'.format(value))

    # value is less than a kilobyte, so it's simply byte
    if byte_val < 1024:
        return '{0:d} Byte'.format(byte_val)

    # do reverse loop on size indexes
    for i in range(len(SYM_NAMES), 0, -1):
        index = i * 10
        size = byte_val >> index
        # not that big for this size index
        if size == 0:
            continue
        # maximum usable size found. add the decimal value as well.
        digit_in_bytes = size << index
        remaining = float(byte_val - digit_in_bytes) / (1 << index)
        size = round(size + remaining, precision)

        if size.is_integer():
            return '{0:d} {1}'.format(int(size), SYM_NAMES[i])
        else:
            return '{0.real} {1}'.format(size, SYM_NAMES[i])


def elide_text(string, max_len, pad=True):
    if len(string) <= max_len - 1:
        return string.rjust(max_len, ' ') if pad else string
    p = re.compile(r'^((?:[\ud800-\udbff][\udc00-\udfff]|.){' + str(max_len - 1) + '}).', re.UNICODE)
    m = p.match(string)
    if not m:
        return string.rjust(max_len, ' ')
    return re.sub(r'…*$', r'…', m.group(1))


def elide_file(string, max_len, pad=True):
    if len(string) <= max_len - 1:
        return string.rjust(max_len, ' ') if pad else string
    name, extension = os.path.splitext(string)
    if extension:
        max_len -= len(extension)
        string = name
    p = re.compile(r'^((?:[\ud800-\udbff][\udc00-\udfff]|.){' + str(max_len - 1) + '}).', re.UNICODE)
    m = p.match(string)
    if not m:
        return string + extension
    return re.sub(r'\.*$', r'.', m.group(1)) + extension

_marker = '...' + os.sep

def truncate_path(path, max_len):
    if len(path) <= max_len - 1:
        return path
    #print('>path', path)
    root, path = path[1:].split(os.sep, maxsplit=1)
    #print('<root', root, 'path', path)
    pattern = re.compile(r"/?.*?/")
    for i in range(1, path.count(os.sep)):
        new_path = pattern.sub(re.escape(_marker), path, i)
        if len(new_path) < max_len:
            return os.path.join('/', root, new_path)
    return os.path.join(os.sep, root, _marker, path.rsplit(os.sep, maxsplit=1)[1]) if os.sep in path else path


#
# SCAN AND DISPLAY
#

if len(sys.argv) < 2:
    print(sys.argv[0], 'Missing argument.')
    exit()

folder = sys.argv[1]
drive, folder = os.path.splitdrive(folder)
if drive and not folder:
    folder = os.path.join(drive, os.sep)

d = Dir(folder)

# table columns
col1 = 10
col2 = 40
col3 = 100

# tree prefix components:
space =  '    '
branch = '|   '
# pointers:
tee =    '|-- '
last =   '+-- '

def file_size(root, file):
    return bytes2human(os.stat(os.path.join(root, f)).st_size).rjust(col1, ' ')

def list_folder():
    for root, dirs, files in d.walk():
        for f in files:
            print(file_size(root, f), '|', elide_file(f, col2), '|', root)

def tree(paths: dict, prefix: str = ''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(paths) - 1) + [last]
    for pointer, path in zip(pointers, paths):
        yield prefix + pointer + path
        if isinstance(paths[path], dict): # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(paths[path], prefix=prefix+extension)

# build tree info
paths = {}
stats = { 'files': 0, 'folders': 0 }
for root, dirs, files in d.walk():
    stats['files'] += len(files)
    if root != folder: # exclude root folder
        stats['folders'] += 1
    if (files and dirs) or root == folder:
        node = paths
        path = root.replace(folder, '')
        if path:
            for p in path[1:].split(os.sep):
                if not p in node:
                    node[p] = {}
                node = node[p]
        for f in dirs:
            node[f] = {}

for root, dirs, files in d.walk():
    if root == folder:
        print('== ROOT FOLDERS ==')
        print('------------------')
        for f in dirs:
            print('<DIR>'.rjust(col1, ' '), '|', f)
        for f in files:
            print(file_size(root, f), '|', f)
        print('')
        print('%d files in %d folders' % (stats['files'], stats['folders']))
        print('')
        print('== FOLDERS TREE ==')
        print('------------------')
        print('o')
        for line in tree(paths):
            print(line)
        print('')
        print('== FILES REPORT ==')
        print('------------------')

    for f in files:
        print(file_size(root, f), '|', elide_file(f, col2), '|', posix_path(truncate_path(root.replace(drive, ''), col3).replace(folder, '')))
