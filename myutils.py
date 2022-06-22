from importlib_metadata import version
from IPython.display import Markdown
import warnings

def get_version(lib): 
    "Returns version of `lib`, can be either a `str` representation or the module itself"
    if isinstance(lib, str): return version(lib)
    else:
        try:
            return lib.__version__
        except:
            raise TypeError('`lib` should either be the string name of the module or the module itself')


def state_versions(*libs):
    "State all the versions currently installed from `libs` in Markdown"
    cell = f"""
"""
    if isinstance(libs[0], list):
        warnings.warn('''Passing in a list of libraries will be deprecated, you should pass them directly such as:
        `state_versions(fastai, fastcore)`
        or:
        `state_versions('fastai', 'fastcore')`
        ''', category=DeprecationWarning)
        libs = libs[0]
    cell += 'Below are the versions of '
    for i, lib in enumerate(libs):
        nm = lib if isinstance(lib, str) else lib.__name__
        if len(libs) == 1: cell += f'`{nm}`'
        elif i < len(libs)-1: 
            cell += ''.join(f'`{nm}`')
            if len(libs) > 2: cell += ', '
            else: cell += ' '
        elif len(libs) > 1: cell += ''.join(f'and `{nm}`')
    cell += ' currently running at the time of writing this:\n'
    for lib in libs:
        nm = lib if isinstance(lib, str) else lib.__name__
        cell += f'* `{nm}` : {get_version(lib)} \n'
    cell += '---'
    return Markdown(cell)

