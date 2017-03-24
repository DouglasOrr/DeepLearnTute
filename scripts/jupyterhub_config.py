# flake8: noqa

c.Spawner.env_keep = [
    'PATH', 'PYTHONPATH', 'LANG', 'LC_ALL',
    'CONDA_ROOT', 'CONDA_DEFAULT_ENV', 'VIRTUAL_ENV',
    'OPENBLAS_NUM_THREADS',
]
c.Authenticator.admin_users = {'admin'}
