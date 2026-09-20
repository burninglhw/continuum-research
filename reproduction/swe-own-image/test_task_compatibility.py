import pytest

from configure_image import constrain_environment, task_compatibility_commands, xarray_environment_constraints


def test_seaborn_012_pandas20_gets_only_a_dependency_override():
    recipe = {'repo': 'mwaskom/seaborn', 'version': '0.12',
              'official_spec': {'pip_packages': ['pandas==2.0.0', 'numpy==1.25.2']}}
    assert task_compatibility_commands(recipe) == ['python -m pip install --no-deps pandas==1.5.3']
    assert recipe['official_spec']['pip_packages'] == ['pandas==2.0.0', 'numpy==1.25.2']


@pytest.mark.parametrize('repo,version,packages', [
    ('mwaskom/seaborn', '0.13', ['pandas==2.0.0']),
    ('mwaskom/seaborn', '0.12', ['pandas==1.5.3']),
    ('mwaskom/seaborn', '0.12', []),
    ('other/repository', '0.12', ['pandas==2.0.0']),
])
def test_other_recipes_do_not_change(repo, version, packages):
    assert task_compatibility_commands({'repo': repo, 'version': version,
                                        'official_spec': {'pip_packages': packages}}) == []


def test_xarray_pins_apply_during_conda_solve_without_mutating_recipe():
    recipe = {'repo': 'pydata/xarray', 'version': '2022.06', 'official_spec': {
        'packages': 'environment.yml',
        'pip_packages': ['numpy==1.23.0', 'pandas==1.5.3', 'dask==2022.8.1', 'pytest==7.4.0'],
    }}
    constraints = xarray_environment_constraints(recipe)
    command = 'name: testbed\ndependencies:\n  - numpy\n  - pandas\n  - dask-core\n  - distributed\n  - pytest\n  - rasterio\n  - pip:\n      - numbagg'
    result = constrain_environment(command, recipe, constraints)
    for dependency in ['numpy=1.23.0', 'pandas=1.5.3', 'dask=2022.8.1', 'distributed=2022.8.1',
                       'pytest=7.4.0', 'xarray=2022.3.0', 'wheel=0.45.1', 'rasterio=1.3.9']:
        assert '  - ' + dependency + '\n' in result
    assert '  - dask-core\n' in result
    assert '      - numbagg' in result
    assert recipe['official_spec']['pip_packages'] == ['numpy==1.23.0', 'pandas==1.5.3', 'dask==2022.8.1', 'pytest==7.4.0']


@pytest.mark.parametrize('repo,version,packages,dask', [
    ('pydata/xarray', '2022.07', 'environment.yml', 'dask==2022.8.1'),
    ('pydata/xarray', '2022.06', 'requirements.txt', 'dask==2022.8.1'),
    ('pydata/xarray', '2022.06', 'environment.yml', 'dask==2023.1.0'),
    ('other/repository', '2022.06', 'environment.yml', 'dask==2022.8.1'),
])
def test_xarray_constraints_do_not_affect_other_recipes(repo, version, packages, dask):
    assert xarray_environment_constraints({'repo': repo, 'version': version,
        'official_spec': {'packages': packages, 'pip_packages': [dask]}}) is None
