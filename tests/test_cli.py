import numpy as np
import pytest
from click.testing import CliRunner

from panqec.cli import cli, read_bias_ratios, read_range_input


@pytest.fixture
def runner():
    """Click CliRunner with isolated file system."""
    _runner = CliRunner()
    with _runner.isolated_filesystem():
        yield _runner
    assert hasattr(_runner, 'invoke')


@pytest.mark.parametrize('arguments', [
    [],
    ['--help'],
    ['--version'],
])
def test_cli_basic(arguments, runner):
    result = runner.invoke(cli, arguments)
    assert result.exit_code == 0


class TestLS:

    def test_ls_all(self, runner):
        result = runner.invoke(cli, ['ls'])
        assert result.exit_code == 0
        assert 'Codes:' in result.output
        assert 'Error Models (Noise):' in result.output
        assert 'Decoders:' in result.output

    def test_ls_codes(self, runner):
        result = runner.invoke(cli, ['ls', 'codes'])
        assert result.exit_code == 0
        assert 'Codes:' in result.output
        assert 'Error Models (Noise):' not in result.output
        assert 'Decoders:' not in result.output

    def test_ls_error_models(self, runner):
        result = runner.invoke(cli, ['ls', 'error_models'])
        assert result.exit_code == 0
        assert 'Codes:' not in result.output
        assert 'Error Models (Noise):' in result.output
        assert 'Decoders:' not in result.output

    def test_ls_decoders(self, runner):
        result = runner.invoke(cli, ['ls', 'decoders'])
        assert result.exit_code == 0
        assert 'Codes:' not in result.output
        assert 'Error Models (Noise):' not in result.output
        assert 'Decoders:' in result.output


def test_read_bias_ratios():
    expected_bias_ratios = [0.5, 1, 3, 10, 30, 100, np.inf]
    eta_string = '0.5,1,3,10,30,100,inf'
    bias_ratios = read_bias_ratios(eta_string)
    assert len(bias_ratios) == len(expected_bias_ratios)
    for eta, expected_eta in zip(bias_ratios, expected_bias_ratios):
        assert eta == expected_eta
        assert type(eta) == type(expected_eta)


@pytest.mark.parametrize('spec,expected_values', [
    ('0:0.6:0.005', np.arange(0, 0.605, 0.005).tolist()),
    ('1,2,3', [1.0, 2.0, 3.0]),
    ('13.21', [13.21]),
    ('1e-2', [0.01]),
])
def test_read_range_input(spec, expected_values):
    values = read_range_input(spec)
    assert len(values) == len(expected_values)
    for value, expected_value in zip(values, expected_values):
        assert value == expected_value
        assert type(value) == type(expected_value)
