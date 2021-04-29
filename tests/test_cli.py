import pytest
from click.testing import CliRunner

from bn3d.cli import cli


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
def test_cli(arguments, runner):
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

    def test_ls_noise(self, runner):
        result = runner.invoke(cli, ['ls', 'noise'])
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
