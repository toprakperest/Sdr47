import pytest
from modules.sdr_receiver import SDRReceiver
from unittest.mock import MagicMock

@pytest.fixture
def mock_sdr():
    sdr = SDRReceiver({'test_mode': True})
    sdr._setup_sdr = MagicMock()
    return sdr

def test_sdr_initialization(mock_sdr):
    assert mock_sdr.test_mode is True
    mock_sdr._setup_sdr.assert_called_once()

def test_sdr_data_processing(mock_sdr):
    test_data = {'samples': [1, 2, 3], 'freq': 1e9}
    processed = mock_sdr._process_data(test_data)
    assert 'processed' in processed