from cgitb import lookup
import pytest
from pydub.generators import WhiteNoise
from lookup.doc_utils import Document
from lookup.wav2vec import DataCollatorCTCWithPadding, get_data_collator, load_trainer, load_w2v, w2v_data_loader
from transformers import Wav2Vec2ForCTC
from transformers import Trainer
from torch import is_tensor

def test_w2v_data_loader(tmp_path, mocker):
    uris = []
    for i in range(3):
        file_path = tmp_path / f"{i}.wav"
        uris.append(str(file_path))
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1)
        audio_segment.export(file_path, format='wav')
    labels = ['one thing', 'another', 'final']
    mock_doc_array = lambda uris, n_dim: [Document(uri=i, tags={'label': labels[ix]}) for ix, i in enumerate(uris)]
    mock = mocker.patch('lookup.dataset._get_array_by_uris', mock_doc_array)
    dataset, _ = w2v_data_loader(uris, 512, 8000, 16000)
    for d in dataset:
        assert len(d['labels']) == len(d['text'])
        assert d['input_values'].shape == d['audio']['array'].shape


def test_load_w2v(mocker):
    mock = mocker.patch('lookup.wav2vec.Wav2Vec2Processor')
    mock.tokenizer.pad_token_id = 1
    model = load_w2v(mock)
    assert isinstance(model, Wav2Vec2ForCTC)

def test_get_data_collator(mocker):
    mock = mocker.patch('lookup.wav2vec.Wav2Vec2Processor')
    data_collator = get_data_collator(mock)
    assert isinstance(data_collator, DataCollatorCTCWithPadding)

def test_load_trainer(mocker):
    mock_processor = mocker.patch('lookup.wav2vec.Wav2Vec2Processor')
    mock_processor.tokenizer.pad_token_id = 1
    mock_model = load_w2v(mock_processor)
    mock_collator = mocker.patch('lookup.wav2vec.DataCollatorCTCWithPadding')
    mock_data = mocker.patch('lookup.dataset')
    trainer = load_trainer(mock_model, mock_collator, mock_data, mock_data, mock_processor)
    assert isinstance(trainer, Trainer)

def test_training_round(tmp_path, mocker):
    file_path = tmp_path / "1.wav"
    uris = [file_path]*32
    audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1000)
    audio_segment.export(file_path, format='wav')
    labels = ['one thing']*32
    mock_doc_array = lambda uris, n_dim: [Document(uri=i, tags={'label': labels[ix]}) for ix, i in enumerate(uris)]
    mock = mocker.patch('lookup.dataset._get_array_by_uris', mock_doc_array)
    dataset, processor = w2v_data_loader(uris, 512, 8000, 16000)
    model = load_w2v(processor)
    data_collator = get_data_collator(processor)
    trainer = load_trainer(model, data_collator, dataset, dataset, processor, epochs=1,output_dir = tmp_path / 'checkpoint')
    trainer.train()
    assert len(trainer.state.log_history) == 1
    assert trainer.state.log_history[0]['epoch'] == 1