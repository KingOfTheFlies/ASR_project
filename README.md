<h1 align="center">Automatic Speech Recognition (ASR)</h1>

## About

The model dased on [Deep Speech 2](https://arxiv.org/pdf/1512.02595).

See [wandb report](https://wandb.ai/dungeon_as_fate/pytorch_template_asr_example).

## Final Result:
```angular2html
                WER     CER
test-clean     35,4     12.5
```


## Installation

0. Create new conda environment:
```bash
conda create -n ASR_project python=3.10

conda activate ASR_project
``` 

1. Install all requirements.
```bash
pip install -r requirements.txt
```

2. Download model checkpoint.
```bash
gdown --folder https://drive.google.com/drive/folders/1j32oNiZH4ffMdb_rVbiQRzKTiE9HUYH5?usp=sharing
```

## Inference
```bash
python inference.py inferencer.from_pretrained="./checkpoints/bpe_rnn_only_models_big/model_best.pth" \
          text_encoder.use_bpe=True \
          datasets.inference.audio_dir=<dir_with_audios> \
          datasets.inference.transcription_dir=<dir_with_transcriptions> \
          inferencer.device=<device> \
          inferencer.save_path=<name_of_folder_for_predicted_texts> \
          text_encoder.beam_size=<beam_size>
```

## Train

training with ByteLevelBPETokenizer if text_encoder.use_bpe=True (for alphabet - False):

```bash
python train.py text_encoder.bpe_vocab_size=1024 \
        trainer.device=cuda trainer.override=True \
        writer.log_checkpoints=True \
        writer.run_name=bpe_rnn_only_models_big \
        text_encoder.use_bpe=True \
        datasets.train.part="train-clean-100" \
        model.use_conv=False \
```


