A very simple framework for state-of-the-art sequence labeling.

## Quick Start

### Requirements and Installation

The project is based on NiuTensor, [refer to this for the requirements](https://github.com/niutrans/niutensor).

Build:
```command
make clean && make -j
```

If successful, you will get an executable file 'NiuTensor' in the 'bin' dir.

## Example Usage
At present (2020-1-18), SLTK only supports inference using the pre-trained models in PyTorch.
We use flair as our front tools for training, [refer to this for flair](https://github.com/flairNLP/flair).

Once you got a pre-trained sequence labeling model (e.g., a WNUT17 NER model) from flair, you can convert it to a NiuTensor model:
```command
python pack_model.py -task wnut17 -src wnut17/best-model.pt -tgt wnut17/wnut17.model
```

Then prepare the pre-trained embeddings and vocabularies:
```command
python get_embeddings.py -task wnut17 -file wnut17/best-model.pt
```

Now tag your text:
```command
./bin/NiuTensor.GPU -devID 0 -batchSize 24 -src test.txt -tgt res.txt
-tagNum 29 -embSize 400 -rnnLayer 1 -hiddenSize 256 -embFile wnut17.emb -tagVocab wnut17.tag.vocab -modelFile wnut17.model -emb1 wnut17crawl.emb -emb2 wnut17twitter.emb
```

Refer to src/Main.cpp for more details of parameters.