/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  * $Created by: HU Chi (huchinlp@foxmail.com) 2020.01.02
  */

#include "SLTKEmbedding.h"
#include "../../tensor/core/CHeader.h"

using namespace std;

/*
load pre-trained embeddings from file
>>> fn - the pre-trained embeddings file
>>> embSize - the embedding dimension
>>> myDevID - the device id
*/
void Embedding::LoadWordEmbeddings(const Vocab& textVocab, const string& fn, int myDevID)
{
    devID = myDevID;

    /* load embeddings vocab */
    embVocab = make_shared<Vocab>((fn+".vocab").c_str());
    
    /* load embeddings */
    FILE* embFile = fopen(fn.c_str(), "rb");
    fread(&vocabSize, sizeof(vocabSize), 1, embFile);
    fread(&embSize, sizeof(embSize), 1, embFile);

    vector<float*> buffers;
    for (int i = 0; i < vocabSize; i++) {
        float* buffer = new float[embSize];
        buffers.push_back(buffer);
        fread(buffer, sizeof(float), embSize, embFile);
    }

    /* trim and reorder embeddings */
    vector<float*> newBuffers;
    embVocab = make_shared<Vocab>(textVocab, embVocab);
    int newIdx = 1;
    for (auto& p : embVocab->word2id) {
        newBuffers.push_back(buffers[p.second]);
        p.second = newIdx++;
    }

    /* combine embeddings */
    vocabSize = embVocab->word2id.size();
    float* data = new float[vocabSize];

    for (int i = 0; i < vocabSize;i++) {
        copy(newBuffers[i], newBuffers[i] + embSize, data + i * embSize);
    }
    InitTensor2DV2(&vecs, vocabSize, embSize, X_FLOAT, devID);
    vecs.SetData(data, vecs.unitNum);

    /* free buffers */
    for (auto p : buffers) {
        delete[] p;
    }
    delete[] data;
}

/*
set embeddings for a mini-batch
>>> input - the input tensor
*/
XTensor Embedding::Embed(const XTensor& input)
{
    return Gather(vecs, input);
}

/* get embeddings */
XTensor StackEmbedding::Embed(const XTensor& input)
{
    TensorList list;
    vector<XTensor> embeddings;
    for (auto emb : staticEmbeddings) {
        embeddings.push_back(emb->Embed(input));
        list.Add(&(embeddings.back()));
    }
    return Concatenate(list, embeddings.back().GetDim(-1));
}
