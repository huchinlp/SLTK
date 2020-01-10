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
void Embedding::LoadWordEmbeddings(const char* fn, int myDevID)
{+
    devID = myDevID;
    FILE* f = fopen(fn, "rb");

    fread(&vocabSize, sizeof(vocabSize), 1, f);
    fread(&embSize, sizeof(embSize), 1, f);

    InitTensor2DV2(&vecs, vocabSize, embSize, X_FLOAT, -1);
    fread(vecs.data, sizeof(float), vecs.unitNum, f);
    vecs.SetDevice(devID);
}

/*
set embeddings for a mini-batch
>>> input - the input tensor
*/
XTensor Embedding::Embed(const XTensor& input)
{
    return Gather(vecs, input);
}