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
  * $Created by: HU Chi (huchinlp@foxmail.com)
  */

#pragma once

#include <initializer_list>
#include "SLTKDataSet.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/XGlobal.h"


using namespace nts;
using namespace std;

struct Embedding
{
    /* device id */
    int devID;

    /* the vocab size */
    size_t vocabSize;

    /* the embedding dimension */
    size_t embSize;

    /* the vocabulary of the embeddings */
    Vocab embVocab;

    /* the pre-trained word embeddings */
    XTensor vec;

    /* constructor */
    explicit Embedding(int myDevID, const char* embFile);

    /* load embeddings from files */
    void LoadWordEmbedding(const char* file);

    /* set word embeddings for a batch of sentences */
    XTensor Embed(const vector<vector<string>>& input);
};

struct LanguageModel
{
    /* device id */
    int devID;

    /* input size */
    int inSize;

    /* output size */
    int outSize;

    /* constructor */
    LanguageModel(int myDevID, const char* file);

    /* load pretrained LM from files */
    void LoadPretrainedLM(const char* file);

    /* get the contextual representation of inputs */
    void GetRepresentation(const vector<vector<string>>& input);
};

struct StackEmbedding
{
public:
    int devID;

    /* stack of multiple embeddings */
    vector<Embedding*> staticEmbeddings;

    /* get embeddings of inputs */
    XTensor Embed(const vector<vector<string>>& input);

    /* constructor */
    StackEmbedding(int myDevID, vector<const char*> files);

    /* de-constructor */
    ~StackEmbedding();
};