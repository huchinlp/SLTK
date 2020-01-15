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

#include <memory>
#include "SLTKCRF.h"
#include "SLTKLSTMCell.h"
#include "SLTKEmbedding.h"
#include "SLTKDataSet.h"
#include "../../model/Model.h"
#include "../../tensor/XGlobal.h"

using namespace nts;
using namespace std;

/* linear model */
struct Lin :public Model
{
    /* constructor */
    Lin(int inputDim, int outputDim);

    /* forward function */
    XTensor Forward(const XTensor& input);
};

/* the sequence labeling model */
struct SequenceTagger : Model
{
private:

    /* dropout rate for input and output */
    float dropout;

    /* dropout rate for input, drop along the legth axis */
    float wordDropout;

    /* dropout rate for input and output, drop along the embedding axis */
    float lockedDropout;

    /* embedding */
    shared_ptr<StackEmbedding> embedding;

    /* embeddings to input */
    shared_ptr<Lin> embedding2NN;

    /* RNNs */
    shared_ptr<LSTM> rnns;

    /* RNN output to tag */
    shared_ptr<Lin> rnn2tag;

    /* CRF layer */
    shared_ptr<CRF> crf;

public:

    /* forward function */
    XTensor Forward(const vector<vector<string>>& input);

    /* get the mask of sentences */
    XTensor GetMask(const vector<vector<string>>& input);

    /* predict tags */
    vector<vector<int>> Predict(const vector<vector<string>>& input);

    /* dump input sequences and label sequences to a file */
    void DumpResult(vector<vector<string>>& src, vector<vector<string>>& tgt, const char* file);

    /* constructor */
    explicit SequenceTagger(int myDevID, int rnnLayer, int hiddenSize, 
                            int tagNum, int embSize, shared_ptr<StackEmbedding> myEmbedding,
                            float myDropout = 0.0f, float myWordropout = 0.0f, float myLockedropout = 0.0f);

    /* de-constructor */
    ~SequenceTagger();
};