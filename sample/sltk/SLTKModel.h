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

#ifndef __BLSTM_CRF_MODEL_H__
#define __BLSTM_CRF_MODEL_H__

#include "SLTKCRF.h"
#include "SLTKLSTMCell.h"
#include "SLTKEmbedding.h"
#include "SLTKDataSet.h"
#include "../../model/Model.h"
#include "../../tensor/XGlobal.h"

using namespace nts;
using namespace std;
#undef Linear

/* the sequence labeling model */
struct SequenceTagger: Model
{
    /* dropout rate for input and output */
    float dropout;

    /* dropout rate for input, drop along the legth axis */
    float wordDropout;

    /* dropout rate for input and output, drop along the embedding axis */
    float lockedDropout;

    /* embedding */
    Embedding* embedding;

    /* embeddings to input */
    shared_ptr<Linear> embedding2NN;

    /* RNNs */
    shared_ptr<LSTM> rnns;

    /* RNN output to tag */
    shared_ptr<Linear> rnn2tag;

    /* CRF layer */
    shared_ptr<CRF> crf;

    /* vocab for input and tags */
    Vocab* vocab;

    /* forward function */
    XTensor Forward(XTensor& input);

    /* predict tags */
    vector<vector<int>> Predict(XTensor& input, XTensor& mask);

    /* constructor */
    explicit SequenceTagger(int rnnLayer, int hiddenSize,
                            Vocab* myVocab, Embedding* myEmbedding,
                            float myDropout, float myWordropout, float myLockedropout);

    /* de-constructor */
    ~SequenceTagger();
};

/* linear model */
struct Linear :public Model
{
    /* constructor */
    Linear(int inputDim, int outputDim);

    /* forward function */
    XTensor Forward(XTensor& input);
};

#endif // __BLSTM_CRF_MODEL_H__