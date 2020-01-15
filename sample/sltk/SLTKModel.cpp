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
  * $Created by: HU Chi (huchinlp@foxmail.com) 2019-12-31
  * happy coding 2020~
  */

#include "SLTKModel.h"
#include "StringUtil.h"
#include "../../tensor/core/CHeader.h"

XTensor SequenceTagger::Forward(const vector<vector<string>>& sentences)
{
    auto input = embedding2NN->Forward(embedding->Embed(sentences));

    auto rnnOutput = rnns->Forward(input);

    return rnn2tag->Forward(rnnOutput);
}

/* 
get the mask of sentences 
>>> input - the input sentences
*/
XTensor SequenceTagger::GetMask(const vector<vector<string>>& input)
{
    XTensor mask;
    int bsz = input.size();

    int maxLen = 0;
    for (const auto sent : input) {
        maxLen = max(maxLen, sent.size());
    }
    InitTensor2DV2(&mask, bsz, maxLen, X_INT, devID);

    int* indices = new int[bsz * maxLen];
    memset(indices, 0, bsz * maxLen * sizeof(int));

    int i = 0;
    int sentID = 0;
    for (const auto sent : input) {
        for (const auto token : sent)
            indices[i++] = 1;
        i = maxLen * sentID++;
    }
    mask.SetData(indices, mask.unitNum);
    delete[] indices;

    return mask;
}

/*
predict tags
>>> input - the input sentences
*/
vector<vector<int>> SequenceTagger::Predict(const vector<vector<string>>& input)
{
    auto mask = GetMask(input);
    auto features = Forward(input);
    return crf->Decode(features, mask);
}

/* dump input sequences and label sequences to a file */
void SequenceTagger::DumpResult(vector<vector<string>>& src, vector<vector<string>>& tgt, const char* file)
{
}

/*
costructor
>>> rnnLayer - number of rnn layers
>>> hiddenSize - hidden dim for rnn
>>> tagNum - number of tags
>>> embSize - embedding dim
>>> myEmbedding - the embedding module
>>> myDropout - drop out rate
>>> myWordDropout - drop out rate
*/
SequenceTagger::SequenceTagger(int myDevID, int rnnLayer, int hiddenSize, 
                               int tagNum, int embSize, shared_ptr<StackEmbedding> myEmbedding,
                               float myDropout, float myWordropout, float myLockedropout)
{
    devID = myDevID;

    embedding = myEmbedding;
    crf = make_shared<CRF>(tagNum);
    embedding2NN = make_shared<Lin>(embSize, embSize);
    rnn2tag = make_shared<Lin>(hiddenSize * 2, tagNum+1);
    rnns = make_shared<LSTM>(embSize, hiddenSize, rnnLayer, true);

    const auto prefix = "SequenceTagger.";
    Register(ConcatString(prefix, "CRF"), *crf);
    Register(ConcatString(prefix, "RNN"), *rnns);
    Register(ConcatString(prefix, "RNN2Tag"), *rnn2tag);
    Register(ConcatString(prefix, "Embedding2NN"), *embedding2NN);
}

/* de-constructor */
SequenceTagger::~SequenceTagger()
{
}

/* constructor */
Lin::Lin(int inputDim, int outputDim)
{
    Register("Bias", { outputDim }, X_FLOAT);
    Register("Weight", { inputDim, outputDim }, X_FLOAT);
}

/* forward function */
XTensor Lin::Forward(const XTensor& input)
{
    return MatrixMul(input, *Get("Weight")) + *Get("Bias");
}