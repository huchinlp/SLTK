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

#include "../../tensor/core/CHeader.h"
#include "SLTKModel.h"
#include "StringUtil.h"

XTensor SequenceTagger::Forward(XTensor& sentences)
{
    XTensor hidden;
    XTensor memory;

    auto input = embedding2NN->Forward(embedding->Embed(sentences));

    auto rnnOutput = rnns->Forward(input, hidden, memory);

    return rnn2tag->Forward(rnnOutput);
}

/*
predict tags
>>> input - (bsz, len)
>>> mask - (bsz, len)
<<< tags - (bsz, len)
*/
vector<vector<int>> SequenceTagger::Predict(XTensor& input, XTensor& mask)
{
    auto features = Forward(input);
    return crf->Decode(features, mask);
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
SequenceTagger::SequenceTagger(int rnnLayer, int hiddenSize, int tagNum, int embSize, Embedding* myEmbedding,
                               float myDropout, float myWordropout, float myLockedropout)
{
    embedding = myEmbedding;

    crf = make_shared<CRF>(tagNum);

    rnns = make_shared<LSTM>(embSize, hiddenSize, rnnLayer, true);

    embedding2NN = make_shared<Lin>(embSize, embSize);

    rnn2tag = make_shared<Lin>(hiddenSize * 2, tagNum);

    const auto prefix = "SequenceTagger.";
    Register(ConcatString(prefix, "CRF"), *crf);
    Register(ConcatString(prefix, "Embedding2NN"), *embedding2NN);
    Register(ConcatString(prefix, "RNN"), *rnns);
    Register(ConcatString(prefix, "RNN2Tag"), *rnn2tag);
}

/* de-constructor */
SequenceTagger::~SequenceTagger()
{
}

/* constructor */
Lin::Lin(int inputDim, int outputDim)
{
    Register("Weight", { inputDim, outputDim }, X_FLOAT);
    Register("Bias", { outputDim }, X_FLOAT);
}

/* forward function */
XTensor Lin::Forward(XTensor& input)
{
    return MatrixMul(input, Get("Weight")) + Get("Bias");
}