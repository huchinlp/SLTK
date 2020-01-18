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

#include <fstream>
#include "SLTKModel.h"
#include "StringUtil.h"
#include "../../tensor/core/CHeader.h"

XTensor SequenceTagger::Forward(const vector<vector<string>>& sentences)
{
    auto input = embedding2NN->Forward(embedding->Embed(sentences));

    auto rnnOutput = rnns->Forward(input);

    auto tags = rnn2tag->Forward(rnnOutput);

    return tags;
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
    for (const auto sent : input)
        maxLen = max(maxLen, sent.size());
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
void SequenceTagger::DumpResult(vector<vector<string>>& src, vector<vector<int>>& tgt, const char* file)
{
    ofstream f(file, ios::app);
    ostringstream buffer;
    for (int i = 0; i < src.size(); i++) {
        for (int j = 0; j < src[i].size(); j++)
            f << src[i][j] << "\t" << tagVocab->id2word[tgt[i][j]] << "\n";
        f << "\n";
    }
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
                               int tagNum, int embSize, shared_ptr<StackEmbedding> myEmbedding, const char* tagVocabF,
                               float myDropout, float myWordropout, float myLockedropout)
{
    devID = myDevID;

    embedding = myEmbedding;
    crf = make_shared<CRF>(tagNum);
    tagVocab = make_shared<Vocab>();
    tagVocab->Load(tagVocabF);
    embedding2NN = make_shared<Lin>(embSize, embSize);
    rnn2tag = make_shared<Lin>(hiddenSize * 2, tagNum);
    rnns = make_shared<LSTM>(embSize, hiddenSize, rnnLayer, true);

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
XTensor Lin::Forward(const XTensor& input)
{
    return  MatrixMul(input, *Get("Weight")) + *Get("Bias");
}