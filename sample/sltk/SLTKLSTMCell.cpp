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

#include <memory>
#include <algorithm>
#include "StringUtil.h"
#include "SLTKLSTMCell.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/function/FHeader.h"

/*
constructor
>>> inputDim - the input size of lstm
>>> hiddenDim - the hidden size of lstm
>>> layerNum - the number of layers
>>> bidirectional - use bi-lstm or not
*/
LSTM::LSTM(int myInputDim, int myHiddenDim, int myLayerNum, bool myBidirectional)
{
    /* register parameters */
    inputDim = myInputDim;
    hiddenDim = myHiddenDim;
    layerNum = myLayerNum;
    bidirectional = myBidirectional;

    int numPerLayer = bidirectional ? 2 : 1;

    for (int i = 0; i < layerNum * numPerLayer; i++) {
        auto cell = make_shared<LSTMCell>(inputDim, hiddenDim, i);
        cells.push_back(cell);
        Register("LSTMCell", *cell);
    }
}

/* generate a range of number */
vector<int> GetRange(int size, bool isReversed)
{
    vector<int> range;
    for (int i = 0; i < size; i++)
        range.push_back(i);
    if (isReversed)
        reverse(range.begin(), range.end());
    return range;
}

/*
lstm forward function
>>> input - (batchSize, maxLen, inputDim)
>>> hidden - (batchSize, hiddenDim)
>>> memory - (batchSize, hiddenDim)
<<< hiddens - (batchSize, maxLen, hiddenDim (x2 if bidirectional is true))
*/
XTensor LSTM::Forward(const XTensor& input, XTensor& hidden, XTensor& memory)
{
    TensorList list;
    Split(input, 1, input.GetDim(1));

    int bsz = input.GetDim(0);
    int maxLen = input.GetDim(1);

    XTensor fwdHiddens;
    XTensor bwdHiddens;
    InitTensor3DV2(&fwdHiddens, bsz, maxLen, hiddenDim, X_FLOAT, input.devID);
    if (bidirectional) {
        InitTensor3DV2(&bwdHiddens, bsz, maxLen, hiddenDim, X_FLOAT, input.devID);
    }

    bool isReversed = false;

    /* iteration of layers */
    for (int i = 0; i < cells.size(); i++) {
        XTensor* hiddens = &fwdHiddens;
        isReversed = ((i % 2 == 0) && bidirectional) ? true : false;
        if (isReversed) {
            hiddens = &bwdHiddens;
        }
        auto range = GetRange(input.GetDim(1), isReversed);

        /* iteration of timesteps */
        for (int idx : range) {
            cells[i]->Forward(*list[idx], hidden, memory, i);
            int srcIdx[]{ 0 };
            int tgtIdx[]{ idx };

            /* collect hidden states of the last layer */
            if (i == cells.size() - 1 || (bidirectional && i == cells.size() - 2)) {
                _CopyIndexed(&hidden, hiddens, 1, srcIdx, 1, tgtIdx);
            }
        }
    }
    if (bidirectional)
        return Concatenate(fwdHiddens, bwdHiddens, 2);
    else
        return fwdHiddens;
}

/*
constructor
>>> inputDim - the input size of lstm
>>> hiddenDim - the hidden size of lstm
>>> number - index of this layer
*/
LSTMCell::LSTMCell(int inputDim, int hiddenDim, int index)
{
    /* register parameters */
    Register(ConcatString("Weight_H_", index), { inputDim + hiddenDim, hiddenDim * 4 }, X_FLOAT);
    Register(ConcatString("Bias_H_", index), { hiddenDim }, X_FLOAT);
    Register(ConcatString("Bias_F_", index), { hiddenDim }, X_FLOAT);
}

/*
lstm cell forward
>>> x - input, (batchSize, inputDim)
>>> h - hidden state, (batchSize, hiddenDim)
>>> c - memory state, (batchSize, hiddenDim)
>>> index - the index of lstm parameters
*/
void LSTMCell::Forward(const XTensor& x, XTensor& h, XTensor& c, int index)
{
    auto weight = Get(ConcatString("Weight_H_", index));
    auto biasH = Get(ConcatString("Bias_H_", index));
    auto biasF = Get(ConcatString("Bias_F_", index));

    /* combine x and h before the transformation */
    XTensor xh = Concatenate(x, h, x.order - 1);
    XTensor noGated = MatrixMul(xh, weight);
    noGated = noGated + biasH;

    /* split the big tensor to 4 parts */
    TensorList splited(4);
    Split(noGated, splited, noGated.order - 1, 4);

    XTensor* i = splited[0];
    XTensor* j = splited[0];
    XTensor* f = splited[0];
    XTensor* o = splited[0];

    /* apply gating to the transformed tensor */
    XTensor g = HardTanH(*j);
    XTensor gatedin = Sigmoid(*i) * g;
    XTensor memory = h * Sigmoid(*f + biasF);

    /* update memory state and hidden state */
    c = memory + gatedin;
    h = HardTanH(c) * Sigmoid(*o);
}