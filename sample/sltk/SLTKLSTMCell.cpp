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

/* split a big tensor into small ones */
void Split(const XTensor& big, vector<XTensor*>& list, int dim)
{
    TensorList l;
    int num = big.GetDim(dim);
    int* dimSize = new int[big.order - 1];
    for (int i = 0; i < dim; i++)
        dimSize[i] = big.dimSize[i];
    for (int i = 0; i < big.order - dim - 1; i++)
        dimSize[dim + i] = big.dimSize[dim + i + 1];

    for (int i = 0; i < num; i++) {
        l.Add(NewTensorV2(big.order - 1, dimSize, big.dataType, big.denseRatio, big.devID));
        list.push_back(l[i]);
    }
    Split(big, l, dim, num);
}

/*
lstm forward function
>>> input - (batchSize, maxLen, inputDim)
<<< hiddens - (batchSize, maxLen, hiddenDim (x2 if bidirectional is true))
*/
XTensor LSTM::Forward(const XTensor& input)
{
    vector<XTensor*> timeSteps;
    Split(input, timeSteps, 1);

    bool isReversed = false;
    int bsz = input.GetDim(0);
    int maxLen = input.GetDim(1);

    /* input of a lstm */
    XTensor fwdHidden, bwdHidden;
    XTensor fwdMemory, bwdMemory;

    /* output of a lstm */
    XTensor fwdHiddens, bwdHiddens;

    /* initialize the hidden and memory */
    InitTensor2DV2(&fwdHidden, bsz, hiddenDim, X_FLOAT, input.devID);
    InitTensor2DV2(&fwdMemory, bsz, hiddenDim, X_FLOAT, input.devID);
    InitTensor3DV2(&fwdHiddens, bsz, maxLen, hiddenDim, X_FLOAT, input.devID);
    fwdHidden.SetZeroAll();
    fwdMemory.SetZeroAll();

    if (bidirectional) {
        InitTensor2DV2(&bwdHidden, bsz, hiddenDim, X_FLOAT, input.devID);
        InitTensor2DV2(&bwdMemory, bsz, hiddenDim, X_FLOAT, input.devID);
        InitTensor3DV2(&bwdHiddens, bsz, maxLen, hiddenDim, X_FLOAT, input.devID);
        bwdHidden.SetZeroAll();
        bwdMemory.SetZeroAll();
    }

    /* iteration of layers */
    for (int i = 0; i < cells.size(); i++) {
        XTensor* hidden = &fwdHidden;
        XTensor* memory = &fwdMemory;
        XTensor* hiddens = &fwdHiddens;
        isReversed = ((i % 2 == 0) && bidirectional) ? true : false;
        if (isReversed)
        {
            hidden = &bwdHidden;
            memory = &bwdMemory;
            hiddens = &bwdHiddens;
        }
        auto range = GetRange(input.GetDim(1), isReversed);

        /* iteration of timesteps */
        for (int idx = 0; idx < range.size(); idx++) {
            cells[i]->Forward(*timeSteps[idx], *hidden, *memory, i);

            /* collect hidden states of the last layer */
            if (i == (cells.size() - 1) || (bidirectional && i == cells.size() - 2)) {
                int srcIdx[]{ 0 };
                int tgtIdx[]{ idx };
                int newDim[] = { bsz, 1, hiddenDim };
                hidden->Reshape(3, newDim);
                _CopyIndexed(hidden, hiddens, 1, srcIdx, 1, tgtIdx);
                hidden->Reshape(bsz, hiddenDim);
            }
        }
    }

    for (auto t : timeSteps)
        DelTensor(t);
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
    Register(ConcatString("Weight_IH_", index), { inputDim, hiddenDim * 4 }, X_FLOAT);
    Register(ConcatString("Weight_HH_", index), { hiddenDim, hiddenDim * 4 }, X_FLOAT);
    Register(ConcatString("Bias_IH_", index), { hiddenDim * 4 }, X_FLOAT);
    Register(ConcatString("Bias_HH_", index), { hiddenDim * 4 }, X_FLOAT);
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
    auto weightIH = Get(ConcatString("Weight_IH_", index));
    auto weightHH = Get(ConcatString("Weight_HH_", index));
    auto biasIH = Get(ConcatString("Bias_IH_", index));
    auto biasHH = Get(ConcatString("Bias_HH_", index));

    /* transformations */
    XTensor noGated = MatrixMul(x, *weightIH) + MatrixMul(h, *weightHH) + *biasIH + *biasHH;

    /* split the big tensor to 4 parts */
    TensorList splited;
    XTensor i, f, g, o;
    for (auto p : { &i, &f, &g, &o }) {
        InitTensorV2(p, &h);
        splited.Add(p);
    }
    Split(noGated, splited, noGated.order - 1, 4);

    /* apply gating to the transformed tensor */

    /* update memory state and hidden state */
    c = Sigmoid(f) * c + Sigmoid(i) * HardTanH(g);
    h = HardTanH(c) * Sigmoid(o);
}