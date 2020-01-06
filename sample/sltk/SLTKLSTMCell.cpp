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

#include "SLTKUtility.h"
#include "SLTKLSTMCell.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/function/FHeader.h"
#include <string>
#include "StringUtil.h"

using namespace ner;
using namespace util;
namespace ner {

/*
constructor
>>> inputDim - the input size of lstm
>>> hiddenDim - the hidden size of lstm
>>> layerNum - the number of layers
>>> bidirectional - use bi-lstm or not
*/
LSTM::LSTM(int inputDim, int hiddenDim, int layerNum, bool bidirectional)
{
    /* register parameters */
    int numPerLayer = bidirectional ? 2 : 1;
    for (int i = 0; i < layerNum * numPerLayer; i++) {
        LSTMCell* cell = new LSTMCell(inputDim, hiddenDim, i);
        cells.push_back(cell);
    }

}


LSTM::~LSTM()
{
}


void LSTM::Forward(XTensor& input, XTensor& lastHidden, XTensor& hiddens)
{
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
    Register(concat("Weight_H_", index), { inputDim + hiddenDim, hiddenDim * 4 }, X_FLOAT);
    Register(concat("Bias_H_", index), { hiddenDim }, X_FLOAT);
}

void LSTMCell::Forward(XTensor& x, XTensor& hPrev, XTensor& cPrev, XTensor& h, XTensor& c, int index)
{
    auto weight = Get(concat("Weight_H_", index));
    auto biasH = Get(concat("Bias_H_", index));
    auto biasF = Get(concat("Bias_F_", index));

    // combine x and h before the transformation
    XTensor xh = Concatenate(x, hPrev, x.order - 1);
    XTensor noGated = MatrixMul(xh, *weight);
    noGated = noGated + *biasH;

    // split the big tensor to 4 parts
    TensorList splited(4);
    Split(noGated, splited, noGated.order - 1, 4);

    XTensor* i = splited[0];
    XTensor* j = splited[0];
    XTensor* f = splited[0];
    XTensor* o = splited[0];

    /* apply gating to the transformed tensor */
    XTensor g = HardTanH(*j);
    XTensor gatedin = Sigmoid(*i) * g;
    XTensor memory = cPrev * Sigmoid(*f + *biasF);

    /* update memory state and hidden state */
    c = memory + gatedin;
    h = HardTanH(c) * Sigmoid(*o);
}


} // namespace ner