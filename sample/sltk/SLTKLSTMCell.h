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

#include <vector>
#include <memory>
#include "../../model/Model.h"

using namespace std;

/* lstm cell */
struct LSTMCell : public Model
{
    /* constructor */
    LSTMCell(int inputDim, int hiddenDim, int index);

    /* lstm forward function in a cell */
    void Forward(XTensor& x, XTensor& h, XTensor& c, int index);
};

/* lstm struct */
struct LSTM : public Model
{
    /* input dim */
    int inputDim;

    /* hidden dim */
    int hiddenDim;

    /* layer number */
    int layerNum;

    /* is bidirectional or not */
    bool bidirectional;

    /* lstm cells */
    vector<shared_ptr<LSTMCell>> cells;

    /* constructor */
    LSTM(int inputDim, int hiddenDim, int layerNum, bool bidirectional);

    /* de-constructor */
    ~LSTM();

    /* forward */
    XTensor Forward(XTensor& input, XTensor& hidden, XTensor& memory);
};