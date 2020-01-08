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
#include "..//..//model/Model.h"
#include "../../tensor/XTensor.h"

using namespace nts;
using namespace std;

/*  This module implements a conditional random field.
 *  The forward computation computes the log likelihood
 *  of the given sequence of tags and emission score tensor.
 *  It also has decode method which finds the best tag sequence
 *  given an emission score tensor using `Viterbi algorithm`.
 *  Ref: "Conditional random fields: Probabilistic models for segmenting and labeling sequence data".
 *  Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
 */

struct CRF : public Model
{
public:

    /* number of tags */
    int tagNum;

    /* if the first dim of input is batchSize */
    bool batchFirst;

    /* constructor */
    CRF(int tagNum, bool batchFirst = true);

    /* initializer */
    void ResetParams();

    /* decoder */
    vector<vector<int>> Decode(XTensor& emissions, XTensor& mask);

    /* viterbi decoder */
    vector<vector<int>> ViterbiDecode(XTensor& emissions, XTensor& mask);
};

/* Return a tensor of elements selected from either x or y, depending on condition. */
XTensor Where(XTensor& condition, XTensor& x, XTensor& y);
