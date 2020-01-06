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

#ifndef BLSTM_CRF_MODEL_H_
#define BLSTM_CRF_MODEL_H_

#include <string>

#include "..//..//model/Model.h"
#include "SLTKCRF.h"
#include "SLTKLSTMCell.h"
#include "SLTKDataUtility.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"

using namespace nts;
using namespace std;
using namespace util;
namespace ner {

/* the sequence labeling model */
struct SequenceTagger: Model
{
    /* embedding */
    Embedding* embedding;

    /* RNNs(bi-directional) */
    Model* rnns;

    /* CRF layer */
    CRF* crf;

    XList GetParameters();

    XTensor Forward(XTensor& input);

    explicit SequenceTagger(Dict* tagDict, bool useCRF, bool useRNN,
                            int rnnLayer, int hiddenSize, Embedding* embedding,
                            const string& tagType, const string& rnnType,
                            float dropout, float wordropout, float lockedropout);

    ~SequenceTagger();
};

} // namespace ner

#endif // BLSTM_CRF_MODEL_H_