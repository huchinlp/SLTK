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
#include "SLTKUtility.h"
#include "../../tensor/core/CHeader.h"

using namespace util;

namespace ner {


XTensor SequenceTagger::Forward(XTensor& input)
{
    return XTensor();
}


SequenceTagger::SequenceTagger(Dict* tagDict, bool useCRF, bool useRNN, 
                               int rnnLayer, int hiddenSize, Embedding* embedding, 
                               const string& tagType, const string& rnnType, 
                               float dropout, float wordropout, float lockedropout)
{
    Register("transition", { tagDict->tagNum, tagDict->tagNum }, X_FLOAT);
    Register("embedding2NN", { embedding->outDim, embedding->outDim }, X_FLOAT);
    Register("linear", { hiddenSize*2, tagDict->tagNum }, X_FLOAT);
    
}

} // namespace ner