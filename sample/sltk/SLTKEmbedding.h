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

#include "SLTKDataSet.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/XGlobal.h"

using namespace nts;

struct Embedding
{
    /* device id */
    int devID;

    /* the vocab size */
    int vocabSize;

    /* the embedding dimension */
    int embSize;

    /* dictionary */
    shared_ptr<Vocab> vocab;

    /* the pre-trained word embeddings */
    XTensor vecs;

    /* load embeddings from a file */
    void LoadWordEmbeddings(const char* fn, int myDevID);

    /* gather embeddings for the input */
    XTensor Embed(const XTensor& input);
};