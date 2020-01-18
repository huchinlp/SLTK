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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-01-16
 */

#include "SLTKNNUtil.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/function/FHeader.h"

/* split a big tensor into small ones (equal size)  */
void Split(const XTensor& big, TensorList& list, int dim)
{
    int num = big.GetDim(dim);
    int* dimSize = new int[big.order - 1];
    for (int i = 0; i < dim; i++)
        dimSize[i] = big.dimSize[i];
    for (int i = 0; i < big.order - dim - 1; i++)
        dimSize[dim + i] = big.dimSize[dim + i + 1];

    for (int i = 0; i < num; i++)
        list.Add(NewTensorV2(big.order - 1, dimSize, big.dataType, big.denseRatio, big.devID));
    Split(big, list, dim, num);
}