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

#include <algorithm>
#include "SLTKCRF.h"
#include "SLTKNNUtil.h"
#include "../../tensor/core/CHeader.h"

/*
constructor
>>> tagNum - number of tags
>>> batchFirst - if the first dim of input is batchSize
*/
CRF::CRF(int myTagNum)
{
    tagNum = myTagNum;
    startID = myTagNum - 2;
    stopID = myTagNum - 1;
    Register("Transitions", { tagNum, tagNum }, X_FLOAT);
    ResetParams();
}

/* initializer */
void CRF::ResetParams()
{
    Get("Transitions")->SetDataRand(-0.1f, 0.1f);
}

/*
decoding
>>> emissions - the input, (bsz, len, tagNum)
>>> mask - the mask, (bsz, len)
<<< the best tag sequence, (bsz, len)
*/
vector<vector<int>> CRF::Decode(const XTensor& emissions, const XTensor& mask)
{
    TensorList batch;
    Split(emissions, batch, 0);
    vector<vector<int>> bestPaths;

    for (int i = 0; i < batch.Size(); i++)
        bestPaths.emplace_back(ViterbiDecode(*batch[i], mask));
    return bestPaths;
}

/* Return a tensor of elements selected from either x or y, depending on condition. */
XTensor Where(const XTensor& condition, const XTensor& x, const XTensor& y)
{
    return condition * x + (1 - condition) * y;
}

/*
viterbi decoding on a sequence
>>> emissions - the input, shape: (len, tagNum)
>>> mask - the mask, shape: (len)
<<< the best tag sequence, shape: (len)
*/
vector<int> CRF::ViterbiDecode(const XTensor& emissions, const XTensor& mask)
{
    int seqLen = emissions.GetDim(0);
    auto trans = Get("Transitions");

    XTensor backpointers;
    InitTensor2DV2(&backpointers, seqLen, tagNum, X_INT, trans->devID);

    /* forwardVar shape: (1, tagNum) */
    XTensor forwardVar;
    InitTensor1DV2(&forwardVar, tagNum, X_FLOAT, trans->devID);
    SetDataFixed(forwardVar, -1e4);
    forwardVar.Set1D(0, startID);

    TensorList timeSteps;
    Split(emissions, timeSteps, 0);

    /* iterations on timesteps  */
    for (int t = 0; t < seqLen; t++) {
        /* feature shape: (tagNum) */
        XTensor& feature = *(timeSteps[t]);

        /* nextTagVar shape: (tagNum, tagNum) */
        XTensor nextTagVar = SumDim(*trans, forwardVar, 0);

        XTensor bestTag, bestScore;
        InitTensor2DV2(&bestTag, tagNum, 1, X_INT, trans->devID);
        InitTensor2DV2(&bestScore, tagNum, 1, X_FLOAT, trans->devID);
        TopK(nextTagVar, bestScore, bestTag, 1, 1);
        bestScore.Reshape(tagNum);
        forwardVar = bestScore + feature;

        /* save the best tag */
        int srcIdx[]{ 0 };
        int tgtIdx[]{ t };
        bestTag.Reshape(1, tagNum);
        _CopyIndexed(&bestTag, &backpointers, 0, srcIdx, 1, tgtIdx);
    }

    /* todo: optimize this section */
    XTensor stopIdx;
    InitTensor1DV2(&stopIdx, 1, X_INT, trans->devID);
    stopIdx.Set1DInt(stopID, 0);
    XTensor stopTransition = Select(*trans, stopIdx, 0);

    /* get the best tag for the last token */
    XTensor terminalVar = Squeeze(forwardVar) + stopTransition;
    terminalVar.Set1D(-1e4, startID);
    terminalVar.Set1D(-1e4, stopID);

    XTensor bestTag, bestScore;
    InitTensor1DV2(&bestTag, 1, X_INT, trans->devID);
    InitTensor1DV2(&bestScore, 1, X_FLOAT, trans->devID);
    TopK(terminalVar, bestScore, bestTag, 0, 1);

    int bestTagID = bestTag.Get1DInt(0);
    vector<int> bestPath{ bestTagID };

    for (int i = backpointers.GetDim(0) - 1; i > 0; i--)
        bestPath.push_back(backpointers.Get2DInt(i, bestTagID));

    reverse(bestPath.begin(), bestPath.end());
    return bestPath;
}