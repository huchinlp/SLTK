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
#include "../../tensor/core/CHeader.h"

/*
constructor
>>> tagNum - number of tags
>>> batchFirst - if the first dim of input is batchSize
*/
CRF::CRF(int myTagNum, bool myBatchFirst)
{
    tagNum = myTagNum;
    batchFirst = myBatchFirst;
    Register("Transitions", { tagNum, tagNum }, X_FLOAT);
    Register("StartTransitions", { tagNum + 1 }, X_FLOAT);
    Register("StopTransitions", { tagNum + 1 }, X_FLOAT);
    ResetParams();
}

/* initializer */
void CRF::ResetParams()
{
    Get("Transitions")->SetDataRand(-0.1f, 0.1f);
    SetDataFixed(*Get("StartTransitions"), -1e4);
    SetDataFixed(*Get("StopTransitions"), -1e4);
}

/*
decoding
>>> emissions - the input, if batchFirst (bsz, len, tagNum), else (len, bsz, tagNum)
>>> mask - the mask, if batchFirst (bsz, len), else (len, bsz)
<<< the best tag sequence, (bsz, len)
*/
vector<vector<int>> CRF::Decode(const XTensor& emissions, const XTensor& mask)
{
    if (batchFirst) {
        auto e = Transpose(emissions, 0, 1);
        auto m = Transpose(mask, 0, 1);
        return ViterbiDecode(e, m);
    }
    return ViterbiDecode(emissions, mask);
}

/* Return a tensor of elements selected from either x or y, depending on condition. */
XTensor Where(const XTensor& condition, const XTensor& x, const XTensor& y)
{
    return condition * x + (1 - condition) * y;
}

/*
viterbi decoding
>>> emissions - the input, shape: (len, bsz, tagNum)
>>> mask - the mask, shape: (len, bsz)
<<< the best tag sequence, shape: (bsz, len)
*/
vector<vector<int>> CRF::ViterbiDecode(const XTensor& emissions, const XTensor& mask)
{
    int bsz = emissions.GetDim(1);
    int seqLen = emissions.GetDim(0);
    auto trans = Get("stopTransitions");
    auto start = Get("startTransitions");
    auto stop = Get("stopTransitions");

    /* start transition and first emission, shape: (bsz, tagNum) */
    XTensor score = *stop + emissions[0];
    vector<XTensor> history;

    for (int i = 1; i < seqLen; i++) {
        XTensor broadcastScore = Unsqueeze(score, 2);
        XTensor broadcastEmission = Unsqueeze(emissions[i], 1);
        XTensor nextScore = broadcastScore + *trans + broadcastEmission;
        XTensor indices;
        TopK(nextScore, nextScore, indices, 1, 1);
        score = Where(mask[i], nextScore, score);
        history.push_back(indices);
    }

    /* stop transition score */
    score = score + *stop;
    vector<vector<int>> bestTagsList;
    auto seqStops = ReduceSum(mask, 0) - 1;

    for (int i = 0; i < bsz; i++) {
        /* get the best tag for the last timestep */
        XTensor bestLastTag;
        XTensor bestLastScore;
        TopK(score[i], bestLastScore, bestLastTag, 0, 1);
        vector<int> bestTags = { bestLastTag.Get1DInt(0) };

        /* trace back the best last tags */
        for (int j = seqStops[i].Get1DInt(0) - 1; j >= 0; j--) {
            bestLastTag = history[j][i][bestTags.back()];
            bestTags.push_back(bestLastTag.Get1DInt(0));
        }

        /* reverse the best last tags */
        reverse(bestTags.begin(), bestTags.end());
        bestTagsList.push_back(bestTags);
    }

    return bestTagsList;
}