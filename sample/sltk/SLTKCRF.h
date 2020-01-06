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

#ifndef CRF_H_
#define CRF_H_

#include <vector>
#include "../../tensor/XTensor.h"
#include "SLTKDataUtility.h"

using namespace nts;
using namespace std;
using namespace util;
namespace ner
{

/* a CRF structure
 * probability computation:  forward-backward algorithm
 * decode:  Viterbi algorithm
 * see "An Introduction to Conditional Random Fields" for more details */
class CRF
{
public:

    /* normalization factors */
    vector<DTYPE> logNorm;

    /* transition matrix */
    XTensor* transition;

    /* constructor */
    CRF(int tagNum, int startTag, int stopTag);

    /* constructor */
    ~CRF();

    /* reset cache */
    void ResetCache();

    /* get alpha_i or beta_i */
    void GetAlphaBeta(XTensor &res, bool isAlpha, int sent, int pos);

    /* set alpha_i or beta_i */
    void SetAlphaBeta(XTensor & src, bool isAlpha, int sent, int pos);

    /* compute the loss in a CRF */
    DTYPE LogLikelihood(XTensor *inputs,
                        const vector<vector<int>> &tagIndices,
                        const vector<int> &realLengths);

    /* compute the unary scores of sequences */
    void GetUnaryScore(const XTensor *inputs,
                    vector<DTYPE> &unaryScores,
                    const vector<vector<int>> &tagIndices,
                    const vector<int> &realLengths);

    /* compute the binary scores of sequences */
    void GetBinaryScore(const XTensor *inputs,
                     vector<DTYPE> &binaryScores,
                     const vector<vector<int>> &tagIndices,
                     const vector<int> &realLengths) const;

    /* compute normalizations in a CRF */
    void CRFLogNorm(XTensor *inputs,
                    vector<DTYPE> &normalizations,
                    const vector<int> &realLengths);

    /* compute forward vector in a CRF */
    void CalAlpha(XTensor &res, int sent, int pos, bool isCached);

    /* compute backward vector in a CRF */
    void CalBeta(XTensor &res, int sent, int pos, bool isCached);

    /* decode the highest scoring sequence of tags for sentences in a mini-batch */
    void Decode(vector<vector<int>> &paths,
                XTensor *extendedScores,
                const vector<int> &realLengths);

    /* decode the highest scoring sequence of tags for a sentence */
    void Viterbi(vector<int> &result, XTensor * score) const;

    /* backward propagation in a CRF */
    void Backward(XTensor * inputs, const vector<int> &realLengths, const vector<vector<int>> &tagIndices);
};

} // namespace ner

#endif // !CRF_H_
