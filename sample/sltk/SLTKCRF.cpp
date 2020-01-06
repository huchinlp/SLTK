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

#include <memory>

#include "SLTKCRF.h"
#include "SLTKUtility.h"
#include "SLTKDataUtility.h"
#include "../../tensor/core/CHeader.h"

using namespace std;
using namespace util;

namespace ner
{

/* 
constructor 
>>> tagNum - the number of different tags
>>> startTag - the id of the start tag
>>> stopTag - the id of the stop tag
*/
CRF::CRF(int tagNum, int startTag, int stopTag)
{
    transition = NewTensor2DV2(tagNum, tagNum);
    transition->SetDataRand();
    _SetDataDim(transition, startTag, 1, 0, -10000.);
    _SetDataDim(transition, stopTag, 1, 1, -10000.);
}

/* de-constructor */
CRF::~CRF()
{
    if (transition != nullptr) {
        DelTensor(transition);
    }
}

/*
compute the log likelihood of tag sequences in a CRF
>>> inputs - a [config->batchSize, maxStepNum, tagNum+1] tensor, score of each tag
>>> tagIndices - a [config->batchSize, maxStepNum] list, record of correct tags
>>> realLengths - a [config->batchSize] list storing real lengths of sequences
<<< scores - the mean loss of a mini-batch
*/
DTYPE CRF::LogLikelihood(XTensor * inputs, const vector<vector<int>> &tagIndices, const vector<int> &realLengths)
{
    

    /* clear cache */
    ResetCache();

    vector<DTYPE> unNormalizedScores(inputs->GetDim(0));

    /* calculate unary scores of a mini-batch */
    GetUnaryScore(inputs, unNormalizedScores, tagIndices, realLengths);

    /* add binary scores to above vector */
    GetBinaryScore(inputs, unNormalizedScores, tagIndices, realLengths);

    vector<DTYPE> normalizations(inputs->GetDim(0));

    /* compute the normalization for a CRF */
    CRFLogNorm(inputs, normalizations, realLengths);

    DTYPE score(0.0F);
    for(int i = 0; i < inputs->GetDim(0); ++i){
        score += (unNormalizedScores.at(i) - normalizations.at(i));
    }

    /* return the mean loss in a mini-batch */
    
    return (score / DTYPE(inputs->GetDim(0)));
}

/*
calculate the unary score of a tag sequence
>>> inputs - a [config->batchSize, maxStepNum+1, tagNum+1] tensor, score of each tag
>>> tagIndices - a [config->batchSize, maxStepNum+1] list, record of correct tags
>>> realLengths - a [config->batchSize] list storing real lengths of sequences
<<< unaryScores - a list to store the unary scores of sequences
we gather scores of each correct tag
here is an example (suppose batch size is 1 for convenience):
tagId = {'B-Loc':1, 'I-Loc':2, 'Start':0} ('Start' is a pre-defined tag for padding)
inputData = ["a", "b", "c", "PAD"]
tagIndices= [1,2,2,0]
inputs    = [[[-3,5,-1], [1,1,3], [-1,2,4], [0,0,0]]]
the unary score of a sentence is the sum of 'score of the correct tag'
so the result is: 5 + 3 + 4 + 0 = 12
*/
void CRF::GetUnaryScore(const XTensor * inputs,
                     vector<DTYPE> &unaryScores,
                     const vector<vector<int>> &tagIndices,
                     const vector<int> &realLengths)
{
    
    for(int i = 0; i < inputs->GetDim(0); ++i){

        /* mask the first score and padding scores */
        for(int j = 0; j < realLengths.at(i); ++j){
            unaryScores.at(i) += inputs->Get3D(i, j, tagIndices.at(i).at(j));
        }
    }
    
}

/*
calculate the transition(binary) scores for tags paths
>>> inputs - a [config->batchSize, maxStepNum+1, tagNum+1] tensor, score of each tag
>>> tagIndices - a [config->batchSize, maxStepNum+1] list, record of correct tags
>>> realLengths - a [config->batchSize] list storing real lengths of sequences
<<< unaryScores - a list to store the unary scores of sequences
suppose the transition matrix is:
------------------------------
        Start   B-Loc   I-Loc
Start   0.1,    0.1,    0.1
B-Loc   0.1,    0,      100
I-Loc   0.1,    100,    1
------------------------------
in our example, there are 2 real tags and another fake tag 'Start'
we further suppose the correct tag path is ['B-Loc','I-Loc','I-Loc']
first, extend the path to ['Start','B-Loc','I-Loc','I-Loc']
then accumulate ['Start'->'B-Loc', 'B-Loc'->'I-Loc', 'I-Loc'->'I-Loc']
so the result is: 0.1 + 100 + 1 = 101.1
*/
void CRF::GetBinaryScore(const XTensor * inputs,
                      vector<DTYPE> &binaryScores,
                      const vector<vector<int>> &tagIndices,
                      const vector<int> &realLengths) const
{
    
    for(int i = 0; i < inputs->GetDim(0); ++i){

        DTYPE binaryScore(0.0F);
        for(int j = 0; j < realLengths.at(i) - 1; ++j){
            binaryScore += transition.Get2D(tagIndices.at(i).at(j), tagIndices.at(i).at(j + 1));
        }
        binaryScores.at(i) += binaryScore;
    }
    
}

/*
calculate the non-normalized probability(normalization factor) in a crf
>>> inputs - a [config->batchSize, maxStepNum+1, tagNum+1] tensor, score of each tag
>>> realLengths - a [config->batchSize] list storing real lengths of sequences
<<< normalizations - a list to store the normalizations in a crf
here is an implementation of the forward algorithm which time complexity is O(n^2)
let n be the real length of a input sequence, Z is our objective, here we have:
----------------------------------------------------------------------
Z(x) = a_n(y_n|x)*cache.beta0, where cache.beta0 is a column vector filled by 1
a_n(y_n|x) = a_0(y_0|x)*M_1(x)*M_2(x)*...*M_n(x)
a_0(y|x) = 1 if y = start, 0 else
M_n(x) records all 'tag-tag'(transition) scores from word n-1 to word n
----------------------------------------------------------------------
*/
void CRF::CRFLogNorm(XTensor * inputs, vector<DTYPE> &normalizations, const vector<int> &realLengths)
{
    

    /* alpha */
    XList *listA = static_cast<XList*>(cache.base->Get(0));
    XTensor *alpha0 = static_cast<XTensor*>(listA->Get(0));
    XTensor *wordScores = NewTensorBuf(alpha0, config->devID, mem);
    XTensor *tmpSumA = NewTensorBuf(&transition, config->devID, mem);
    XTensor *padSumA = NewTensorBuf(alpha0, config->devID, mem);

    /* beta */
    XTensor *beta0 = static_cast<XTensor*>(listA->Get(listA->count - 1));
    XTensor *tmpSumB = NewTensorBuf(&transition, config->devID, mem);
    XTensor *padSumB = NewTensorBuf(beta0, config->devID, mem);

    for(int i = 0; i < inputs->GetDim(0); ++i){

        XList *sentence = static_cast<XList*>(cache.base->Get(i));

        /* m1 to mn, shape (tagNum+1, tagNum+1)
           value: transition + wordScores       */
        for(int j = 1; j < realLengths.at(i); ++j){

            wordScores->SetZeroAll();
            for(int k = 0; k < inputs->GetDim(2); ++k){
                wordScores->Set2D(inputs->Get3D(i, j, k), 0, k);
            }

            XTensor *mj = static_cast<XTensor*>(sentence->Get(j));
            _CopyValues(&transition, mj);
            _SumDim(mj, wordScores, 1);
        }
        _CopyValues(alpha0, padSumA);
        _CopyValues(beta0, padSumB);

        for(int j = 1; j < realLengths.at(i); ++j){

            /* only compute forward-vector when training the model */
            if(config->isTrain){
                XTensor *a = static_cast<XTensor*>(sentence->Get(j));
                _CopyValues(a, tmpSumA);
                /* use addition to replace exponent multiplication */
                _SumDim(tmpSumA, padSumA, 0);
                int rawShapeA[]{padSumA->GetDim(0),padSumA->GetDim(1)};
                int newShapeA[]{padSumA->GetSize()};
                padSumA->Reshape(1, newShapeA);
                *padSumA = LogReduceSumExp(*tmpSumA, 0);
                padSumA->Reshape(2, rawShapeA);
            }

            XTensor *b = static_cast<XTensor*>(sentence->Get(realLengths.at(i) - j));
            _CopyValues(b, tmpSumB);

            /* use addition to replace exponent multiplication */
            _SumDim(tmpSumB, padSumB, 1);
            int rawShapeB[]{padSumB->GetDim(0),padSumB->GetDim(1)};
            int newShapeB[]{padSumB->GetSize()};
            padSumB->Reshape(1, newShapeB);
            *padSumB = LogReduceSumExp(*tmpSumB, 1);
            padSumB->Reshape(2, rawShapeB);

            if(config->isTrain){
                SetAlphaBeta(*padSumA, true, i, j);
                SetAlphaBeta(*padSumB, false, i, j);
            }
        }

        logNorm.push_back(padSumB->Get2D(0, 0));
        normalizations.at(i) += logNorm.back();
    }
    DelTensorBuf(wordScores);
    DelTensorBuf(padSumA);
    DelTensorBuf(tmpSumA);
    DelTensorBuf(padSumB);
    DelTensorBuf(tmpSumB);
    
}

/*
infer the final tags for sequences in a mini-batch
>>> extendedScores - a [config->batchSize, maxStepNum+1, tagNum+1] tensor records the unary scores
>>> realLengths - a [config->batchSize] list for real lengths of sequences
<<< paths - a [config->batchSize, realLength] matrix records the final tags
*/
void CRF::Decode(vector<vector<int>> &paths, XTensor * extendedScores, const vector<int> &realLengths)
{
    XTensor score;
    InitTensor2D(&score, extendedScores->GetDim(1), extendedScores->GetDim(2), X_FLOAT, config->devID, mem);

    for(int i = 0; i < int(realLengths.size()); ++i){

        score.SetZeroAll();
        const int realLength = realLengths.at(i) - 1;
        int srcIndex[]{i};
        int tgtIndex[]{0};
        _CopyIndexed(extendedScores, &score, 0, srcIndex, 1, tgtIndex, 1);

        XTensor realScore;
        realScore = SelectRange(score, 0, 0, realLength + 1);

        Viterbi(paths.at(i), &realScore);
    }
    
}

/*
an implementation of Viterbi algorithm
>>> score - a [maxStepNum, tagNum] tensor records the unary scores
>>> paths - a [realLength] list to record the predicted tags
infer the highest scoring sequence of tags
*/
void CRF::Viterbi(vector<int> &result, XTensor * score) const
{
    XTensor trellis;
    InitTensor2D(&trellis, score->GetDim(0), score->GetDim(1), X_FLOAT, config->devID, mem);
    trellis.SetZeroAll();

    /* trellis[0] := score[0] */
    int srcIndex[1] = {0};
    int tgtIndex[1] = {0};
    _CopyIndexed(score, &trellis, 0, srcIndex, 1, tgtIndex, 1);

    /* use int for indices */
    XTensor backPointers;
    InitTensor2D(&backPointers, score->GetDim(0), score->GetDim(1), X_INT, config->devID, mem);
    backPointers.SetZeroAll();

    for(int i = 1; i < score->GetDim(0); ++i){

        /* v := trellis[t-1] + transition */
        // todo: refactor all XTensor type
        XTensor v, part1;
        part1 = SelectRange(trellis, 0, i - 1, i);
        v = SumDim(transition, part1, 0);

        /* trellis.at(i) := score.at(i) + ReduceMax(v, 0) */
        XTensor tmp;
        XTensor max;
        XTensor part2;

        InitTensor2D(&part2, 1, score->GetDim(1), X_FLOAT, config->devID, mem);
        srcIndex[0] = i;
        _CopyIndexed(score, &part2, 0, srcIndex, 1, tgtIndex, 1);

        max = ReduceMax(v, 0);
        tmp = Sum(part2, max);

        int src[] = {0};
        int tgt[] = {i};
        _CopyIndexed(&tmp, &trellis, 0, src, 1, tgt, 1);

        /* backPointers[t] := argmax(v,0) */
        XTensor argmax;
        argmax = ArgMaxMat(v, 0);
        _CopyIndexed(&argmax, &backPointers, 0, src, 1, tgt, 1);

    }

    /* path := [argmax(trellis[-1])] */
    result.push_back(ArgMaxVec(trellis, trellis.GetDim(0) - 1));
    for(int i = backPointers.GetDim(0) - 1; i > 0; --i){
        result.push_back(backPointers.Get2DInt(i, result.back()));
    }
    std::reverse(result.begin(), result.end());


    
}

/*
back-propagation within a mini-batch in a crf
calculate the derivative dL/di, where L is the CRF forward result and i is the input
>>> tagIndices  - a [config->batchSize, config->maxTime+1] table indicates correct tag
>>> realLengths - a [config->batchSize] list indicates the real length of each sentence
>>> inputs      - a [config->batchSize, config->maxTime+1, tagNum+1] tensor indicates input of a crf
*/
void CRF::Backward(XTensor * inputs, const vector<int> &realLengths, const vector<vector<int>> &tagIndices)
{
    

    inputs->grad->SetZeroAll();
    transition.grad->SetZeroAll();

    int sentDimSize[] = {inputs->GetDim(1),inputs->GetDim(2)};

    /* the correct tag transition paths */
    XTensor *goldTrans = NewTensorBuf(&transition, config->devID, mem);

    /* the correct tag paths */
    XTensor *gold = NewTensorBuf(2, sentDimSize, X_FLOAT, 1, config->devID, mem);

    /* gradient of a sentence */
    XTensor *sentGrads = NewTensorBuf(2, sentDimSize, X_FLOAT, 1.0F, config->devID, mem);

    int wordDimSize[] = {inputs->GetDim(2)};
    XTensor *tokenGrads = NewTensorBuf(1, wordDimSize, X_FLOAT, 1.0F, config->devID, mem);

    /* accumulated m1 for post processing */
    XTensor *accumulateM = NewTensorBuf(&transition, config->devID, mem);

    for(int i = 0; i < inputs->GetDim(0); ++i){

        XList *sentence = static_cast<XList*>(cache.base->Get(i));

        /* clear states */
        gold->SetZeroAll();
        goldTrans->SetZeroAll();
        sentGrads->SetZeroAll();
        tokenGrads->SetZeroAll();
        accumulateM->SetZeroAll();

        for(int j = 1; j < realLengths.at(i); ++j){

            cache.leftTmp->SetZeroAll();
            cache.rightTmp->SetZeroAll();

            gold->Set2D(1.0F, j, tagIndices.at(i).at(j));
            const int lastGold = tagIndices.at(i)[j - 1];
            const int currentGold = tagIndices.at(i).at(j);
            goldTrans->Set2D(1.0F + goldTrans->Get2D(lastGold, currentGold), lastGold, currentGold);

            /* dZ/dM_j, where j ranged from 1 to realLength_i */
            GetAlphaBeta(*cache.leftTmp, true, i, j - 1);
            GetAlphaBeta(*cache.rightTmp, false, i, realLengths.at(i) - j - 1);

            XTensor *mj = static_cast<XTensor*>(sentence->Get(j));
            mj->grad->SetZeroAll();

            _SumDim(mj->grad, cache.leftTmp, 0);
            _SumDim(mj->grad, cache.rightTmp, 1);
            _SumMe(mj->grad, mj);
            _ScaleAndShiftMe(mj->grad, 1.0F, -logNorm.at(i));
            _ExpMe(mj->grad);

            /* dL/dT = sum(dZ/dM_j)/Z - goldTrans */
            if(j != 1)
                _SumMe(transition.grad, mj->grad);
            else{
                int srcIndex[]{0};
                int tgtIndex[]{transition.grad->GetDim(0) - 1};
                _CopyIndexed(mj->grad, accumulateM, 0, srcIndex, 1, tgtIndex, 1);
            }

            /* collect gradient of each token */
            _ReduceSum(mj->grad, tokenGrads, 0);
            _Copy(tokenGrads, sentGrads, j);
        }

        _SumMe(transition.grad, accumulateM);
        _SubMe(transition.grad, goldTrans);

        /* ************************************************************ */
        /* dL/dScores = Merge(sum(dZ/dM_j)) - gold                      */
        /* the scores shape [batchSize, maxTime+1, tagNum+1]            */
        /* each sentence has maxTime+1-realLength positions for padding */
        /* ************************************************************ */
        _SubMe(sentGrads, gold);

        /* collect gradient of each sentence */
        _Copy(sentGrads, inputs->grad, i);
    }

    _ScaleAndShiftMe(transition.grad, 1.0F / DTYPE(inputs->GetDim(0)));
    _ScaleAndShiftMe(inputs->grad, 1.0F / DTYPE(inputs->GetDim(0)));

    DelTensorBuf(goldTrans);
    DelTensorBuf(gold);
    DelTensorBuf(sentGrads);
    DelTensorBuf(tokenGrads);
    DelTensorBuf(accumulateM);

    
}

/* calculate alpha */
void CRF::CalAlpha(XTensor &res, int sent, int pos, bool isCached)
{
    
    if(isCached)
        GetAlphaBeta(res, true, sent, pos);
    else{

    }
    
}

/* calculate beta */
void CRF::CalBeta(XTensor &res, int sent, int pos, bool isCached)
{
    


    if(isCached){
        /* hit beta0 directly */
        GetAlphaBeta(res, false, sent, 0);
    }
    else{
        /* hit beta_pos+1 */
        XList *sentence = static_cast<XList*>(cache.beta->Get(sent));
        const int cachePos = sentence->count - pos;
        GetAlphaBeta(res, false, sent, cachePos);

    }
    //XTensor *ones = static_cast<XTensor*>(list.Get(list.count - 1));
    //InitTensor(res, ones);
    //_CopyValues(ones, res);
    //while(begin < end){
    //    end--;
    //    XTensor tmpSum(*static_cast<XTensor*>(list.Get(end)));
    //    _SumDim(&tmpSum, res, 1);
    //    XTensor logReduceSumExp;
    //    logReduceSumExp = LogReduceSumExp(tmpSum, 1);
    //    int shape[]{res->GetDim(0),res->GetDim(1)};
    //    logReduceSumExp.Reshape(2, shape);
    //    _CopyValues(&logReduceSumExp, res);
    //}
    
}
} // namespace ner