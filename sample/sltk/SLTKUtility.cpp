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

#include "SLTKUtility.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"

namespace util
{

/*
return the indices of the maximum values along an axis
this function is only used for vectors
*/
int ArgMaxVec(const XTensor &s, const int axis)
{
    
    CheckNTErrors(s.order == 2, "this function is only used for vectors!");
    int argMax(0);
    DTYPE minVal(-1000.0F);
    for(int i = 0; i < s.GetDim(1); ++i){
        if(s.Get2D(axis, i) >= minVal){
            minVal = s.Get2D(axis, i);
            argMax = i;
        }
    }
    
    return argMax;
}

/*
return the indices of the maximum values along an axis
this function is only used for matrices
*/
XTensor ArgMaxMat(XTensor &s, const int axis)
{
    

    CheckNTErrors(s.order == 2, "this function is only used for matrices!");
    XTensor r;
    r.SetTMPFlag();
    InitTensor2D(&r, s.GetDim(0), s.GetDim(1), X_FLOAT, s.devID, s.mem);
    r.SetZeroAll();
    XTensor index;
    InitTensor2D(&index, s.GetDim(0), s.GetDim(1), X_INT, s.devID, s.mem);
    TopK(s, r, index, axis, s.GetDim(axis));

    
    return SelectRange(index, 0, 0, 1);
}

/*
do padding on scores, that is add a row and a column to the raw tensor
the first row and the last column are filled with MIN_VALUE to mask corresponding tags
>>> src - a [batchSize, maxStepNum, tagNum] tensor
<<< tgt - a [batchSize, maxStepNum+1, tagNum+1] tensor
*/
void _PaddingScores(const XTensor *src, XTensor *tgt)
{
    

    const DTYPE m(-1000.0F);

    /* padding scores for time step: shape [batchSize, 1, tagNum+1] */
    XTensor t1, t2;
    InitTensor3D(&t1, src->GetDim(0), 1, src->GetDim(2), X_FLOAT, src->devID, src->mem);
    InitTensor3D(&t2, src->GetDim(0), 1, 1, X_FLOAT, src->devID, src->mem);
    SetDataFixed(t1, m);
    t2.SetZeroAll();
    XList t(2);
    t.Add(&t1);
    t.Add(&t2);
    XTensor startScores;
    startScores = Concatenate(t, 2);

    /* padding scores for tags: shape [batchSize, maxStepNum, 1] */
    XTensor padScores;
    InitTensor3D(&padScores, src->GetDim(0), src->GetDim(1), 1, X_FLOAT, src->devID, src->mem);
    SetDataFixed(padScores, m);
    t.Clear();
    t.Add(src);
    t.Add(&padScores);

    /* the target shape is [batchSize, maxStepNum+1, tagNum+1] */
    XTensor scores1;
    scores1 = Concatenate(t, 2);
    t.Clear();
    t.Add(&startScores);

    t.Add(&scores1);
    InitTensor3D(tgt, src->GetDim(0), src->GetDim(1) + 1, src->GetDim(2) + 1, X_FLOAT, src->devID, src->mem);
    _Concatenate(&t, tgt, 1);

    
}

/*
shrink scores, that is remove the first row and the last column of the src
>>> src - a [batchSize, maxStepNum+1, tagNum+1] tensor
<<< tgt - a [batchSize, maxStepNum, tagNum] tensor
*/
void _ShrinkScores(const XTensor *src, XTensor *tgt)
{
    


    XTensor tmp;
    InitTensor3D(&tmp, src->GetDim(0), src->GetDim(1) - 1, src->GetDim(2), X_FLOAT, src->devID, src->mem);
    tmp.SetTMPFlag();
    _SelectRange(src, &tmp, 1, 1, src->GetDim(1));
    InitTensor3D(tgt, tmp.GetDim(0), tmp.GetDim(1), tmp.GetDim(2) - 1, X_FLOAT, src->devID, src->mem);
    _SelectRange(&tmp, tgt, 2, 0, tmp.GetDim(2) - 1);


    
}

/* return log(reducesum(exp(x), dim)) */
XTensor LogReduceSumExp(XTensor &t, const int dim)
{
    


    CheckNTErrors(dim < t.order, "reduced tensor must be smaller!");
    CheckNTErrors(t.order < 3, "TODO!");
    XTensor res;

    const auto Protect = [](DTYPE &value)
    {
        if(IsNAN(value) || IsINF(value)){
            value = 0;
            return false;
        }
        return true;
    };

    InitTensor2D(&res, t.GetDim(0), t.GetDim(1), X_FLOAT, t.devID, t.mem);
    /* return a scalar */
    if(dim == -1){
        const DTYPE max = ReduceMaxScalar(t);
        SetDataFixed(res, max);
        _NegateMe(&res);
        _SumMe(&res, &t);
        _ExpMe(&res);

        XTensor r1, r2;
        r1 = ReduceSum(res, 0);
        r2 = ReduceSum(r1, 0);
        int shape[] = {1};
        r2.Reshape(1, shape);
        _LogMe(&r2);
        DTYPE data[]{max + r2.Get1D(0)};
        Protect(data[0]);
        r2.SetData(data, 1);


        
        return r2;
    }
    /* return a vector */
    else{
        XTensor max;
        InitTensor1D(&max, t.GetDim(dim), X_FLOAT, t.devID, t.mem);
        _ReduceMax(&t, &max, dim);
        _SubDim(&t, &max, &res, !dim);
        _ExpMe(&res);
        XTensor r;
        InitTensor1D(&r, t.GetDim(dim), X_FLOAT, t.devID, t.mem);
        _ReduceSum(&res, &r, dim);
        _LogMe(&r);
        _SumMe(&r, &max);

        /*for(int p = 0; p < r.GetSize(); ++p){
            Protect(static_cast<DTYPE*>(r.data)[p]);
        }*/
        
        return r;
    }


    
}

/* return a scalar max value of a tensor */
DTYPE ReduceMaxScalar(const XTensor &t)
{
    


    vector<XTensor> r;
    r.push_back(t);
    while(r.back().order){
        XTensor tmp;
        tmp = ReduceMax(r.back(), 0);
        r.push_back(tmp);
    }

    int shape[] = {1};
    r.back().Reshape(1, shape);
    
    return r.back().Get1D(0);
}

DTYPE GetMin(XTensor &t)
{
    DTYPE minVal = 0.0F;
    DTYPE *p = static_cast<DTYPE*>(t.data);
    for(int i = 0; i < t.GetSize(); ++i){
        const DTYPE d = p[i];
        minVal = minVal < d ? minVal : d;
    }
    return minVal;
}

DTYPE GetMax(XTensor &t)
{
    DTYPE maxVal = 0.0F;
    DTYPE *p = static_cast<DTYPE*>(t.data);
    for(int i = 0; i < t.GetSize(); ++i){
        const DTYPE d = p[i];
        maxVal = maxVal > d ? maxVal : d;
    }
    return maxVal;
}

/* check value in a tensor */
bool CheckTensor(DTYPE minVal, DTYPE maxVal, XTensor & t, int index)
{
    DTYPE *p = static_cast<DTYPE*>(t.data);
    for(int i = 0; i < t.GetSize(); ++i){
        const DTYPE d = p[i];
        if(IsNAN(d) || IsINF(d) || d < minVal || d > maxVal){
            if(t.order == 2){
                ERR("shape: %d, %d", t.GetDim(0), t.GetDim(1));
            }
            if(t.order == 3)
                ERR("shape: %d, %d, %d", t.GetDim(0), t.GetDim(1), t.GetDim(2));
            ERR("index: %d\tmax: %f\tmin: %f", index, GetMax(t), GetMin(t));
            LOG("-------------------------");
            return false;
        }
    }
    return true;
}


/* compute the derivative of hard tanh */
void _HardTanHBackward(XTensor *y, XTensor *dydx)
{
    

    DTYPE *r = static_cast<DTYPE *>(y->data);
    DTYPE *p = static_cast<DTYPE *>(dydx->data);

    for(int i = 0; i < y->GetSize(); ++i){
        if(r[i] < -1.0F || r[i] > 1.0F){
            p[i] = 0.0F;
        }
        else{
            p[i] = 1.0F;
        }
    }


    
}

/* compute the derivative of sigmoid */
void _SigmoidBackward(XTensor *y, XTensor *dydx)
{
    


    DTYPE *pin = static_cast<DTYPE *>(y->data);
    DTYPE *pout = static_cast<DTYPE *>(dydx->data);
    for(int i = 0; i < dydx->GetSize(); ++i){
        pout[i] = pin[i] * (1 - pin[i]);
    }


    
}

/*
t[index] := s
>>> s - the source tensor
>>> t - the target tensor
>>> index - the index to place s
*/
void _Copy(const XTensor *s, XTensor *t, int index)
{
    


    vector<int> srcIndex;
    vector<int> tgtIndex;
    for(int n = 0; n < s->GetDim(0); ++n){
        srcIndex.push_back(n);
        tgtIndex.push_back(n + s->GetDim(0) * index);
    }
    _CopyIndexed(s, t, 0, &srcIndex[0], s->GetDim(0), &tgtIndex[0], 1);


    
}

} // namespace util