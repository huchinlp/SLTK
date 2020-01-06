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

#ifndef UTILITY_H
#define UTILITY_H

#define _CRTDBG_MAP_ALLOC

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new (_NORMAL_BLOCK, __FILE__, __LINE__)
#endif
#endif  // _DEBUG  

#include <vector>
#include "../../tensor/XTensor.h"

using namespace nts;
using namespace std;

namespace util
{

#define __FN__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define OK                                               \
    do {                                                 \
        fprintf(stderr, "[EXIT] ");                      \
        fprintf(stderr, "%-20sline %d", __FN__, __LINE__);\
        fprintf(stderr, "\t%-30s\n", __FUNCTION__);      \
    } while (0)

#define CALL                                             \
    do {                                                 \
        fprintf(stderr, "[CALL] ");                      \
        fprintf(stderr, "%-20sline %d", __FN__, __LINE__);\
        fprintf(stderr, "\t%-30s\n", __FUNCTION__);      \
    } while (0)

//#define OK 
//#define CALL

#define LOG(...)                                             \
    do {                                                     \
        fprintf(stderr, "[INFO] ");                          \
        fprintf(stderr, __VA_ARGS__);                        \
        fprintf(stderr, "\n");                               \
        fflush(stdout);                                      \
    } while (0)

#define ERR(...)                                             \
    do {                                                     \
        fprintf(stderr, "[ERROR] ");                          \
        fprintf(stderr, __VA_ARGS__);                        \
        fprintf(stderr, "\n");                               \
        fflush(stdout);                                      \
    } while (0)

XTensor ArgMaxMat(XTensor &s, int axis);
DTYPE ReduceMaxScalar(const XTensor &t);
int ArgMaxVec(const XTensor &s, int axis);
XTensor LogReduceSumExp(XTensor &t, int dim);
void _Copy(const XTensor *s, XTensor *t, int index);
void _ShrinkScores(const XTensor *src, XTensor *tgt);
void _PaddingScores(const XTensor *src, XTensor *tgt);
void _SigmoidBackward(XTensor *y, XTensor *dydx);
void _HardTanHBackward(XTensor *y, XTensor *dydx);
bool CheckTensor(DTYPE minVal, DTYPE maxVal, XTensor &t, int index);
DTYPE GetMin(XTensor &t);
DTYPE GetMax(XTensor &t);
} // namespace util

#endif // UTILITY_H_
