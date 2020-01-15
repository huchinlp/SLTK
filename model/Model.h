/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northestern University. 
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
* 
* the basic class for NN
*
* $Created by: HU Chi (huchinlp@foxmail.com) 2019-09-12
*
*/

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "../tensor/XGlobal.h"
#include "../tensor/XTensor.h"

using namespace std;
using namespace nts;

using Dim = std::initializer_list<int>;

/* Parameter is a base class for parameters */
struct Parameter {

public:
    /* the parameter list */
    vector<shared_ptr<XTensor>> paramList;

    /* the name list for parameters */
    vector<string> nameList;

public:
    /* add a parameter to the list */
    void AddParameter(const string& name, Dim dims, TENSOR_DATA_TYPE dataType);

    /* get a parameter by its name */
    shared_ptr<XTensor> GetParameter(const string& name);
};

/* Model is a base class for neural networks */
struct Model {

public:
    /* device id */
    int devID;

    /* model parameters */
    Parameter parameters;

public:
    /* load a model from a binary file */
    void Load(const char* fn);

    /* save the model to a binary file */
    void Save(const char* fn);

    /* get a parameter by its name */
    shared_ptr<XTensor> Get(const string& name);

    /* get a parameter by its name */
    shared_ptr<XTensor> operator[] (const string& name);

    /* register a parameter with a unique name */
    void Register(const string& name, Dim dims, TENSOR_DATA_TYPE dataType);

    /* register a module */
    void Register(const string& prefix, const Model& module);

    /* print all parameters */
    void Print();

    /* load the model on device */
    void ToDevice(int devID);
};