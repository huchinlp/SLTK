/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-01-03
 */

#pragma once

#include <cstdio>
#include <memory>
#include <unordered_map>
#include "../../tensor/XTensor.h"

using namespace std;
using namespace nts;

constexpr int UNK = 0;
constexpr int PAD = 1;
constexpr int SOS = 2;
constexpr int EOS = 3;

struct Dict
{
    string word;
    int id;
};

/* the vocabulary class that supports variable fileds */
struct Vocab
{
    int vocabSize;

    unordered_map<string, int> word2id;
    unordered_map<int, string> id2word;

    /* load a vocabulary from a file */
    void Load(const string& src);

    /* save a vocabulary to a file */
    void Save(const string& src);
};

/* the dataset class for sequence labeling */
class DataSet
{
private:
    /* tokens and tags */
    vector<vector<string>> buffers;

    /* current index for batching */
    int cur = 0;

    /* use shuffled batch or not */
    bool isShuffled = false;

    /* load dataset from a text file (column-fomat) */
    void LoadFromFile(const string& src);

    /* reset index of the current data entry */
    void Reset();

public:

    /* buffer size */
    size_t bufferSize;

    /* load a batch of sentences from the buffer */
    vector<vector<string>> LoadBatch(int batchSize);

    /* constructor */
    DataSet(const string& src, bool myShuffle = false);
};

/* compare two length (descending) */
bool cmp(pair<int, int>& a, pair<int, int>& b);