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

#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "StringUtil.h"
#include "SLTKDataSet.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/core/getandset/SetData.h"

using namespace std;

/* load data from column-format file */
void DataSet::LoadFromFile(const string& src)
{
    ifstream f(src, ios::in);

    /* read tokens */
    int pos = 0;
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto sentences = SplitString(str, "\n\n");

    buffers.reserve(sentences.size());

    /* convert tokens to ids */
    for (const auto& sent : sentences)
        buffers.emplace_back(SplitString(sent, "\n"));

    bufferSize = buffers.size();
}

/*
load a batch of sentences from the buffer
>>> batchSize - as it is
*/
vector<vector<string>> DataSet::LoadBatch(int batchSize)
{
    CheckNTErrors(batchSize > 0 && batchSize <= bufferSize, "invalid batch size");

    /* calculate the real batch size */
    int bsz = (cur + batchSize) > int(bufferSize) ? int(bufferSize) - cur : batchSize;

    /* load data to a mini-batch */
    vector<vector<string>> batch(buffers.begin() + cur, buffers.begin() + cur + bsz);
    
    cur += bsz;

    //sort(batch.begin(), batch.end(), [](auto& a, auto& b) {return a.size() > b.size();});

    return batch;
}

/* reset index of the current data entry */
void DataSet::Reset()
{
    cur = 0;
}

/*
constructor
>>> myDev - device id
>>> myShuffle - shuffle the data or not
*/
DataSet::DataSet(const string& src, bool myShuffle)
{
    isShuffled = myShuffle;
    LoadFromFile(src);
}

/* load a vocabulary from a file */
void Vocab::Load(const string& src)
{
    string line;
    ifstream f(src, ios::in);

    /* get the vocab size */
    f >> line;
    vocabSize = stol(line);

    for (int i = 0; i < vocabSize; i++) {
        string word, id;
        f >> word >> id;
        word2id[word] = stol(id);
        id2word[stol(id)] = word;
    }
    f.close();
}

/* save a vocabulary to a file */
void Vocab::Save(const string& src)
{
    ofstream f(src, ios::out);
    f << vocabSize;
    for (auto p : word2id) {
        f << p.first << "\t" << p.second;
    }
    f.close();
}