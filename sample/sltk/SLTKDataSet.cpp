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
#include "StringUtil.h"
#include "SLTKDataSet.h"
#include "../../tensor/XGlobal.h"
#include <algorithm>
#include "../../tensor/core/getandset/SetData.h"
#include <iostream>

using namespace std;

/* compare two length (descending) */
bool cmp(pair<int, int>& a, pair<int, int>& b)
{
    return (a.first - a.second) < (b.first - b.second);
};

/* load data from column-format file */
void DataSet::LoadFromFile(const string& src)
{
    ifstream f(src, ios::in);

    /* read tokens */
    int pos = 0;
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto sentences = SplitString(str, "\n\n");

    size_t tokenNumber = 0;
    for (const auto& s : sentences) {
        tokenNumber += s.size();
    }

    /* allocate buffers */
    for (int i = 0; i < vocabs.size(); i++) {
        int* buffer = new int[tokenNumber];
        buffers.push_back(buffer);
    }

    /* convert tokens to ids */
    for (auto& sent : sentences) {
        auto lines = SplitString(sent, "\n");
        indices.push_back({ pos, pos + lines.size() });
        for (auto& line : lines) {
            auto tokens = SplitString(line, " ");
            for (size_t i = 0; i < tokens.size(); i++) {
                buffers[i][pos] = vocabs[i]->word2id.at(tokens[i]);
            }
            pos++;
        }
    }

    /* sort index by length (descending) */
    sort(indices.begin(), indices.end(), cmp);
    bufferSize = indices.size();
}

/*
load 'field' batches from the buffer
>>> list - a list to load these batches
>>> batchSize - as it is
*/
void DataSet::LoadBatch(TensorList& list, int batchSize)
{
    CheckNTErrors(batchSize > 0 && batchSize <= bufferSize, "invalid batch size");

    /* calculate the real batch size */
    int bsz = (cur + batchSize) > int(indices.size()) ? int(indices.size()) - cur : batchSize;

    /* shuffle the whole buffer */
    if (isShuffled && bsz > 1) {
        std::random_device rd;
        std::mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);

        sort(indices.begin() + cur, indices.begin() + cur + bsz, cmp);
    }

    /* load data to batches */
    int maxLen = (indices[cur].second - indices[cur].first);
    for (int n = 0; n < vocabs.size(); n++) {
        XTensor* batch = NewTensor2DV2(bsz, int(maxLen), X_INT, devID);
        int* data = new int[batch->unitNum];
        fill_n(data, batch->unitNum, vocabs[n]->word2id.at("<PAD>"));
        for (int i = 0; i < bsz; i++) {
            auto pos = Locate(cur + i, n);
            copy_n(pos.first, pos.second, data + i * maxLen);
        }
        batch->SetData(data, batch->unitNum);
        list.Add(batch);
        delete[] data;
    }
    cur += batchSize;
}

/*
get the position of an example by its field and index
>>> index - the index of an example
>>> field - the index of a field
<<< the first is the start postion, the second is the length
*/
pair<int*, int> DataSet::Locate(int index, int field)
{
    CheckNTErrors(vocabs.size() > field, "invalid field");
    int fieldLen = indices[index].second - indices[index].first;
    return { buffers[field] + indices[index].first, fieldLen };
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
DataSet::DataSet(int myDev, bool myShuffle, const string& src)
{
    devID = myDev;
    isShuffled = myShuffle;
    LoadFromFile(src);
}

/* load a vocabulary from a file */
void Vocab::Load(const char* src)
{
    string line;
    ifstream f(src, ios::in);
    
    /* get the vocab size */
    f >> line;
    vocabSize = stoll(line);

    for (int i = 0; i < vocabSize; i++) {
        string word, id;
        f >> word >> id;
        word2id[word] = stoll(id);
        id2word[stoll(id)] = word;
    }
    f.close();
}

/* save a vocabulary to a file */
void Vocab::Save(const char* src)
{
    ofstream f(src, ios::out);
    f << vocabSize;
    for (auto p : word2id) {
        f << p.first << "\t" << p.second;
    }
    f.close();
}
