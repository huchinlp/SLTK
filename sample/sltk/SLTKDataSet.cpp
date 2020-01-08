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

/*
build the vocabulary from a file
format (one token per line):
token tag1 tag2 ...
*/
Vocab::Vocab(const char* src)
{
    tokenNumber = 0;
    ifstream f(src, ios::in);

    vector<string> preDefinedTokens = { "<PAD>" };
    int startID = int(preDefinedTokens.size());

    /* get all sentences */
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto sentences = SplitString(str, "\n\n");

    /* get the number of fields */
    fields = int(SplitString(sentences[0], "\n").size());
    for (int i = 0; i < fields; i++) {
        word2IDs.push_back(unordered_map<string, int>());
        id2Words.push_back(unordered_map<int, string>());
    }
    vector<int> idx;
    for (int i = 0; i < word2IDs.size(); i++) {
        idx.push_back(int(preDefinedTokens.size()));
    }

    /* assign indices to tokens and tags */
    for (auto& sent : sentences) {
        auto lines = SplitString(sent, "\n");
        tokenNumber += lines.size();
        for (auto& line : lines) {
            auto tokens = SplitString(line, " ");
            for (size_t i = 0; i < tokens.size(); i++) {
                auto token = tokens[i];
                if (word2IDs[i].find(token) == word2IDs[i].end()) {
                    word2IDs[i][token] = idx[i];
                    id2Words[i][idx[i]++] = token;
                }
            }
        }
    }

    /* add pre-defined tokens to the dict */
    for (int n = 0; n < word2IDs.size(); n++) {
        vocabSizes.push_back(int(word2IDs[n].size()));
        for (int i = 0; i < preDefinedTokens.size(); i++) {
            word2IDs[n][preDefinedTokens[i]] = i;
            id2Words[n][i] = preDefinedTokens[i];
        }
    }
}

/* load data from column-format file */
void DataSet::LoadFromFile(const string& src)
{
    vocab = new Vocab(src.c_str());

    for (int i = 0; i < vocab->fields; i++) {
        int* buffer = new int[vocab->tokenNumber];
        buffers.push_back(buffer);
    }

    ifstream f(src, ios::in);

    /* read tokens */
    int pos = 0;
    std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto sentences = SplitString(str, "\n\n");

    /* convert tokens to ids */
    for (auto& sent : sentences) {
        auto lines = SplitString(sent, "\n");
        indices.push_back({ pos, pos + lines.size() });
        for (auto& line : lines) {
            auto tokens = SplitString(line, " ");
            for (size_t i = 0; i < tokens.size(); i++) {
                buffers[i][pos] = vocab->word2IDs[i].at(tokens[i]);
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
    for (int n = 0; n < vocab->fields; n++) {
        XTensor* batch = NewTensor2DV2(bsz, int(maxLen), X_INT, devID);
        int* data = new int[batch->unitNum];
        fill_n(data, batch->unitNum, vocab->word2IDs[n].at("<PAD>"));
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
    CheckNTErrors(vocab->fields > field, "invalid field");
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

/* de-constructor */
DataSet::~DataSet()
{
    for (int i = 0; i < buffers.size(); i++) {
        delete buffers[i];
    }
    if (vocab != nullptr) {
        delete vocab;
    }
}