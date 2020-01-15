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
  * $Created by: HU Chi (huchinlp@foxmail.com) 2020.01.02
  */

#include "StringUtil.h"
#include "SLTKEmbedding.h"
#include "../../tensor/core/CHeader.h"
#include <iostream>

/*
load pre-trained embeddings from file
>>> file - the pre-trained embeddings file
*/
void Embedding::LoadWordEmbedding(const char* file)
{
    /* load embeddings vocab */
    embVocab.Load(ConcatString(file, ".vocab"));

    /* load embeddings */
    FILE* embFile = fopen(file, "rb");
    fread(&vocabSize, sizeof(vocabSize), 1, embFile);
    fread(&embSize, sizeof(embSize), 1, embFile);

    InitTensor2DV2(&vec, vocabSize, embSize, X_FLOAT, devID);
    vec.BinaryRead(embFile, vocabSize * embSize);
    fclose(embFile);
}

/*
constructor
>>> myDevID - device
>>> file - the pre-trained embeddings file
*/
Embedding::Embedding(int myDevID, const char* file)
{
    devID = myDevID;

    LoadWordEmbedding(file);
}

/*
set word embeddings for a batch of sentences
>>> input - the input sentences
*/
XTensor Embedding::Embed(const vector<vector<string>>& input)
{
    XTensor idx;
    int bsz = input.size();
    int maxLen = 0;
    for (const auto sent : input)
        maxLen = max(maxLen, int(sent.size()));
    InitTensor2DV2(&idx, bsz, maxLen, X_INT, devID);
    
    int* indices = new int[bsz * maxLen];
    memset(indices, 0, sizeof(int) * bsz * maxLen);
    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < input[i].size(); j++) {
            if (embVocab.word2id.find(input[i][j]) != embVocab.word2id.end())
                indices[i * maxLen + j] = embVocab.word2id[input[i][j]];
        }
    }
    idx.SetData(indices, bsz * maxLen);
    delete[] indices;

    return Gather(vec, idx);
}

/* get embeddings of the inputs */
XTensor StackEmbedding::Embed(const vector<vector<string>>& input)
{
    auto emb = staticEmbeddings[0]->Embed(input);
    for (int i = 1; i < staticEmbeddings.size();i++) {
        emb = Concatenate(emb, staticEmbeddings[i]->Embed(input), 2);
    }

    return emb;
}

/* 
constructor 
>>> myDevID - device
>>> files - a list of pre-trained embedding files
*/
StackEmbedding::StackEmbedding(int myDevID, vector<const char*> files)
{
    devID = myDevID;
    for (auto file : files)
        staticEmbeddings.push_back(new Embedding(devID, file));
}

/* de-constructor */
StackEmbedding::~StackEmbedding()
{
    for (auto emb : staticEmbeddings)
        delete emb;
}

/* load pretrained LM from files */
void LanguageModel::LoadPretrainedLM(const char* file)
{
}

/* get the contextual representation of inputs */
void LanguageModel::GetRepresentation(const vector<vector<string>>& input)
{
}

/*
constructor
>>> myDevID - device
>>> file - the pre-trained embeddings file
*/
LanguageModel::LanguageModel(int myDevID, const char* file)
{
    devID = myDevID;

    LoadPretrainedLM(file);
}