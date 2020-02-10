#include <fstream>
#include <iostream>
#include <memory>

#include "model/Model.h"
#include "sample/sltk/SLTKDataSet.h"
#include "sample/sltk/SLTKModel.h"
#include "sample/sltk/StringUtil.h"
#include "tensor/core/getandset/SetData.h"
#include "tensor/core/movement/CopyIndexed.h"

using namespace std;
using namespace nts;

shared_ptr<SequenceTagger> BuildModel(const int argc, const char** argv)
{
    int devID = LoadParamInt(argc, argv, "devID", 0);
    int tagNum = LoadParamInt(argc, argv, "tagNum", 29);
    int embSize = LoadParamInt(argc, argv, "embSize", 400);
    int rnnLayer = LoadParamInt(argc, argv, "rnnLayer", 1);
    int hiddenSize = LoadParamInt(argc, argv, "hiddenSize", 256);

    auto embFile = LoadParamString(argc, argv, "embFile", "wnut17.emb");
    auto tagVocab = LoadParamString(argc, argv, "tagVocab", "wnut17.tag.vocab");
    auto modelFile = LoadParamString(argc, argv, "modelFile", "wnut17.model");
    auto emb1 = LoadParamString(argc, argv, "modelFile", "wnut17crawl.emb");
    auto emb2 = LoadParamString(argc, argv, "modelFile", "wnut17twitter.emb");

    auto embeddings = make_shared<StackEmbedding>(devID, vector<const char*>{emb1, emb2});
    auto model = make_shared<SequenceTagger>(devID, rnnLayer, hiddenSize, tagNum, embSize, embeddings, tagVocab);

    model->Load(modelFile);
    model->ToDevice(0);
    return model;
}

void Predict(const int argc, const char** argv)
{
    auto model = BuildModel(argc, argv);
    int batchSize = LoadParamInt(argc, argv, "batchSize", 1);
    auto srcFile = LoadParamString(argc, argv, "src", "tiny.txt");
    auto tgtFile = LoadParamString(argc, argv, "tgt", "res.txt");

    DataSet dataSet(srcFile);

    for (auto i = 0; i < dataSet.bufferSize; i += batchSize) {
        /* input sequences */
        auto src = dataSet.LoadBatch(batchSize);

        /* label sequences */
        auto labels = model->Predict(src);

        /* dump tag results to a file */
        model->DumpResult(src, labels, tgtFile);
    }
}

int main(const int argc, const char** argv)
{
    Predict(argc, argv);

    return 0;
}