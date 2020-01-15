#include "model/Model.h"
#include "sample/sltk/SLTKDataSet.h"
#include "sample/sltk/SLTKModel.h"
#include "sample/sltk/StringUtil.h"
#include "tensor/core/getandset/SetData.h"
#include <fstream>
#include <iostream>
#include <memory>
using namespace std;
using namespace nts;

void testModel()
{
    Model model;
    model.Register("transition", { 20, 20 }, X_FLOAT);
    model.ToDevice(0);
    model["transition"]->SetDataRand();
    model.Save("model.bin");
    model.Print();
}

bool Exists(const char* file)
{
    FILE* f = fopen(file, "r");
    if (f == NULL)
        return false;
    else {
        fclose(f);
        return true;
    }
}

shared_ptr<SequenceTagger> BuildModel(const int argc, const char** argv)
{
    int devID = LoadParamInt(argc, argv, "devID", 0);
    int tagNum = LoadParamInt(argc, argv, "tagNum", 28);
    int embSize = LoadParamInt(argc, argv, "embSize", 400);
    int rnnLayer = LoadParamInt(argc, argv, "rnnLayer", 1);
    int hiddenSize = LoadParamInt(argc, argv, "hiddenSize", 256);
    
    auto embFile = LoadParamString(argc, argv, "embFile", "wnut17.emb");
    auto modelFile = LoadParamString(argc, argv, "modelFile", "wnut17.model");

    auto embeddings = make_shared<StackEmbedding>(devID, vector<const char*>{"wnut17crawl.emb", "wnut17twitter.emb"});
    auto model = make_shared<SequenceTagger>(devID, rnnLayer, hiddenSize, tagNum, embSize, embeddings);
    model->Load(modelFile);
    model->ToDevice(0);
    return model;
}

void Predict(const int argc, const char** argv)
{
    auto model = BuildModel(argc, argv);
    int batchSize = LoadParamInt(argc, argv, "batchSize", 24);
    auto srcFile = LoadParamString(argc, argv, "src", "test.txt");
    auto tgtFile = LoadParamString(argc, argv, "tgt", "test.txt.res");

    DataSet dataSet(srcFile);
    for (auto i = 0; i < dataSet.bufferSize; i += batchSize) {
        /* input sequences */
        auto src = dataSet.LoadBatch(batchSize);

        /* label sequences */
        auto labels = model->Predict(src);
        //model->DumpResult(src, labels, tgtFile);
    }
}


int main(const int argc, const char** argv)
{
    Predict(argc, argv);
    return 0;
}