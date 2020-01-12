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
    model["transition"].SetDataRand();
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
    int devID = 0;
    int embSize = 400;
    int hiddenSize = 256;
    int rnnLayer = 1;
    int tagNum = 28;

    auto embeddings = make_shared<Embedding>(devID);
    embeddings->Load("wnut17.emb");
    //embeddings->LoadWordEmbedding({"wnut17crawl", "wnut17twitter"});
    //embeddings->LoadPretrainedLM({"news-forward", "news-backward"});

    auto model = make_shared<SequenceTagger>(rnnLayer, hiddenSize, tagNum, embSize, embeddings);
    model->Load("wnut17.bin");
    model->ToDevice(0);

    return model;
}

int main(const int argc, const char** argv)
{
    auto model = BuildModel(argc, argv);
    DataSet dataSet("test.txt");
    int batchSize = 24;
    for (auto i = 0; i < dataSet.bufferSize; i += batchSize) {
        /* input sequences */
        auto src = dataSet.LoadBatch(batchSize);

        /* label sequences */
        auto labels = model->Predict(src[0]);
        model.DumpResult("res.txt");
    }
    return 0;
}