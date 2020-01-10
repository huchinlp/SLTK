#include "model/Model.h"
#include "sample/sltk/SLTKDataSet.h"
#include "sample/sltk/SLTKModel.h"
#include "sample/sltk/StringUtil.h"
#include "tensor/core/getandset/SetData.h"
#include <fstream>
#include <iostream>
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

void testSequenceTagger()
{
    int embSize = 400;
    int hiddenSize = 256;
    int rnnLayer = 1;
    int tagNum = 28;
    SequenceTagger model(rnnLayer, hiddenSize, tagNum, embSize, nullptr);
    model.ToDevice(0);
    model["SequenceTagger.RNN2Tag.Bias"].SetZeroAll();
    SetDataFixed(model["SequenceTagger.RNN2Tag.Bias"], 1.0);
    model.Load("wnut17.bin");
    model.Print();
    /*model["SequenceTagger.Embedding2NN.Bias"].Dump(stderr);*/
    model["SequenceTagger.RNN2Tag.Bias"].Dump(stderr);
}

void testRead()
{
    DataSet dataSet(0, false, "file.txt");
    int batchSize = 2;
    for (auto i = 0; i < dataSet.bufferSize; i += batchSize) {
        TensorList list;
        dataSet.LoadBatch(list, batchSize);
        auto input = list[0];
        auto target = list[1];
        input->Dump(stderr);
        target->Dump(stderr);
    }
    
}

int main(const int argc, const char** argv)
{
    testSequenceTagger();
    return 0;
}