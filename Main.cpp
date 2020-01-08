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
    model.Dump("model.bin");
    model.Print();
}

void testSequenceTagger()
{
    int embSize = 1;
    int hiddenSize = 1;
    int rnnLayer = 1;
    int tagNum = 1;
    SequenceTagger model(rnnLayer, hiddenSize, tagNum, embSize, nullptr, 0., 0., 0.);
    model.ToDevice(0);
    model.Dump("model.bin");
    model.Load("model.bin");
    model.Print();
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