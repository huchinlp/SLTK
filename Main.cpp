//#include "sample/sltk/SLTKTrainer.h"
//#include "sample/sltk/SLTKUtility.h"
#include <fstream>
#include <iostream>
#include "model/Model.h"
#include "sample/sltk/StringUtil.h"
#include "tensor/core/getandset/SetData.h"
#include "sample/sltk/SLTKDataSet.h"


using namespace std;
using namespace nts;

void testModel()
{
    Model model;
    model.Register("transition", { 20, 20 }, X_FLOAT);
    model.ToDevice(0);
    model["transition"]->SetDataRand();
    model["transition"]->Dump(stderr);
    model.Dump("model.bin");
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
     
    //NERMain(argc, argv);
    testRead();

    return 0;
}
