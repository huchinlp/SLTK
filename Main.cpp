#include "SLTKTrainer.h"
#include "SLTKUtility.h"

using namespace ner;

int main(const int argc, const char **argv)
{
    CALL;

    srand(clock());
    NERMain(argc, argv);

    _CrtDumpMemoryLeaks();
    OK;
    return 0;
}
