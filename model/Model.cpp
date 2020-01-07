#include "Model.h"


/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* register a parameter with a unique name */
void Model::Register(const string& name, Dim dims, TENSOR_DATA_TYPE dataType)
{
    parameters.AddParameter(name, dims, dataType);
}

/* set devices for all parameters */
void Model::ToDevice(int devID)
{
    for (size_t i = 0; i < parameters.list.count; i++) {
        parameters.list[i]->SetDevice(devID);
    }
}

/* get a parameter by its name */
XTensor& Model::operator[](const char* name)
{
    return Get(name);
}

/* load a model from a binary file */
void Model::Load(const char* fn)
{
    CheckNTErrors(parameters.list.Size() > 0, "empty tensor list");

    FILE* file = fopen(fn, "rb");
    LongList offset(parameters.list.Size());

    /* check number of parameter */
    unsigned long int number;
    fread(&number, sizeof(number), 1, file);
    CheckNTErrors(number == parameters.list.Size(), "parameter number not matched");

    /* read offset from the file */
    fread(parameters.list.items, sizeof(long), offset.Size(), file);

    /* read parameters from the file */
    for (int i = 0; i < offset.Size(); i++) {
        parameters.list[i]->BinaryRead(file, offset[i]);
    }
    fclose(file);
}

/* dump a model to a binary file */
void Model::Dump(const char* fn)
{
    FILE* file = fopen(fn, "wb");
    
    /* dump number of parameter */
    unsigned long int number = parameters.list.Size();
    fwrite(&number, sizeof(number), 1, file);

    /* dump offset of parameters */
    unsigned long int offset = sizeof(number);
    for (int i = 0; i < parameters.list.Size(); i++) {
        if (i > 0) {
            offset += parameters.list[i - 1]->unitNum;
        }
        fwrite(&offset, sizeof(offset), 1, file);
    }

    /* dump parameters to the file */
    for (int i = 0; i < parameters.list.Size(); i++) {
        parameters.list[i]->BinaryDump(file);
    }
    fclose(file);
}

/* get a parameter by its name */
XTensor& Model::Get(const string& name)
{
    return *parameters.GetParameter(name);
}

/* add a parameter to the list */
void Parameter::AddParameter(const string& name, Dim dims, TENSOR_DATA_TYPE dataType)
{
    CheckNTErrors(GetParameter(name) == NULL, "the name must be unique");

    IntList dim;
    for (int i : dims) {
        dim.Add(i);
    }
    XTensor* p = NewTensorV2(dims.size(), dim.items, dataType);
    strcpy(p->name, name.c_str());
    list.Add(p);
}

/* get a parameter by its name */
XTensor* Parameter::GetParameter(const string& name)
{
    for (int i = 0; i < list.Size(); i++) {
        if (strcmp(list[i]->name, name.c_str()) == 0)
            return list[i];
    }

    /* if miss, return a null pointer */
    return NULL;
}

}