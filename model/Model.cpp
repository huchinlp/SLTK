#include <vector>
#include "Model.h"
#include "../sample/sltk/StringUtil.h"
#include <iostream>

/* register a parameter with a unique name */
void Model::Register(const string& name, Dim dims, TENSOR_DATA_TYPE dataType)
{
    parameters.AddParameter(name, dims, dataType);
}

/* register a module */
void Model::Register(const string& prefix, const Model& module)
{
    for (auto p : module.parameters.list) {
        string newName = prefix + "." + p->name;
        strcpy(p->name, newName.c_str());
        parameters.list.push_back(p);
    }
}

/* print all parameters */
void Model::Print()
{
    ostringstream msg;
    size_t totalParams = 0;
    for (const auto& p:parameters.list) {
        auto shape = vector<int>(p->dimSize, p->dimSize+p->order);
        msg << p->name << " shape: (";
        size_t params = 1;
        for (int s : shape) {
            msg  << s << ",";
            params *= s;
        }
        totalParams += params;
        msg << ")\n";
    }
    cout << "total parameters: " << parameters.list.size() << " size: " << totalParams << "\n";
    cout << msg.str();
}

/* set devices for all parameters */
void Model::ToDevice(int devID)
{
    for (size_t i = 0; i < parameters.list.size(); i++) {
        parameters.list[i]->SetDevice(devID);
    }
}

/* get a parameter by its name */
XTensor& Model::operator[](const char* name)
{
    return Get(name);
}

/* 
load a model from a binary file 
a model file consists of three. parts:
part 1: number of offsets, int64_t
part 2: offsets of parameters, int64_t
part 3: parameters, float32_t
*/
void Model::Load(const char* fn)
{
    CheckNTErrors(parameters.list.size() > 0, "empty tensor list");

    FILE* file = fopen(fn, "rb");
    vector<int64_t> offsets(parameters.list.size());

    /* check number of parameter */
    int64_t number;
    fread(&number, sizeof(number), 1, file);
    CheckNTErrors(number == parameters.list.size(), "parameter number not matched");

    /* read parameter offsets from the file */
    fread(&offsets[0], sizeof(offsets[0]), offsets.size(), file);

    /* read parameters from the file */
    for (int i = 0; i < offsets.size(); i++) {
        parameters.list[i]->BinaryRead(file, offsets[i]);
    }
    fclose(file);
}

/* dump a model to a binary file */
void Model::Save(const char* fn)
{
    FILE* file = fopen(fn, "wb");

    /* dump number of parameter */
    int64_t number = parameters.list.size();
    fwrite(&number, sizeof(number), 1, file);

    /* dump offset of parameters */
    int64_t offset = 0;
    for (const auto& p:parameters.list) {
        offset = p->unitNum;
        fwrite(&offset, sizeof(offset), 1, file);
    }

    /* dump parameters to the file */
    for (int i = 0; i < parameters.list.size(); i++) {
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
    XTensor* p = NewTensorV2(int(dims.size()), dim.items, dataType);
    strcpy(p->name, name.c_str());
    list.push_back(p);
}

/* get a parameter by its name */
XTensor* Parameter::GetParameter(const string& name)
{
    for (int i = 0; i < list.size(); i++) {
        if (strcmp(list[i]->name, name.c_str()) == 0)
            return list[i];
    }

    /* if miss, return a null pointer */
    return NULL;
}