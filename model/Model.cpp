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
    for (auto p : module.parameters.paramList) {
        string name = prefix + "." + p->name;
        parameters.paramList.push_back(p);
        parameters.nameList.emplace_back(name);
    }
}

/* print all parameters */
void Model::Print()
{
    ostringstream msg;
    size_t totalParams = 0;
    for (int i = 0; i < parameters.paramList.size();i++) {
        auto p = parameters.paramList[i];
        auto name = parameters.nameList[i];
        auto shape = vector<int>(p->dimSize, p->dimSize + p->order);
        msg << name << " shape: (";
        size_t params = 1;
        for (int s : shape) {
            msg << s << ",";
            params *= s;
        }
        totalParams += params;
        msg << ")\n";
    }
    cout << "total parameters: " << parameters.paramList.size() << " size: " << totalParams << "\n";
    cout << msg.str();
}

/* set devices for all parameters */
void Model::ToDevice(int devID)
{
    for (size_t i = 0; i < parameters.paramList.size(); i++) {
        if(parameters.paramList[i]->devID != devID)
            parameters.paramList[i]->SetDevice(devID);
    }
}

/* get a parameter by its name */
shared_ptr<XTensor> Model::operator[](const string& name)
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
    CheckNTErrors(parameters.paramList.size() > 0, "empty tensor list");

    FILE* file = fopen(fn, "rb");
    vector<int64_t> offsets(parameters.paramList.size());

    /* check number of parameter */
    int64_t number;
    fread(&number, sizeof(number), 1, file);
    CheckNTErrors(number == parameters.paramList.size(), "parameter number not matched");

    /* read parameter offsets from the file */
    fread(&offsets[0], sizeof(offsets[0]), offsets.size(), file);

    /* read parameters from the file */
    for (int i = 0; i < offsets.size(); i++) {
        parameters.paramList[i]->BinaryRead(file, offsets[i]);
    }
    fclose(file);
}

/* dump a model to a binary file */
void Model::Save(const char* fn)
{
    FILE* file = fopen(fn, "wb");

    /* dump number of parameter */
    int64_t number = parameters.paramList.size();
    fwrite(&number, sizeof(number), 1, file);

    /* dump offset of parameters */
    int64_t offset = 0;
    for (const auto& p : parameters.paramList) {
        offset = p->unitNum;
        fwrite(&offset, sizeof(offset), 1, file);
    }

    /* dump parameters to the file */
    for (int i = 0; i < parameters.paramList.size(); i++) {
        parameters.paramList[i]->BinaryDump(file);
    }
    fclose(file);
}

/* get a parameter by its name */
shared_ptr<XTensor> Model::Get(const string& name)
{
    return parameters.GetParameter(name);
}

/* add a parameter to the list */
void Parameter::AddParameter(const string& name, Dim dims, TENSOR_DATA_TYPE dataType)
{
    CheckNTErrors(GetParameter(name) == nullptr, "the name must be unique");

    IntList dim;
    for (int i : dims) {
        dim.Add(i);
    }
    auto p = make_shared<XTensor>();
    InitTensorV2(p.get(), int(dim.Size()), dim.items, dataType);
    nameList.push_back(name);
    paramList.push_back(p);
}

/* get a parameter by its name */
shared_ptr<XTensor> Parameter::GetParameter(const string& name)
{
    for (int i = 0; i < paramList.size(); i++) {
        if (name == nameList[i])
            return paramList[i];
    }

    /* if miss, return a null pointer */
    return nullptr;
}