/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /*
  * $Created by: HU Chi (huchinlp@foxmail.com) 2020-01-03
  */
#pragma once

#include <string>
#include <vector>
#include <utility>
#include <sstream>

using namespace std;

/* Splits a string based on the given delimiter string. Each pair in the
 * returned vector has the start and past-the-end positions for each of the
 * parts of the original string. Empty fields are not represented in the output. */
vector<uint64_t> SplitToPos(const string& s, const string& delimiter);
vector<int64_t> SplitInt(const string& s, const string& delimiter);
vector<float> SplitFloat(const string& s, const string& delimiter);
vector<string> SplitString(const string& s, const string& delimiter);

/* concat variable parameters to a string */
inline void AddToStream(ostringstream&) {}

template<typename T, typename... Args>
inline void AddToStream(ostringstream& a_stream, T&& a_value, Args&& ... a_args)
{
    a_stream << forward<T>(a_value);
    AddToStream(a_stream, forward<Args>(a_args)...);
}

template<typename... Args>
inline string ConcatString(Args&& ... a_args)
{
    ostringstream s;
    AddToStream(s, forward<Args>(a_args)...);
    return s.str();
}

/* string transformation */
vector<string> Transform(const string& str);

/* load arguments */
void ShowParams(int argc, const char** argv);
int LoadParamInt(int argc, const char** argv, const char* name, int defaultP);
bool LoadParamBool(int argc, const char** argv, const char* name, bool defaultP);
float LoadParamFloat(int argc, const char** argv, const char* name, float defaultP);
const char* LoadParamString(int argc, const char** argv, const char* name, const char* defaultP);