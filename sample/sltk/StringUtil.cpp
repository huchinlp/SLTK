/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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

#include "StringUtil.h"

 /*
 split string by delimiter, this will return indices of all sub-strings
 >>> s - the original string
 >>> delimiter - as it is
 >>> a - the indices of all sub-strings
 */
vector<uint64_t> SplitToPos(const string& s, const string& delimiter)
{
    vector<uint64_t> indices;
    if (delimiter.length() == 0) {
        indices.push_back(0);
    }
    size_t pos = 0;
    uint64_t start = 0;
    while ((pos = s.find(delimiter, start)) != string::npos) {
        if (pos != start) {
            indices.push_back(start);
        }
        start = pos + delimiter.length();
    }
    if (start != s.length()) {
        indices.push_back(start);
    }
    return indices;
}

/* split a string to a int64_t list */
vector<int64_t> SplitInt(const string& s, const string& delimiter)
{
    vector<int64_t> values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.size(); i++) {
        values.push_back(strtol(s.data() + indices[i], nullptr, 10));
    }
    return values;
}

/* split a string to a float list */
vector<float> SplitFloat(const string& s, const string& delimiter)
{
    vector<float> values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.size(); i++) {
        values.push_back(strtof(s.data() + indices[i], nullptr));
    }
    return values;
}

/* split a string to a sub-string list */
vector<string> SplitString(const string& s, const string& delimiter)
{
    vector<string> values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.size(); i++) {
        auto offset = (i != (indices.size() - 1)) ? indices[i + 1] - indices[i] - delimiter.size() : s.size() - indices[i];
        values.push_back(s.substr(indices[i], offset));
    }
    return values;
}