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

#ifndef __STRING_UTIL_H__
#define __STRING_UTIL_H__

#include <string>
#include <vector>

using namespace std;

/* Splits a string based on the given delimiter string. Each pair in the
 * returned vector has the start and past-the-end positions for each of the
 * parts of the original string. Empty fields are not represented in the output.
 */
vector<uint64_t> SplitToPos(const string& s, const string& delimiter);
vector<int64_t> SplitInt(const string& s, const string& delimiter);
vector<float> SplitFloat(const string& s, const string& delimiter);
vector<string> SplitString(const string& s, const string& delimiter);

/* concatenate string, int, float... */
void addToStream(std::ostringstream&);

template<typename T, typename... Args>
void addToStream(std::ostringstream& a_stream, T&& a_value, Args&& ... a_args);

template<typename... Args>
std::string concat(Args&& ... a_args);

#endif // __STRING_UTIL_H__