/*
==============================================================================
 * Copyright 2019
 * Author: Ifigeneia Apostolopoulou iapostol@andrew.cmu.edu, ifiaposto@gmail.com.
 * All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 *    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ==============================================================================
 */
#ifndef DEBUG_HPP_
#define DEBUG_HPP_

#define DEBUG_LEARNING   0

#define LOGL_PROFILING 0


#include <pthread.h>

extern std::ofstream debug_learning_file;

#if DEBUG_LEARNING
static pthread_mutex_t debug_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif

#endif /* DEBUG_HPP_ */
