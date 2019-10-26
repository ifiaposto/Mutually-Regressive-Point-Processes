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
#ifndef STRUCT_UTILS_HPP_
#define STRUCT_UTILS_HPP_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>

//templates can only be instantiated in header files
template<typename KeyType, typename ValueType, typename Lambda>
inline void erase_if_all_map(std::map<KeyType,ValueType> & map, Lambda f){


	for(auto it = map.begin(), ite = map.end(); it != ite;)
	{
	  if(f(*it)){
	    it = map.erase(it);
	  }
	  else
	    ++it;
	}
}

template<typename KeyType, typename ValueType, typename Lambda>
inline typename std::map<KeyType, ValueType>::iterator erase_if_map(std::map<KeyType,ValueType> & map, Lambda f){

	typename std::map<KeyType, ValueType>::iterator it;
	for(auto ite = map.end(),it = map.begin(); it != ite;){
	  if(f(*it)){
	    it = map.erase(it);
	    return it;
	  }
	  else
	    ++it;
	}
	return it;
}

//it drops the key and converts the map to a vector
template<typename T1, typename T2>
inline void map_to_vector(const T1 &m, T2 &v){
	v.reserve(m.size()+v.size());
   for( typename T1::const_iterator it = m.begin(); it != m.end(); ++it ) {
        v.push_back( it->second );
    }
}

//it merges two boost::ublas vectors and puts the result at the end of the first
template <typename T>
inline boost::numeric::ublas::vector<T> &operator+=(boost::numeric::ublas::vector<T> &A, const boost::numeric::ublas::vector<T> &B)
{
	unsigned int i=A.size();
	A.resize(A.size()+B.size());
	for(auto iter=B.begin();iter!=B.end();iter++)//TODO: more elegant/efficient way to do that???
		A.insert_element(i++,*iter);
    return A;
}

//it merges the boost::ublas vectors contained in the std::vector in one boost::ublas vector, it returns a new vector with the result
template <typename T>
inline boost::numeric::ublas::vector<T> merge_ublasv(const std::vector<boost::numeric::ublas::vector<T>> &A)
{
	boost::numeric::ublas::vector<T> B;
	for(auto iter=A.begin();iter!=A.end();iter++){
		 B+=*iter;
	}
	return B;
}

//it forces vector to clear its memory
template <typename T>
void forced_clear( T & t ) {
    T tmp;
    t.swap( tmp );
}

#endif /* STRUCT_UTILS_HPP_ */
