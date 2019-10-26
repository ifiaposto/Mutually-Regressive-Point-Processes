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

#ifndef EVENTSEQUENCE_HPP
#define EVENTSEQUENCE_HPP

#include <boost/filesystem.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <map>
#include <fstream>
#include <string>
#include <iostream>
#include <cctype>
#include "debug.hpp"




class Event;
class EventSequence{
	public:
	
		std::string name;
		unsigned int K; //number of event types
		double start_t; //start of the observation window
		double end_t; //end of the observation window
		unsigned N; //number of observed/realized events in the sequence
		unsigned Nt=0; //number of thinned/latent events in the sequence
	
		//structures for the observed events
		std::vector<std::map<double,Event *>> type; //it contains the events of each type separated, and ordered according to their arrival time
		std::map<double,Event *> full; //it contains the events of all types merged, and ordered according to their arrival time
		//structures for the thinned/latent events, similar structures as for the observed events
		std::vector<std::map<double,Event *>> thinned_type;
		std::map<double,Event *> thinned_full;
		

		bool observed=false; //false if thinned_ structures may not contain all or some of the latent events of the sequence, or if there is
		//an event whose parent is unknown
		
		//TODO: include field observed in the constructorss argument list
		EventSequence();//empty constructor
		EventSequence(const EventSequence &s); //copy constructor
		EventSequence(EventSequence &&s); //move constructor
		
		EventSequence(std::string name);
		EventSequence(std::string name, unsigned int k);//initialize the name and the number of types k for the event sequence
		EventSequence(std::string name, unsigned int k, double s, double e);//initialize the name, the number of types k for the event sequence and the observation window [s,e]
		EventSequence(std::string name, unsigned int k, double s, double e, std::map<double,Event *> seq);
		EventSequence(std::string name, unsigned int k, double s, double e, std::map<double,Event *> o_seq, std::map<double,Event *> t_seq);
		EventSequence(std::string name, unsigned int k, double s, double e, std::vector<std::map<double,Event *>> type_seqs);
		EventSequence(std::string name, unsigned int k, double s, double e, std::vector<std::map<double,Event *>> o_type_seqs, std::vector<std::map<double,Event *>> t_type_seqs);
		EventSequence(std::string name, unsigned int k, double s, double e, std::ifstream & file);//constructor which initializes the event sequence from a csv file

		EventSequence(unsigned int k);
		EventSequence(unsigned int k, double s, double e);
		EventSequence(unsigned int k, double s, double e, std::map<double,Event *> seq);
		EventSequence(unsigned int k, double s, double e, std::map<double,Event *> o_seq, std::map<double,Event *> t_seq);
		EventSequence(unsigned int k, double s, double e, std::vector<std::map<double,Event *>> type_seqs);
		EventSequence(unsigned int k, double s, double e, std::vector<std::map<double,Event *>> o_type_seqs, std::vector<std::map<double,Event *>>  t_type_seqs);
		EventSequence(unsigned int k, double s, double e, std::ifstream & file);
		
		//TODO: define move operator

		EventSequence& operator=(const EventSequence &s); //copy assignment (it creates deep copies of the events)
		EventSequence& operator=(EventSequence &&s); //move assignment

		~EventSequence();//TODO: why if I define the destructor the move instead of the copy assignment operator is used????

		void print(std::ostream &file, unsigned int file_type=0, bool observed=true) const;//0 for txt, 1 for csv
		void plot()const;
		void plot(const std::string & dir_path_str)const;
		void save(std::ofstream &file) const;
		void load(std::ifstream &file, unsigned int file_type=0);//0 from serialized object, 1 from csv file
		void summarize(std::ofstream &file);
		void flush();
		void flushThinnedEvents();

		std::map<double, Event*>::iterator pruneCluster(Event *e);
		void returnCluster(Event *e, EventSequence &s) const;
		void pruneGeneration(Event *e);
		
		void addEvent(Event *e, Event *p=0);
		Event * findEvent(double t) const ;

		std::map<double, Event*>::iterator delEvent(Event *e);
		void changeParent(Event *e, Event *p);
		void setParent(Event *e, Event *p);
		
		
		//it returns the number if events of type k2 triggered by events of type k.
		unsigned int countTriggeredEvents(int k, int k2) const;
		//it returns the number of clusters in the sequence.
		unsigned int countClusters() const;
		//it puts the arrival times of certain type k of events in the vector time.
		void getArrivalTimes(unsigned int k, boost::numeric::ublas::vector<double> & time) const;

		
	private:
		
	   friend class boost::serialization::access;

	   void serialize(boost::archive::text_iarchive &ar, const unsigned int version);
	   void serialize(boost::archive::text_oarchive &ar, const unsigned int version);
	   void serialize(boost::archive::binary_iarchive &ar, const unsigned int version);
	   void serialize(boost::archive::binary_oarchive &ar, const unsigned int version);
	   void plot_(const std::string & dir_path_str)const;

};

class Event{

	public:
		Event();
		Event(const Event &e); //copy constructor
		Event(unsigned long id,int l, double t, unsigned int k);
		Event(unsigned long id, int l, double t, unsigned int k, Event * p);
		Event(unsigned long id, int l, double t, unsigned int k, bool o, Event * p);
		Event(unsigned long id, int l, double t, unsigned int k, EventSequence *seq);
		Event(unsigned long id, int l, double t, unsigned int k, Event * p, EventSequence *seq);
		Event(unsigned long id, int l, double t, unsigned int k, bool o, Event * p, EventSequence *seq);
		Event(int l, double t, unsigned int k);
		Event(int l, double t, unsigned int k, Event * p);
		Event(int l, double t, unsigned int k, bool o, Event * p);
		Event(int l, double t, unsigned int k, EventSequence *seq);
		Event(int l, double t, unsigned int k, Event * p, EventSequence *seq);
		Event(int l, double t, unsigned int k, bool o, Event * p, EventSequence *seq);
		Event& operator=(const Event &s);
		~Event();
		
		void print(std::ostream &file, unsigned int file_type=0, bool observed=true) const;//0 for txt, 1 for csv
		void save(std::ofstream &file) const;
		void load(std::ifstream &file);
		
		void removeChild(Event *e);
		void addChild(Event *e);
		
		unsigned long id=0;
		int type;
		double time;
		unsigned int K=0;
		//bool observed=true; //false if the event is thinned
		bool observed=true;
		Event *parent=0; 	//!=0 if it was trigerred by another event, 0 if it is an immigrant (belongs to the background process)
		EventSequence *offsprings=0;
		#if DEBUG_LEARNING
		double p_observed; //realization probability
		#endif

	private:
		
	    friend class boost::serialization::access;
	    
	    void serialize(boost::archive::text_oarchive &ar, const unsigned int version);
	    void serialize(boost::archive::text_iarchive &ar, const unsigned int version);
	    void serialize(boost::archive::binary_oarchive &ar, const unsigned int version);
	    void serialize(boost::archive::binary_iarchive &ar, const unsigned int version);

};

#endif /* EVENTSEQUENCE_HPP_ */
