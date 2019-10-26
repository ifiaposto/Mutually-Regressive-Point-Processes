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

#include "EventSequence.hpp"
#include <iostream>

#include "plot_utils.hpp"
#include "struct_utils.hpp"

EventSequence& EventSequence::operator=(EventSequence &&s){
	name=s.name;
	K=s.K;
	start_t=s.start_t;
	end_t=s.end_t;
	N=s.N;
	Nt=s.Nt;
	observed=s.observed;

	type=s.type;
	full=s.full;
	thinned_type=s.thinned_type;
	thinned_full=s.thinned_full;

	s.N=0;
	s.Nt=0;
	s.full.clear();
	s.thinned_full.clear();
	for(auto t_iter=s.type.begin();t_iter!=s.type.end();t_iter++){
		t_iter->clear();
	}
	s.type.clear();

	for(auto t_iter=s.thinned_type.begin();t_iter!=s.thinned_type.end();t_iter++){
		t_iter->clear();
	}
	s.type.clear();
	return *this;
}

EventSequence& EventSequence::operator=(const EventSequence &s){
	name=s.name;
	K=s.K;
	start_t=s.start_t;
	end_t=s.end_t;

	observed=s.observed;
	N=0;//initialize the number of observed events in the sequence
	type.resize(s.K);
	
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(s.K);
	
	//create deep copies of events, the events are indexed by their arrival times
	
	//copy observed events
	for(auto iter=s.full.begin();iter!=s.full.end();iter++){
		Event *e_copied;
		if(!iter->second) return *new EventSequence();
		if(iter->second->parent){//this is true for all events except the virtual event (which corresponds to the background process)
			//the parent of the new deep copy is indexed by the  arrival time and already exists in the structure since the events are tranversed by chronological order
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,full[iter->second->parent->time]};
			if(!e_copied) return *new EventSequence();
			addEvent(e_copied,full[iter->second->parent->time]);
		}
		else{
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,(Event *)0};//only for the virtual event
			addEvent(e_copied,0);
		}
	}
	//copy thinned events
	for(auto iter=s.thinned_full.begin();iter!=s.thinned_full.end();iter++){
		Event *e_copied;
		if(iter->second->parent){//this should be true for all the events
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,iter->second->observed, full[iter->second->parent->time]};//the parent is an observed event
			if(!e_copied) return *new EventSequence();
			addEvent(e_copied,full[iter->second->parent->time]);
		}
	}
	return *this;
}

//create empty event sequence with no types, and no events (observed or thinned)
EventSequence::EventSequence(){
	K=0;
	N=0;//initialize the number of observed events in the sequence
	Nt=0;//initialize the number of latent events in the sequence
}


//create empty event sequence with no types, and no events (observed or thinned)
EventSequence::EventSequence(std::string n):name{n}{
	K=0;
	N=0;//initialize the number of observed events in the sequence
	Nt=0;//initialize the number of latent events in the sequence
}

//TODO: is this called???do we need deep copies for that???
EventSequence::EventSequence(const EventSequence &s):name{s.name}, K{s.K}, start_t{s.start_t}, end_t{s.end_t}, N{s.N}, Nt{s.Nt}, observed{s.observed}{
	

	N=0;//initialize the number of observed events in the sequence
	type.resize(s.K);
	
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(s.K);
	
	//create deep copies of events, the events are indexed by their arrival times
	
	//copy observed events
	for(auto iter=s.full.begin();iter!=s.full.end();iter++){
		Event *e_copied;
		if(iter->second->parent){//this is true for all events except the virtual event (which corresponds to the background process)
			//the parent of the new deep copy is indexed by the  arrival time and already exists in the structure since the events are tranversed by chronological order
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,full[iter->second->parent->time]};
			addEvent(e_copied,full[iter->second->parent->time]);
		}
		else{
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,(Event *)0};//only for the virtual event
			addEvent(e_copied,0);
		}
	}
	//copy thinned events
	for(auto iter=s.thinned_full.begin();iter!=s.thinned_full.end();iter++){
		Event *e_copied;
		if(iter->second->parent){//this should be true for all the events
			e_copied=new Event{iter->second->id, iter->second->type, iter->second->time, iter->second->K,iter->second->observed, full[iter->second->parent->time]};//the parent is an observed event
			addEvent(e_copied,full[iter->second->parent->time]);
		}
	}
}


//we need move constructor for efficiency since objects of this class are returned by other functions
EventSequence::EventSequence(EventSequence &&s):name{s.name}, K{s.K}, start_t{s.start_t}, end_t{s.end_t}, N{s.N}, Nt{s.Nt}, type{s.type}, full{s.full}, thinned_type{s.thinned_type}, thinned_full{s.thinned_full}, observed{s.observed}{

	s.N=0;
	s.Nt=0;
	s.full.clear();
	s.thinned_full.clear();
	for(auto t_iter=s.type.begin();t_iter!=s.type.end();t_iter++){
		t_iter->clear();
	}
	s.type.clear();

	for(auto t_iter=s.thinned_type.begin();t_iter!=s.thinned_type.end();t_iter++){
		t_iter->clear();
	}
	s.type.clear();
	std::cout<<"sequence move constructor is called done\n";
}

EventSequence::EventSequence(std::string n, unsigned int k, double s, double e): name{n}, K{k}, start_t{s}, end_t{e}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
}

EventSequence::EventSequence(std::string n, unsigned int k): name{n}, K{k}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
};

//destructor, it preserves the pointers to the event objects, it releases the sequence's structures
EventSequence::~EventSequence(){

	if(!full.empty()){
		full.clear();
	}
	if(!type.empty()){
		for(unsigned int k=0;k<type.size();k++)
			if(!type[k].empty())
				type[k].clear();
		type.clear();
	}
	if(!thinned_full.empty()){
		thinned_full.clear();
	}
	if(!thinned_type.empty()){
		for(unsigned int k=0;k<thinned_type.size();k++)
			if(!thinned_type[k].empty())
				thinned_type[k].clear();
		thinned_type.clear();
	}
}

//TODO: flush instead of destructor since this function also releases the pointer of events and is not called implicitly
//when a function returns
void EventSequence::flush(){
	
	//release observed events
	if(!full.empty()){
		//ADVICE:TODO: avoid recursions such as delEvent which change full since this loop will crush in next iteration (iter+1 points nowhere then)
		for(auto iter=full.begin();iter!=full.end();iter++){
			if(iter->second){
				if(iter->second->offsprings) {
					//delete iter->second->offsprings;: TODO: why doesn't this work????and why should I use free instead???
					free(iter->second->offsprings);
				}
				free(iter->second);
			}
		}
		full.clear();
	}
	
	if(!type.empty()){
		for(unsigned int k=0;k!=type.size();k++){
			if(!type[k].empty()){
				type[k].clear();
			}
		}
		type.clear();
	}
	
	//release thinned events
	if(!thinned_full.empty()){
		//ADVICE:TODO: avoid recursions such as delEvent which change full since this loop will crush in next iteration (iter+1 points nowhere then)
		for(auto iter=thinned_full.begin();iter!=thinned_full.end();iter++){
			if(iter->second){
				free(iter->second);
			}
		}
		thinned_full.clear();
	}
	
	if(!thinned_type.empty()){
		for(unsigned int k=0;k!=thinned_type.size();k++){
			if(!thinned_type[k].empty()){
				thinned_type[k].clear();
			}
		}
		thinned_type.clear();
	}
}

void EventSequence::flushThinnedEvents(){
	
	if(!thinned_full.empty()){
		//ADVICE:TODO: avoid recursions such as delEvent which change full since this loop will crush in next iteration (iter+1 points nowhere then)
		for(auto iter=thinned_full.begin();iter!=thinned_full.end();){
			if(iter->second){
				iter=delEvent(iter->second);
			}
			else
				iter++;
		}
		thinned_full.clear();
	}
	
	if(!thinned_type.empty()){
		for(unsigned int k=0;k!=thinned_type.size();k++){
			if(!thinned_type[k].empty()){
				thinned_type[k].clear();
			}
		}
	}
	
}
//constructor which initializes the name, the number of types K, the observation window, and the observed events from a merged with all types of events sequence
EventSequence::EventSequence(std::string n, unsigned int k, double s, double e, std::map<double,Event *> seq): name{n}, K{k}, start_t{s}, end_t{e}, full{seq}{
	N=full.size();//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;
	thinned_type.resize(K);
	//separate the observed events by type
	for(auto e_iter=seq.begin();e_iter!=seq.end(); e_iter++){
		type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}
}
//constructor which initializes the name, the number of types K, the observation window, the observed events and the latent events from a merged with all types of events sequence
EventSequence::EventSequence(std::string n, unsigned int k, double s, double e, std::map<double,Event *> o_seq, std::map<double,Event *> t_seq): name{n}, K{k}, start_t{s}, end_t{e}, full{o_seq}, thinned_full{t_seq}{
	N=full.size();//initialize the number of observed events in the sequence
	type.resize(K);
	//separate the observed events by type
	for(auto e_iter=o_seq.begin();e_iter!=o_seq.end(); e_iter++){
		if(e_iter->second->type>=(int)K) return;
		type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}

	Nt=thinned_full.size();//initialize the number of latent events in the sequence
	thinned_type.resize(K);
	//separate the thinned events by type
	for(auto e_iter=t_seq.begin();e_iter!=t_seq.end(); e_iter++){
		if(e_iter->second->type>=(int)K) return;
		thinned_type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}
}
//constructor which initializes the name, the number of types K, the observation window, and the observed events when they are separated by type
EventSequence::EventSequence(std::string n, unsigned int k, double s, double e, std::vector<std::map<double,Event *>> tseqs): name{n}, K{k}, start_t{s}, end_t{e}, type{tseqs}{
	//merge the events of all types in one sequence
	for(auto t_iter=tseqs.begin();t_iter!=tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			full[e_iter->second->time]=e_iter->second;
	N=full.size();//initialize the number of observed events in the sequence
	Nt=0;
	thinned_type.resize(K);
}
//constructor which initializes the name, the number of types K, the observation window, and the observed and latent events when they are separated by type
EventSequence::EventSequence(std::string n, unsigned int k, double s, double e, std::vector<std::map<double,Event *>> o_tseqs, std::vector<std::map<double,Event *>> t_tseqs): name{n}, K{k}, start_t{s}, end_t{e}, type{o_tseqs}, thinned_type{t_tseqs}{
	for(auto t_iter=o_tseqs.begin();t_iter!=o_tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			full[e_iter->second->time]=e_iter->second;
	N=full.size();//initialize the number of observed events in the sequence
	
	for(auto t_iter=t_tseqs.begin();t_iter!=t_tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			thinned_full[e_iter->second->time]=e_iter->second;
	Nt=thinned_full.size();//initialize the number of latent events in the sequence
}

EventSequence::EventSequence(std::string n, unsigned int k, double s, double e, std::ifstream &file): name{n}, K{k}, start_t{s}, end_t{e}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
	load(file,1);
}

//constructor which initializes the number of types for the event sequence and the observation window [s,e]
EventSequence::EventSequence(unsigned int k, double s, double e): K{k}, start_t{s}, end_t{e}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
};
//constructor which initializes the number of types for the event sequence
EventSequence::EventSequence(unsigned int k): K{k}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
};

//constructor which initializes the number of types K, the observation window, and the observed events from a merged with all types of events sequence
EventSequence::EventSequence(unsigned int k, double s, double e, std::map<double,Event *> seq): K{k}, start_t{s}, end_t{e}, full{seq}{
	N=full.size();//initialize the number of observed events in the sequence
	//separate events by type
	type.resize(K);
	for(auto e_iter=seq.begin();e_iter!=seq.end(); e_iter++){
		if(e_iter->second->type>=(int)K) return;
		type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}
	Nt=0;
	thinned_type.resize(K);
	
}

//constructor which initializes the number of types K, the observation window, and the observed events from a merged with all types of events sequence
EventSequence::EventSequence(unsigned int k, double s, double e, std::map<double,Event *> o_seq, std::map<double,Event *> t_seq): K{k}, start_t{s}, end_t{e}, full{o_seq}, thinned_full{t_seq}{
	N=full.size();//initialize the number of observed events in the sequence
	type.resize(K);
	//separate observed events by type
	for(auto e_iter=o_seq.begin();e_iter!=o_seq.end(); e_iter++){
		type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}
	
	Nt=thinned_full.size();//initialize the number of latent events in the sequence
	//separate latent events by type
	thinned_type.resize(K);
	for(auto e_iter=t_seq.begin();e_iter!=t_seq.end(); e_iter++){
		if(e_iter->second->type>=(int)K) return;
		thinned_type[e_iter->second->type][e_iter->second->time]=e_iter->second;
	}
}
//constructor which initializes the number of types K, the observation window, and the observed and latent events when they are separated by type
EventSequence::EventSequence(unsigned int k, double s, double e, std::vector<std::map<double,Event *>> tseqs): K{k}, start_t{s}, end_t{e}, type{tseqs}{ 
	//merge the observed events of different types in one sequence
	for(auto t_iter=tseqs.begin();t_iter!=tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			full[e_iter->second->time]=e_iter->second;
	N=full.size();//initialize the number of observed events in the sequence
}


EventSequence::EventSequence(unsigned int k, double s, double e, std::vector<std::map<double,Event *>> o_tseqs, std::vector<std::map<double,Event *>> t_tseqs): K{k}, start_t{s}, end_t{e}, type{o_tseqs}, thinned_type{t_tseqs}{ 
	//merge the observed events of different types in one sequence
	for(auto t_iter=o_tseqs.begin();t_iter!=o_tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			full[e_iter->second->time]=e_iter->second;
	N=full.size();//initialize the number of observed events in the sequence
	//merge the latent events of different types in one sequence
	for(auto t_iter=t_tseqs.begin();t_iter!=t_tseqs.end();t_iter++)
		for(auto e_iter=t_iter->begin(); e_iter!=t_iter->end();e_iter++)
			thinned_full[e_iter->second->time]=e_iter->second;
	Nt=thinned_full.size();//initialize the number of latent events in the sequence
}

EventSequence::EventSequence(unsigned int k, double s, double e, std::ifstream &file): K{k}, start_t{s}, end_t{e}{
	N=0;//initialize the number of observed events in the sequence
	type.resize(K);
	Nt=0;//initialize the number of latent events in the sequence
	thinned_type.resize(K);
	load(file,1);
}

void EventSequence::print(std::ostream & file, unsigned int type,  bool observed) const{
	
	//print the observed events of the sequence ordered by arrival time
	switch(type){
		case 0:{
			file<<"Observed Events:\n";
			for(auto e_iter=full.begin();e_iter!=full.end();e_iter++)
				if(e_iter->second)
					e_iter->second->print(file,observed);
			//print the latent events of the sequence ordered by arrival time
			if(!observed){
				file<<"Thinned Events:\n";
				for(auto e_iter=thinned_full.begin();e_iter!=thinned_full.end();e_iter++)
					if(e_iter->second)
						e_iter->second->print(file,observed);
			}
		}
		break;
		case 1:{
	 	   //write headers
           file<<"arrival time, event type, event id, label, parent arrival time, nof offsprings, offsprings <type>,offsprings <id>"<<std::endl;
			for(auto e_iter=full.begin();e_iter!=full.end();e_iter++)
				if(e_iter->second)
					e_iter->second->print(file, type,observed);
			if(observed){
				std::cout<<"start printing the thinned events\n";
				for(auto e_iter=thinned_full.begin();e_iter!=thinned_full.end();e_iter++)
					if(e_iter->second)
						e_iter->second->print(file, type,observed);
			}

		}
		break;
	}
	
}

//plot the event sequence, save the figures in the current directory
void EventSequence::plot() const{
	//create output file for the gnuplot-separate per type the sequences
	boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
    std::string curr_path_str=curr_path_boost.string();
    plot_(curr_path_str);

}

//plot the event sequence, save the figures in the directory dir_path_str
void EventSequence::plot(const std::string & dir_path_str) const{
	plot_(dir_path_str);
}

void EventSequence::plot_(const std::string & dir_path_str) const{
	std::vector<std::string> plot_specs;
	std::string file_path_str;
	
	 //create output file for the gnuplot-separate per type the sequences of the observed events
    if(!name.empty())
    	file_path_str=dir_path_str+"/merged_event_sequence_"+name+".png";
    else
    	file_path_str=dir_path_str+"/merged_event_sequence.png";

    
    if(N+Nt==0) return; //the sequence is empty.
    
    //create plot specifications
//	for(unsigned int k=0;k<type.size();k++){
//		if(!type[k].empty())//plot the type if there is at least one event of this type in the sequence
//			plot_specs.push_back("with points pt 13 ps 0.3 title 'Type "+std::to_string(k)+"'");
//	}
    std::vector<std::string> colors={"'#0000FF'", "'#FF0000'", "'#6B8E23'","'#7FFF00'", "'#8A2BE2'", "'#008080'", "'#000000'", "'#A52A2A'","'#FFA500'", "'#00FFFF'",
    								 "'#B8860B'", "'#FF8C00'", "'#FFCC00'","'#008B8B'", "'#F03232'", "'#EE82EE'", "'#C8C800'", "'#00FFCC'","'#228B22'", "'#7FFFD4'",
									 "'#003333'", "'#666699'", "'#00FF99'","'#FFD700'", "'#3399FF'", "'#008080'", "'#000000'", "'#A52A2A'","'#FFA500'", "'#00FFFF'"
    };
    		                        // "'#FF0000'", "'#B8860B'", "'#FF8C00'","'#FFFFFF'", "'#008B8B'", "'#A52A2A'", "'#EE82EE'", "'#C8C800'","#'F03232'", "'#228B22'",
    								// "'#A52A2A'", "'#C8C800'", "'#00FF00'","'#7FFFD4'", "'#B8860B'" };
	for(unsigned int k=0;k<type.size();k++){
		if(!type[k].empty())//plot the type if there is at least one event of this type in the sequence
			plot_specs.push_back("with points pt 13 ps 0.8 lc rgb "+colors[k]+" notitle");
	}

	Gnuplot gp;
	create_gnuplot_script(gp,file_path_str,plot_specs,start_t,end_t,0,(K+1),"Time (msec)","Neuron");
	//start creating the plot points
    for(unsigned int k=0;k<type.size();k++){
    	std::vector<double> pts_x;
    	std::vector<double> pts_y;
    	for(auto e_iter=type[k].begin();e_iter!=type[k].end();e_iter++){
    		pts_x.push_back(e_iter->second->time);
    		pts_y.push_back((k+1));
    	}
      	if(!pts_x.empty())
      		gp.send1d(boost::make_tuple(pts_x, pts_y));
    }

    //create output file for the gnuplot-separate per cluster the sequences of observed events. 
    Gnuplot gp2;
    file_path_str.clear();
    plot_specs.clear();
    if(!name.empty())
    	file_path_str=dir_path_str+"/cluster_event_sequence_"+name+".png";
    else
    	file_path_str=dir_path_str+"/cluster_event_sequence.png";
 
    //get the number of clusters. there is one cluster per event of the background process. 
    //the cluster can contain different types of events but they are all trigerred by an offspring of an event in the background process.
	unsigned int C=countClusters();
	if(C) {
		
		//create plot specifications
		for(unsigned int c=0;c<C+1;c++){
			plot_specs.push_back("with points pt 13 ps 0.3 notitle");
		}
		
		create_gnuplot_script(gp2,file_path_str,plot_specs,start_t,end_t,0,(C+1)*2.0,"Time","Cluster");

		//formulate and plot the clusters generated by each event in the background process
		unsigned int c=0;//counter for the number of clusters
		//traverse the offsprings of the virtual event (which is the single event of a virtual, last, type), which are all the events in the background process
		Event *e=type[K-1].begin()->second;
		EventSequence cluster(K, start_t, end_t);
		returnCluster(e,cluster);
		std::vector<double> pts_x;
		std::vector<double> pts_y;
		for(auto e_iter_2=cluster.full.begin();e_iter_2!=cluster.full.end();e_iter_2++){
			pts_x.push_back(e_iter_2->second->time);
			pts_y.push_back((c+1)*2.0);
		}
		//plot the points
		if(!pts_x.empty())
			gp2.send1d(boost::make_tuple(pts_x, pts_y));
		c++;

		for(auto e_iter=type[K-1].begin()->second->offsprings->full.begin();e_iter!=type[K-1].begin()->second->offsprings->full.end();e_iter++){
			//extract the cluster generated by the immigrant event
			EventSequence cluster(K, start_t, end_t);
			returnCluster(e_iter->second,cluster);
			//get the arrival time of each event
			std::vector<double> pts_x;
			std::vector<double> pts_y;
			for(auto e_iter_2=cluster.full.begin();e_iter_2!=cluster.full.end();e_iter_2++){
				pts_x.push_back(e_iter_2->second->time);
				pts_y.push_back((c+1)*2.0);
			}
			//plot the points
			if(!pts_x.empty())
				gp2.send1d(boost::make_tuple(pts_x, pts_y));
			c++;
		}
	}
	
	
	// check if there are thinned events 
	if(thinned_full.empty()) return;
	
	//start plotting the thinned events 
	
  
    file_path_str.clear();
    plot_specs.clear();
    //create the plot specifications
	for(unsigned int k=0;k<thinned_type.size();k++){
		if(!thinned_type[k].empty()){//there are thinned events of this type
			plot_specs.push_back("with points pt 13 ps 0.3 title 'Type "+std::to_string(k)+"'");
		}
	}
	if(plot_specs.size()>0){
		  //create output file for the gnuplot-separate per type the sequences of thinned events
	    Gnuplot gp3;
	    if(!name.empty())
	    	file_path_str=dir_path_str+"/thinned_merged_event_sequence_"+name+".png";
	    else
	    	file_path_str=dir_path_str+"/thinned_merged_event_sequence.png";
	    
	    create_gnuplot_script(gp3,file_path_str,plot_specs,start_t,end_t,0,(K+1));

	    //start creating the points
	    for(unsigned int k=0;k<thinned_type.size();k++){
	    	std::vector<double> pts_x;
	    	std::vector<double> pts_y;
	    	for(auto e_iter=thinned_type[k].begin();e_iter!=thinned_type[k].end();e_iter++){
	    		pts_x.push_back(e_iter->second->time);
	    		//pts_y.push_back((k+1)*2.0);
	    		pts_y.push_back((k+1));
	    	}
	    	if(!pts_x.empty())
	    		gp3.send1d(boost::make_tuple(pts_x, pts_y));
	    }
	}
    
    //create output file for the gnuplot-separate per cluster the thinned sequences 
    file_path_str.clear();
    plot_specs.clear();
	std::vector<std::vector<double>> pts_x;
	std::vector<std::vector<double>> pts_y;
	unsigned int c=0; //number of clusters with at least one thinned event
	std::vector<double> c_pts_x;
	std::vector<double> c_pts_y;
	
	//form a cluster with the thinned events which belong to the background process
	for(auto e_iter=type[K-1].begin()->second->offsprings->thinned_full.begin();e_iter!=type[K-1].begin()->second->offsprings->thinned_full.end();e_iter++){
		c_pts_x.push_back(e_iter->second->time);
		c_pts_y.push_back((c+1)*2.0);
	}
	if(c_pts_x.size()){
		c++;
		pts_x.push_back(c_pts_x);
		pts_y.push_back(c_pts_y);
	}
	//form clusters with thinned events trigerred bby events in the background process
	for(auto e_iter=type[K-1].begin()->second->offsprings->full.begin();e_iter!=type[K-1].begin()->second->offsprings->full.end();e_iter++){
		std::vector<double> c_pts_x;
		std::vector<double> c_pts_y;
		//extract the cluster generated by the immigrant event
		EventSequence cluster(K, start_t, end_t);//TODOL: merge it with the previous loop for the formulation of the clusters
		returnCluster(e_iter->second,cluster);
		//get the arrival time of each event
	
		for(auto e_iter_2=cluster.thinned_full.begin();e_iter_2!=cluster.thinned_full.end();e_iter_2++){
			c_pts_x.push_back(e_iter_2->second->time);
			c_pts_y.push_back((c+1)*2.0);
		}
		//plot the points
		if(!c_pts_x.empty()){
			pts_x.push_back(c_pts_x);
			pts_y.push_back(c_pts_y);
			c++;
		}
		
	}

	if(c>0){
	    Gnuplot gp4;
	    if(!name.empty()){
	    	file_path_str=dir_path_str+"/thinned_cluster_event_sequence_"+name+".png";
	    }
	    else{
	    	file_path_str=dir_path_str+"/thinned_cluster_event_sequence.png";
	    }
		for(unsigned int i=0;i<c;i++){
			plot_specs.push_back("with points pt 13 ps 0.3 notitle");
		}
		create_gnuplot_script(gp4,file_path_str,plot_specs,start_t,end_t,0,(c+1)*2.0,"Time","Cluster");
	
		for(unsigned int i=0;i<c;i++)
			gp4.send1d(boost::make_tuple(pts_x[i], pts_y[i]));
	}
}

void EventSequence::serialize(boost::archive::text_oarchive &ar, const unsigned int version){
	ar & name; //serialize the name of the sequence
	ar & K; //serialize the number of event types of the sequence
	ar & start_t; //serialize the beginning of the observation window of the sequence
	ar & end_t; //serialize the end of the observation window of the sequence
	ar & N; //serialize the number of observed events in the sequence
	ar & Nt; //serialize the number of thinned events in the sequence
	ar & type; //serialize the per-type event sequences of osberved events
	ar & full; //serialize the merged event sequence of observed events
	ar & thinned_type; //serialize the per-type event sequences of thinned events
	ar & thinned_full;//serialize the merged event sequence of thinned events
	ar & observed;
	
};

void EventSequence::serialize(boost::archive::text_iarchive &ar, const unsigned int version){
	
	ar & name; //serialize the name of the sequence
	ar & K; //serialize the number of event types of the sequence
	ar & start_t; //serialize the beginning of the observation window of the sequence
	ar & end_t; //serialize the end of the observation window of the sequence
	ar & N; //serialize the number of observed events in the sequence
	ar & Nt; //serialize the number of thinned events in the sequence
	ar & type; //serialize the per-type event sequences of osberved events
	ar & full; //serialize the merged event sequence of observed events
	ar & thinned_type; //serialize the per-type event sequences of thinned events
	ar & thinned_full;//serialize the merged event sequence of thinned events
	ar & observed;
	
};

void EventSequence::serialize(boost::archive::binary_oarchive &ar, const unsigned int version){
	
	ar & name; //serialize the name of the sequence
	ar & K; //serialize the number of event types of the sequence
	ar & start_t; //serialize the beginning of the observation window of the sequence
	ar & end_t; //serialize the end of the observation window of the sequence
	ar & N; //serialize the number of observed events in the sequence
	ar & Nt; //serialize the number of thinned events in the sequence
	ar & type; //serialize the per-type event sequences of osberved events
	ar & full; //serialize the merged event sequence of observed events
	ar & thinned_type; //serialize the per-type event sequences of thinned events
	ar & thinned_full;//serialize the merged event sequence of thinned events
	ar & observed;
};

void EventSequence::serialize(boost::archive::binary_iarchive &ar, const unsigned int version){
	
	ar & name; //serialize the name of the sequence
	ar & K; //serialize the number of event types of the sequence
	ar & start_t; //serialize the beginning of the observation window of the sequence
	ar & end_t; //serialize the end of the observation window of the sequence
	ar & N; //serialize the number of observed events in the sequence
	ar & Nt; //serialize the number of thinned events in the sequence
	ar & type; //serialize the per-type event sequences of osberved events
	ar & full; //serialize the merged event sequence of observed events
	ar & thinned_type; //serialize the per-type event sequences of thinned events
	ar & thinned_full;//serialize the merged event sequence of thinned events
	ar & observed;
	
};

void EventSequence::save(std::ofstream &file) const{

	if(!file){
		boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
	    std::string curr_path_str=curr_path_boost.string();
	    std::string file_path_str=curr_path_str+"/event_sequence_"+name;
	    file.open((file_path_str).c_str(),std::ios::binary | std::ios::trunc);
	    if(!file)
	    	std::cerr<<"Couldn't create the output file."<<std::endl;
	}
	//serialize the save the object
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void EventSequence::load(std::ifstream &file, unsigned int file_type){
	if(!file)
		std::cerr<<"Couldn't open the input file."<<std::endl;
	switch(file_type){
		case 0:{
			//deserialize and load object
			boost::archive::binary_iarchive ia{file};
			ia >> *this;
			}
		break;
		case 1:{
	
			//parse out the headers line

			std::string headers_line;
			getline(file, headers_line);

		
			std::vector<std::string> headers_strs;
			boost::algorithm::split(headers_strs, headers_line, boost::is_any_of(","));
			for(auto iter=headers_strs.begin();iter!=headers_strs.end();){
	
				iter->erase(std::remove_if(iter->begin(), iter->end(), ::isspace), iter->end());
				
				if(iter->empty()){
					iter=headers_strs.erase(iter);
				}
				else
					iter++;
			
			}	

			unsigned int nof_headers=headers_strs.size();
			if(nof_headers<2)
				std::cerr<<"wrong csv format\n";
			//start reading the events
			std::string line;

			while (getline(file, line))
			{
				if(line.empty()) break;

				//read the next event line
				std::vector<std::string> event_strs;
				boost::algorithm::split(event_strs, line, boost::is_any_of(","));
				line.clear();

				std::string::size_type sz;
				if(event_strs.size()<2)
					std::cerr<<"wrong csv format\n";
				double arrival_time=std::stod (event_strs[0],&sz);
				//std::cout<<"arrival time "<<arrival_time<<std::endl;
				int event_type=std::stoi (event_strs[1],&sz);
				//std::cout<<"event type "<<event_type<<std::endl;
				if(nof_headers==2){
					observed=false;
					//only the arrival time and the event type are given in the csv
					Event *e=new Event{type[event_type].size(),event_type,arrival_time,K};
					addEvent(e);
				}
				else if(nof_headers==3){//only the arrival time, the event type, and the event id is given in the file
					observed=false;
					if(event_strs.size()<3)
						std::cerr<<"wrong csv format\n";
					unsigned long int event_id=std::stoul (event_strs[2],&sz);
					Event *e=new Event{event_id,event_type, arrival_time,K};
					addEvent(e);
				}
				else{//the arrival time, the event type, the event id, nof and ids of offspirng events are given in the file
					//observed=true; TODO: change it
					observed=false;
					unsigned long int event_id=std::stoul (event_strs[2],&sz);
					bool event_observed_flag=std::stoi (event_strs[3],&sz);
					double event_parent_arrival_time=std::stod(event_strs[4],&sz);
					auto parent_iter=full.find(event_parent_arrival_time);
					Event *event_parent=0;
					if(parent_iter!=full.end())
						event_parent=parent_iter->second;
					Event *e=new Event{event_id,event_type,arrival_time, K, event_observed_flag, event_parent};
					addEvent(e,event_parent);

					//todo: for now the list of offsprings is ignored, normally i should check whether the parent and the offspring information is consistent
				}
				line.clear();
			}
		}
		break;
	}
}

std::map<double, Event*>::iterator EventSequence::pruneCluster(Event *e){
	if(!e) return full.end();
	//remove the subclusters generated by its offsprings, until the event becomes a leaf
	while(e->offsprings && e->offsprings->N){
		auto o_iter=e->offsprings->full.begin();//get the next observed object which has not been deleted yet
		pruneCluster(o_iter->second);
	}
	
	//remove the event, it also deletes the thinned events generated by e
	 return(delEvent(e));
}


std::map<double, Event*>::iterator EventSequence::delEvent(Event *e){
	if(!e) return full.end();
	
	//delete the event from it's parents offsprings
	if(e->parent){
		e->parent->removeChild(e);
	}
	
	//lambda expression which finds an event by its arrival time
	auto find_event_lambda=[=](const std::pair<double,Event *> & e_entry)->bool{return e_entry.second==e;};
	
	//make the observed offsprings of the event to be immigrant events, this condition is true only for observed events
	if(e->offsprings){
		for(auto o_iter=e->offsprings->full.begin();o_iter!=e->offsprings->full.end();o_iter++){
			o_iter->second->parent=full.begin()->second;//todo: map them to the virtual event
		}
		//remove the thinned events
		for(auto o_iter=e->offsprings->thinned_full.begin();o_iter!=e->offsprings->thinned_full.end();o_iter++){
			Nt--;//decrease by one the number of thinned events
			//update the structures for the thinned events of the sequence
		    o_iter = thinned_full.erase(o_iter);
		    if(o_iter->second->type>=(int)thinned_type.size()) return full.end();
			erase_if_map(thinned_type[o_iter->second->type], find_event_lambda);
			//release resources for the current thinned event
			free(o_iter->second);
		}
		
		//flush the structure of event sequence for the offsprings of the event
		e->offsprings->~EventSequence();//todo: write a destructor which doesn't delete the event pointers (this is flush!!!)
		free(e->offsprings);//todo: isn't it the same with the above line??? does it do something additional??
	}
	
	std::map<double, Event*>::iterator next_e_iter;
	if(e->observed){
		//update the structures for the observed events of the sequence
		 next_e_iter=erase_if_map(full,find_event_lambda);
		 if(e->type>=(int)type.size()) return full.end();
		erase_if_map(type[e->type], find_event_lambda);
		N--;
	}else{
		//update the structures for the observed events of the sequence
		next_e_iter=erase_if_map(thinned_full,find_event_lambda);
		if(e->type>=(int)thinned_type.size()) return thinned_full.end();
		erase_if_map(thinned_type[e->type], find_event_lambda);
		Nt--;
	}
	
	//delete the event
	free(e); //delete e calls the destructor
	return next_e_iter;//return the iterator for the event which is next in the merged sequence
}

void EventSequence::returnCluster(Event *e, EventSequence &s) const{

	if(!e) return;
	s.addEvent(e,e->parent);//add the parent event in the sequence of the cluster 
	if(!e->offsprings) return;
	
	//add the thinned events, the thinned events do not have offsprings therefore there is no need for recursion
	for(auto e_iter=e->offsprings->thinned_full.begin();e_iter!=e->offsprings->thinned_full.end();e_iter++){
		s.addEvent(e_iter->second, e);
	}
	
	//start traversing by dfs the tree of the offsprings of offsprings to continue forming the cluster
	for(auto e_iter=e->offsprings->full.begin();e_iter!=e->offsprings->full.end();e_iter++){
		returnCluster(e_iter->second,s);
	}
}

void EventSequence::changeParent(Event *e, Event *p){
	if(!e||!p) return;
	//remove the event from the offsprings of its old parent
	Event *old_p=e->parent;
	if(old_p && old_p!=p){//if the event has a parent (it is not the virtual event) and it is different from the new parent to be assigned
		//remove event from the offsprings of its old parent
		old_p->removeChild(e);
	}
	e->parent=p;
	if(p)
		p->addChild(e);
}

void EventSequence::setParent(Event *e, Event *p){
	if(!e) return;
	e->parent=p;
	if(p) 	//add the event to the new parent's offsprings
		p->addChild(e);
}

Event * EventSequence::findEvent(double t) const{
	for(auto e_iter=full.begin();e_iter!=full.end();e_iter++)
		if(e_iter->second->time==t)
			return e_iter->second;
	
	return 0;
}

void EventSequence::addEvent(Event *e, Event *p){
	
	if(!e) return;

	//update the sequence structures
	if(e->observed){
		if ( full.find(e->time) != full.end() ) 
			std::cout<<"same arrival time "<<e->time<<" exists\n";
		else{
			full[e->time]=e;
			if(e->type>=(int)type.size()) return;
				type[e->type][e->time]=e;
			N++;
		}
	}
	else{
		if (thinned_full.find(e->time) == thinned_full.end() ) {
			thinned_full[e->time]=e;
			if(e->type>=(int)thinned_type.size()){ 
				std::cerr<<"invalid type"<<std::endl;
				std::cerr<<e->type<<std::endl;
				std::cerr<<thinned_type.size()<<std::endl;
				return;
			}
			thinned_type[e->type][e->time]=e;
			Nt++;
		} else {
		  std::cout<<"same arrival time "<<e->time<<" exists\n";
		}
	}
	//connect the event with its parent
	e->parent=p;
	if(p){//only the virtual event does not have parent
		p->addChild(e);
	}
}

void EventSequence::summarize(std::ofstream & file){
	
	if(!file){
		boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
	    std::string curr_path_str=curr_path_boost.string();
	    std::string file_path_str=curr_path_str+"/summary_"+name+".txt";
	    file.open((file_path_str).c_str());
	    if(!file)
	    	std::cerr<<"Couldn't create the output file."<<std::endl;
	}

    /*print the number of event types*/
     file<<"Number of event types: "<<K<<std::endl;


	/*print the number events per type in the sequence and how many events of a type are generated by events of this type.*/
	for(unsigned int k=0;k!=type.size();k++){
		file<<"Number of events for type "<<k<<": "<<type[k].size()<<std::endl;
		for(unsigned int k2=0;k2!=K;k2++){
			file<<"Type  "<<k<<": generates "<<countTriggeredEvents(k,k2)<<" events of type "<<k2<<"."<<std::endl;
		}
	}
	
	/*print the mean temporal distance between different pairs of event types*/
	for(unsigned int k=0;k<type.size();k++){
		
		std::vector <double> temporal_distance(K,0.0);
		//how many pairs (type_1, type_2) with type_1.arrival_time<type_2.arrival_time in the sequence
		std::vector <int> type_pair(K,0);

		//the position of the potential closest next event for each type. the events before this position correspond to surely earlier events of this type
		std::vector<std::map<double, Event*>::iterator> next_arrival(K);
		for(unsigned int k2=0;k2<K;k2++){
			next_arrival[k2]=type[k2].begin();
		}

		//for each event of the current type sequence
		for(auto e_iter=type[k].begin(); e_iter!=type[k].end();e_iter++){
		
			for(unsigned int k2=0;k2<K;k2++){
				//find closest next event of type k2
				std::map<double, Event*>::iterator next_e_iter;
				for(next_e_iter=next_arrival[k2];;next_e_iter++){
					if(next_e_iter==type[k2].end()) {
						break;
					}
					if(!(next_e_iter!=type[k2].end()
							&& e_iter->second->time
							>=next_e_iter->second->time)) break;
				}
				//keep this position for next pairs of events of these types
				next_arrival[k2]=next_e_iter;

     			//if next event of type k exists and no other event of the same type exists between them
				auto e_iter_2=e_iter;
				e_iter_2++;
				if(next_arrival[k2]!=type[k2].end() && (e_iter_2==type[k].end()|| e_iter_2->second->time>=next_e_iter->second->time)){
					type_pair[k]++;
					temporal_distance[k]+=next_e_iter->second->time-e_iter->second->time;
				}
			}
		}

		//print the mean temporal distances
		for(unsigned int k2=0; k2<K; k2++){
			file<<"mean temporal distance from events of type "<<k<<" to type "<<k2<<" is "<<temporal_distance[k2]/type_pair[k2]<<std::endl;
		}
	}
}

//it counts the number of observed events of type k2 trigerred by event of type k
inline unsigned int EventSequence::countTriggeredEvents(int k, int k2) const{
	if(type.empty()||(int)type.size()<k+1||type[k].empty()) return 0;
	
	unsigned int N2=0;
	for(auto e_iter=type[k].begin();e_iter!=type[k].end();e_iter++){//for each event of type k
		if(e_iter->second->offsprings){
			N2+=e_iter->second->offsprings->type[k2].size();//add by the number of realized offsprings of type k2
			N2+=e_iter->second->offsprings->thinned_type[k2].size();//add by the number of realized offsprings of type k2
		}
	}
	return N2;
}

//it counts the number of clusters; the number of events in the background process
inline unsigned int EventSequence::countClusters() const{
	if(type[type.size()-1].begin()->second->offsprings)
		return type[type.size()-1].begin()->second->offsprings->full.size();
	return 0;
}

//it puts in a vector the arrival times of the observed events of type k
void EventSequence::getArrivalTimes(unsigned int k, boost::numeric::ublas::vector<double> & time) const{

	if(type.size()<k+1) return;
	unsigned int nof_events=type[k].size();
	if(time.size()<nof_events)
		time.resize(nof_events);
	unsigned int i=0;
	for(auto e_iter=type[k].begin();e_iter!=type[k].end();e_iter++){
		if(!e_iter->second) return;
		time(i)=e_iter->second->time;
		i++;
	}
}


Event::Event()=default;

Event& Event::operator=(const Event &e){
	id=e.id;//assign the event id
	type=e.type;//assign the event type
	time=e.time;//assign the arrival time
	K=e.K;//assign the number of types of events that it can generate
	observed=e.observed;//assign the label which indicates whether the event is observed or thinned
	parent=e.parent;//assign the parent of the event;the event which trigerred event e
	offsprings=e.offsprings;//assign the offsprings of the event 
	return *this;
}

Event::Event(const Event &e):id{e.id}, type{e.type}, time{e.time}, K{e.K}, observed{e.observed}, parent{e.parent}, offsprings{e.offsprings} {};

Event::Event(unsigned long i, int l, double t, unsigned int k) : id{i}, type{l}, time {t}, K{k}{};

Event::Event(unsigned long i, int l, double t, unsigned int k, Event *p) : id{i}, type{l}, time {t}, K{k}, parent{p}{} 

Event::Event(unsigned long i, int l, double t, unsigned int k, bool o, Event *p) : id{i}, type{l}, time {t}, K{k}, observed{o}, parent{p}{} 
//comment: we don't want a deep copy of this pointer in the constructor since it refers to an event in the event sequence

Event::Event(unsigned long i, int l, double t, unsigned int k, EventSequence *s) : id{i}, type{l}, time {t}, K{k}, offsprings{s}{}

Event::Event(unsigned long i, int l, double t, unsigned int k, Event *p, EventSequence *s) : id{i}, type{l}, time {t}, K{k}, parent{p}, offsprings{s}{}

Event::Event(unsigned long i, int l, double t, unsigned int k, bool o, Event *p, EventSequence *s) : id{i}, type{l}, time {t}, K{k}, observed{o}, parent{p}, offsprings{s}{}

Event::Event(int l, double t, unsigned int k) : type{l}, time {t}, K{k}{}

Event::Event(int l, double t, unsigned int k, bool o, Event *p) : type{l}, time {t}, K{k}, observed{o}, parent{p}{}

Event::Event(int l, double t, unsigned int k, EventSequence *o) : type{l}, time {t}, K{k}, offsprings{o}{}

Event::Event(int l, double t, unsigned int k, Event *p, EventSequence *s) : type{l}, time {t}, K{k}, parent{p}, offsprings{s}{}

Event::Event(int l, double t, unsigned int k, bool o, Event *p, EventSequence *s) : type{l}, time {t}, K{k}, observed{o}, parent{p}, offsprings{s}{}

Event::~Event(){
	auto find_event_lambda=[=](const std::pair<double,Event *> & e_entry)->bool{return e_entry.second==this;};
	
	//delete the event from it's parents offsprings
	if(parent){
		if(!parent->offsprings) return;
		erase_if_map(parent->offsprings->full,find_event_lambda);
		if((int)parent->offsprings->type.size()<type+1) return;
		erase_if_map(parent->offsprings->type[type],find_event_lambda);
	}
	
	if(offsprings){
		for(auto o_iter=offsprings->full.begin();o_iter!=offsprings->full.end();o_iter++){
			if(o_iter->second)
				o_iter->second->parent=parent;//TODO:: is this correct??? they are assigned to their grandparent!!!
			
			//release thinned events. TODO: should I do something else????
			for(auto o_iter=offsprings->thinned_full.begin();o_iter!=offsprings->thinned_full.end();o_iter++){
				free(o_iter->second);
			}
		}
	}
}

void Event::print(std::ostream &file, unsigned int file_type, bool observed2) const{
	 switch(file_type) {
	 	 case 0:{//write a text file
			file<<"Event id: "<<std::to_string(id)<<" type: "<<std::to_string(type)<<" arrival time: "<<std::to_string(time)<<" observed: "<<observed<<std::endl;
			file<<"address of parent "<<parent<<std::endl;
			if(parent) {
				file<<parent->id<<std::endl;
				file<<parent->type<<std::endl;
			}
			else{
				file<<"parent: 0"<<std::endl;
			}
			file<<"offsprings:";
			if(offsprings){
				file<<"\nObserved Offsprings\n";
				for(auto e_iter=offsprings->full.begin();e_iter!=offsprings->full.end();e_iter++){
					if(e_iter->second)
						file<<"<"<<e_iter->second->type<<","<<e_iter->second->id<<"> ";
				}
				if(observed){
					file<<"\nThinned Offsprings\n";
					for(auto e_iter=offsprings->thinned_full.begin();e_iter!=offsprings->thinned_full.end();e_iter++){
						if(e_iter->second)
							file<<"<"<<e_iter->second->type<<","<<e_iter->second->id<<"> ";
					}
				}
			}
			file<<'\n';

			if (file.bad())
				std::cerr<<"Unable to write in file."<<std::endl;
		}
	 	 break;
	 	 case 1:{//write a csv file, a line per event
	 		file<<std::to_string(time)<<","<<std::to_string(type)<<","<<std::to_string(id)<<","<<observed;
	 		if(parent){
	 			file<<","<<std::to_string(parent->time);
	 		}
	 		else{
	 			file<<","<<-1.0;
	 		}
	 		if(offsprings){
	 			if(!observed2)
	 				file<<","<<offsprings->full.size();
	 			else
	 				file<<","<<offsprings->full.size()+offsprings->thinned_full.size();
				for(auto e_iter=offsprings->full.begin();e_iter!=offsprings->full.end();e_iter++){
					if(e_iter->second)
						file<<","<<std::to_string(e_iter->second->type)<<","<<std::to_string(e_iter->second->id);
				}
				if(!observed2)
					for(auto e_iter=offsprings->thinned_full.begin();e_iter!=offsprings->thinned_full.end();e_iter++){
						if(e_iter->second)
							file<<","<<std::to_string(e_iter->second->type)<<","<<std::to_string(e_iter->second->id);
					}
					
	 		}
	 		else{
	 			 file<<","<<0;
	 		}
			file<<"\n";
	 	 }
	 	 break;
	 }
}


void Event::serialize(boost::archive::text_oarchive &ar, const unsigned int version){
	ar & id; //serialize the evnt id
	ar & type; //serialize the event type
	ar & time; //serialize the event arrival time
	ar & K; //serialzie the number of types of offsprings of the event
	ar & observed; //serialize the label of the event
	ar & parent; //serialize the parent of the event
	ar & offsprings; //serialize the offsprings of the event
	
};


void Event::serialize(boost::archive::text_iarchive &ar, const unsigned int version){
	ar & id; //serialize the evnt id
	ar & type; //serialize the event type
	ar & time; //serialize the event arrival time
	ar & K; //serialzie the number of types of offsprings of the event
	ar & observed; //serialize the label of the event
	ar & parent; //serialize the parent of the event
	ar & offsprings; //serialize the offsprings of the event
	
};

void Event::serialize(boost::archive::binary_oarchive &ar, const unsigned int version){	
	ar & id; //serialize the evnt id
	ar & type; //serialize the event type
	ar & time; //serialize the event arrival time
	ar & K; //serialzie the number of types of offsprings of the event
	ar & observed; //serialize the label of the event
	ar & parent; //serialize the parent of the event
	ar & offsprings; //serialize the offsprings of the event
};


void Event::serialize(boost::archive::binary_iarchive &ar, const unsigned int version){
	ar & id; //serialize the evnt id
	ar & type; //serialize the event type
	ar & time; //serialize the event arrival time
	ar & K; //serialzie the number of types of offsprings of the event
	ar & observed; //serialize the label of the event
	ar & parent; //serialize the parent of the event
	ar & offsprings; //serialize the offsprings of the event
};

void Event::save(std::ofstream &file) const{
	
	if(!file){
		boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
	    std::string curr_path_str=curr_path_boost.string();
	    std::string file_path_str=curr_path_str+"/event_"+std::to_string(id);
	    file.open((file_path_str).c_str(),std::ios::binary | std::ios::trunc);
	    if(!file)
	    	std::cerr<<"Couldn't create the output file."<<std::endl;
	}
	
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void Event::load(std::ifstream &file){
	if(!file)
		std::cerr<<"Couldn't open the input file."<<std::endl;
	
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void Event::removeChild(Event *e){
	
	//find event with a spefic pointer in the map structure
	auto find_event_lambda=[=](const std::pair<double,Event *> & e_entry)->bool{return e_entry.second==e;};
		
	if(e->observed){
		//delete the event from its old parent's observed offsprings
		if(!offsprings) return;
		erase_if_map(offsprings->full,find_event_lambda);
		if((int)offsprings->type.size()<e->type+1) return;
		erase_if_map(offsprings->type[e->type],find_event_lambda);
		offsprings->N--;//update the number of observed offsprings
		if(!(offsprings->N+offsprings->Nt)){//if there are no more offsprings (thinned or observed)
			offsprings->~EventSequence();//release the eventsequence of the offsprings
			offsprings=0;
		}
	}
	else{	//delete the event from its old parent's thinned offsprings
		if(!offsprings) return;
		erase_if_map(offsprings->thinned_full,find_event_lambda);
		if((int)offsprings->thinned_type.size()<e->type+1) return;
		erase_if_map(offsprings->thinned_type[e->type],find_event_lambda);
		offsprings->Nt--;//update the number of thinned offsprings
		if(!(offsprings->N+offsprings->Nt)){
			offsprings->~EventSequence();
			offsprings=0;
		}
	}
}

void Event::addChild(Event *e){
	
	if(!offsprings){//create an event sequence of offsprings if it is the first offspring
		offsprings=new EventSequence{e->K};
	}
	
	if(e->observed){
		//put the event in the observed offsprings
		offsprings->full[e->time]=e;
		if((int)offsprings->type.size()<e->type+1) return;
		offsprings->type[e->type][e->time]=e;
		offsprings->N++;
	}
	else{
		//put the event in the thinned offsprings
		offsprings->thinned_full[e->time]=e;
		if((int)offsprings->thinned_type.size()<e->type+1) return;
		offsprings->thinned_type[e->type][e->time]=e;
		offsprings->Nt++;
	}
}
