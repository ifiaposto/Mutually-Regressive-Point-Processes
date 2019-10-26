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
#include <boost/filesystem.hpp>

#include "PointProcess.hpp"
#include "Kernels.hpp"
#include "stat_utils.hpp"
#include "plot_utils.hpp"
#include "struct_utils.hpp"
#include "debug.hpp"
#include "HawkesProcess.hpp"


BOOST_CLASS_EXPORT(HawkesProcess)


/****************************************************************************************************************************************************************************
 *
 * Tranditional Hawkes Process (only mutual excitation)
 *
******************************************************************************************************************************************************************************/


/****************************************************************************************************************************************************************************
 * Model Construction and Destruction Methods
******************************************************************************************************************************************************************************/

/***************************************************   anononymous point processes    *********************************************/

HawkesProcess::HawkesProcess()=default;

HawkesProcess::HawkesProcess(unsigned int k, double ts, double te): PointProcess(k,ts,te){}

HawkesProcess::HawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p): PointProcess(k,ts,te){
	
	for(auto iter=m.begin(); iter!=m.end();iter++){
		mu.push_back((*iter)->clone());
	}
	
	for(auto iter=p.begin(); iter!=p.end();iter++){
		std::vector<Kernel*> phi_k;
		for(auto iter2=iter->begin();iter2!=iter->end();iter2++){
			phi_k.push_back((*iter2)->clone());
		}
		phi.push_back(phi_k);
	}
}

HawkesProcess::HawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m): PointProcess(k,ts,te){
	for(auto iter=m.begin(); iter!=m.end();iter++){
		mu.push_back((*iter)->clone());
	}
}

/***************************************************   named point processes    *********************************************/

HawkesProcess::HawkesProcess(std::string n): PointProcess(n){};

HawkesProcess::HawkesProcess(std::string n,unsigned int k, double ts, double te): PointProcess(n,k,ts,te){}

HawkesProcess::HawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p): PointProcess(n,k,ts,te){
	
	for(auto iter=m.begin(); iter!=m.end();iter++){
		mu.push_back((*iter)->clone());
	}
	
	for(auto iter=p.begin(); iter!=p.end();iter++){
		std::vector<Kernel*> phi_k;
		for(auto iter2=iter->begin();iter2!=iter->end();iter2++){
			phi_k.push_back((*iter2)->clone());
		}
		phi.push_back(phi_k);
	}
	
	
}

HawkesProcess::HawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m): PointProcess(n,k,ts,te) {
	for(auto iter=m.begin(); iter!=m.end();iter++){
		mu.push_back((*iter)->clone());
	}
}

HawkesProcess::HawkesProcess(std::ifstream &file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

HawkesProcess::~HawkesProcess(){
	for(long unsigned int run_id=0;run_id<mcmc_state.size();run_id++){
		for(long unsigned int l=0;l!=mcmc_state[run_id].seq.size();l++)
			mcmc_state[run_id].seq[l].flush();
		forced_clear(mcmc_state[run_id].seq);
	}

	//free posterior samples
	if(!mu_posterior_samples.empty()){
		for(long unsigned int k=0;k<mu_posterior_samples.size();k++){
			if(!mu_posterior_samples[k].empty()){
				for(unsigned int p=0;p<mu[k]->nofp;p++){
					if(!mu_posterior_samples[k][p].empty()){
						for(long unsigned int r=0;r<mu_posterior_samples[k][p].size();r++){
							if(!mu_posterior_samples[k][p][r].empty()){
								mu_posterior_samples[k][p][r].clear();
							}
						}
						if(!mu_posterior_samples[k][p].empty())
							forced_clear(mu_posterior_samples[k][p]);
					}
				}
				if(!mu_posterior_samples[k].empty())
					forced_clear(mu_posterior_samples[k]);
			}
		}
		if(!mu_posterior_samples.empty())
			forced_clear(mu_posterior_samples);

	}
	
	if(!phi_posterior_samples.empty()){
		for(long unsigned int k=0;k<phi_posterior_samples.size();k++){
			if(!phi_posterior_samples[k].empty()){
				for(long unsigned int k2=0;k2<phi_posterior_samples[k].size();k2++){
					if(!phi_posterior_samples[k][k2].empty()){
						for(unsigned int p=0;p<phi[k][k2]->nofp;p++){
							if(!phi_posterior_samples[k][k2][p].empty()){
								for(long unsigned int r=0;r<phi_posterior_samples[k][k2][p].size();r++){
									if(!phi_posterior_samples[k][k2][p][r].empty()){
										phi_posterior_samples[k][k2][p][r].clear();
									}
								}
								if(!phi_posterior_samples[k][k2][p].empty())
									forced_clear(phi_posterior_samples[k][k2][p]);
									
							}
						}
						if(!phi_posterior_samples[k][k2].empty())
							forced_clear(phi_posterior_samples[k][k2]);
							
					}
				}
				if(!phi_posterior_samples[k].empty())
					forced_clear(phi_posterior_samples[k]);
					
			}
		
		}
		if(!phi_posterior_samples.empty())
			forced_clear(phi_posterior_samples);
			
	}
	
	//free mutual excitation kernels
	for(auto iter=phi.begin();iter!=phi.end();iter++)
		for(auto iter_2=iter->begin(); iter_2!=iter->end();iter_2++)
			if(!(*iter_2))
				free(*iter_2);
	
	if(!phi.empty()){
		for(long unsigned int k=0;k<phi.size();k++){
			if(!phi[k].empty())
				forced_clear(phi[k]);
				
		}
		forced_clear(phi);
		
	}
	//free background intensity kernels
	if(!mu.empty())
		forced_clear(mu);
	
	
	//free posterior mean mutual excitation kernels
	for(auto iter=post_mean_phi.begin();iter!=post_mean_phi.end();iter++)
		for(auto iter_2=iter->begin(); iter_2!=iter->end();iter_2++)
			if(!(*iter_2))
				free(*iter_2);
	
	if(!post_mean_phi.empty()){
		for(long unsigned int k=0;k<post_mean_phi.size();k++){
			if(!post_mean_phi[k].empty())
				forced_clear(post_mean_phi[k]);
			
		}
		forced_clear(post_mean_phi);
		
	}
	//free posterior mean background intensity kernels
	if(!post_mean_mu.empty())
		forced_clear(post_mean_mu);
	

	//free posterior mode mutual excitation kernels
	for(auto iter=post_mode_phi.begin();iter!=post_mode_phi.end();iter++)
		for(auto iter_2=iter->begin(); iter_2!=iter->end();iter_2++)
			if(!(*iter_2))
				free(*iter_2);

	if(!post_mode_phi.empty()){
		for(long unsigned int k=0;k<post_mode_phi.size();k++){
			if(!post_mode_phi[k].empty())
				forced_clear(post_mode_phi[k]);
		}
		forced_clear(post_mode_phi);
	}
	//free posterior mean background intensity kernels
	if(!post_mode_mu.empty())
		forced_clear(post_mode_mu);

}

/****************************************************************************************************************************************************************************
 * Model Utility Methods
******************************************************************************************************************************************************************************/

std::string HawkesProcess::createModelDirectory() const{
	
	boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
    std::string mcmc_curr_path_str=curr_path_boost.string();
    std::string dir_path_str;
    if(!name.empty())
    	dir_path_str=mcmc_curr_path_str+"/"+name+"/";
    else
    	dir_path_str=mcmc_curr_path_str+"/hp/";
    
    if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the mcmc plots."<<std::endl;
    
    return dir_path_str;
    
}

std::string HawkesProcess::createModelInferenceDirectory() const{
	
	std::string dir_path_str=createModelDirectory();
	dir_path_str+="/inference_plots/";
	if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the mcmc plots."<<std::endl;
	
	return dir_path_str;
}

/****************************************************************************************************************************************************************************
 * Model Serialization/Deserialization Methods
******************************************************************************************************************************************************************************/

void HawkesProcess::State::serialize(boost::archive::text_oarchive &ar, const unsigned int version){
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & mu;
	ar & phi;
	ar & seq;
	ar & observed_seq_v;
	ar & seq_v;
	ar & mcmc_iter;
	
}


void HawkesProcess::State::serialize(boost::archive::text_iarchive &ar, const unsigned int version){
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & mu;
	ar & phi;
	ar & seq;
	ar & observed_seq_v;
	ar & seq_v;
	ar & mcmc_iter;
	
}

void HawkesProcess::State::serialize(boost::archive::binary_iarchive &ar, const unsigned int version){
	

	ar & K;
	ar & start_t;
	ar & end_t;
	ar & mu;
	ar & phi;
	ar & seq;
	ar & observed_seq_v;
	ar & seq_v;
	ar & mcmc_iter;
	
}

void HawkesProcess::State::serialize(boost::archive::binary_oarchive &ar, const unsigned int version){
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & mu;
	ar & phi;
	ar & seq;
	ar & observed_seq_v;
	ar & seq_v;
	ar & mcmc_iter;
	
}

void HawkesProcess::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<HawkesProcess,PointProcess>();
	    
	    ar & boost::serialization::base_object<PointProcess>(*this);
		ar & mu;
		ar & phi;	
		ar & post_mean_mu;
		ar & post_mean_phi;
		ar & post_mode_mu;
		ar & post_mode_phi;
		ar & mcmc_state;
		ar & phi_posterior_samples;
		ar & mu_posterior_samples;
		ar & profiling_mcmc_step;
		ar & profiling_parent_sampling;
		ar & profiling_intensity_sampling;

}

void HawkesProcess::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<HawkesProcess,PointProcess>();
	    ar & boost::serialization::base_object<PointProcess>(*this);
		ar & mu;
		ar & phi;
		ar & post_mean_mu;
		ar & post_mean_phi;
		ar & post_mode_mu;
		ar & post_mode_phi;
		ar & mcmc_state;
		ar & phi_posterior_samples;
		ar & mu_posterior_samples;
		ar & profiling_mcmc_step;
		ar & profiling_parent_sampling;
		ar & profiling_intensity_sampling;
	   
}

void HawkesProcess::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<HawkesProcess,PointProcess>();
	    ar & boost::serialization::base_object<PointProcess>(*this);
	    
		ar & mu;
		
		ar & phi;
		ar & post_mean_mu;
		ar & post_mean_phi;
		ar & post_mode_mu;
		ar & post_mode_phi;
		ar & mcmc_state;
		ar & phi_posterior_samples;
		ar & mu_posterior_samples;
		ar & profiling_mcmc_step;
		ar & profiling_parent_sampling;
		ar & profiling_intensity_sampling;
	    
	

}

void HawkesProcess::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<HawkesProcess,PointProcess>();
	    ar & boost::serialization::base_object<PointProcess>(*this);
		ar & mu;
		ar & phi;
		ar & post_mean_mu;
		ar & post_mean_phi;
		ar & post_mode_mu;
		ar & post_mode_phi;
		ar & mcmc_state;
		ar & phi_posterior_samples;
		ar & mu_posterior_samples;
		ar & profiling_mcmc_step;
		ar & profiling_parent_sampling;
		ar & profiling_intensity_sampling;

}

void HawkesProcess::save(std::ofstream & file) const{
	
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void HawkesProcess::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void HawkesProcess::saveParameters(std::ofstream & file) const{
	for(unsigned int k=0;k<K;k++){
		mu[k]->save(file);
		for(unsigned int k2=0;k2<K;k2++){
			phi[k][k2]->save(file);
		}
	}
}


void HawkesProcess::loadData(std::ifstream &file, std::vector<EventSequence> & data, unsigned int file_type, const std::string & seq_name, double t0, double t1){
	EventSequence data_seq;
//	data_seq.start_t=start_t;
//	data_seq.end_t=end_t;
	data_seq.type.resize(K+1);
	data_seq.K=K+1;//the virtual event which corresponds to the background process has an additional type
	data_seq.N=0;
	data_seq.Nt=0;
	data_seq.thinned_type.resize(K+1);
	//load the sequence
	data_seq.load(file,file_type);
	data_seq.name=!seq_name.empty()? seq_name:name;
	//add virtual event if it doesn't exist 
	Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
	data_seq.addEvent(ve,0);

	data_seq.start_t=t0>=0?t0:start_t;//to compute the likelihood in the specificied interval
	double last_arrival_time=data_seq.full.rbegin()->first;
	data_seq.end_t=(t1>data_seq.start_t)?t1:last_arrival_time;
	data.push_back(data_seq);
    boost::filesystem::path ghp_curr_path_boost=boost::filesystem::current_path();
 
}


void HawkesProcess::loadData(std::vector<std::ifstream> &files, std::vector<EventSequence> & data, unsigned int file_type, double t0, double t1){
	unsigned int seq_id=0;
	for(auto file_iter=files.begin();file_iter!=files.end();file_iter++){
		EventSequence data_seq;
//		data_seq.start_t=start_t;
//		data_seq.end_t=end_t;
		data_seq.type.resize(K+1);
		data_seq.K=K+1;//the virtual event which corresponds to the background process has an additional type
		data_seq.N=0;
		data_seq.Nt=0;
		data_seq.thinned_type.resize(K+1);

		//load the sequence
		data_seq.load(*file_iter,file_type);

		//add virtual event if it doesn't exist
		Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
		data_seq.addEvent(ve,0);
	    data_seq.name=name+"_sequence_"+std::to_string(seq_id);
		data_seq.start_t=t0>=0?t0:start_t;//to compute the likelihood in the specificied interval
		double last_arrival_time=data_seq.full.rbegin()->first;
		data_seq.end_t=(t1>data_seq.start_t)?t1:last_arrival_time;
		
	    seq_id++;
		data.push_back(data_seq);
	}
}

/****************************************************************************************************************************************************************************
 * Model Generation Methods
******************************************************************************************************************************************************************************/

//Generates a new model from the priors. it is used for generation of synthetic datasets
void HawkesProcess::generate(){
	//generate the model parameters from the priors
	for(auto iter=mu.begin();iter!=mu.end();iter++){
		(*iter)->generate();
	}
	for(auto iter=phi.begin();iter!=phi.end();iter++)
		for(auto iter2=iter->begin();iter2!=iter->end();iter2++)
			(*iter2)->generate();

}

/****************************************************************************************************************************************************************************
 * Model Simulation Methods
******************************************************************************************************************************************************************************/
//non-static methods
EventSequence HawkesProcess::simulate(unsigned int id){
	return(HawkesProcess::simulate(mu, phi, start_t, end_t, name, id));
}


void HawkesProcess::simulate(EventSequence & seq, double dt){
	return(HawkesProcess::simulate(mu, phi, seq, dt));
	
}

void HawkesProcess::simulateNxt(const EventSequence & seq, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N){
	return(HawkesProcess::simulateNxt(mu, phi, seq, start_t, end_t, nxt_events, N));
	
}

void HawkesProcess::simulate(Event *e, int k, double ts, double te, EventSequence &s) {
	HawkesProcess::simulate(phi,e,k,ts,te,s);
}

//static methods

EventSequence HawkesProcess::simulate(std::vector< ConstantKernel *> & mu, std::vector<std::vector< Kernel *>>  & phi, double start_t, double end_t, std::string name, unsigned int id){
	
	std::string seq_name;
	if(!name.empty())
		seq_name=name+"_"+"sequence_"+std::to_string(id);
	else
		seq_name="hp_sequence_"+std::to_string(id);
	
	unsigned int K= mu.size();

	EventSequence hp_seq(seq_name, K+1, start_t, start_t);
	
	//create a virtual event at 0 which corresponds to the background process. d
	Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
	hp_seq.addEvent(ve,0);

	
	HawkesProcess::simulate(mu, phi, hp_seq, end_t-start_t);

	return hp_seq;
}

//simulate the model for a time interval of length dt and expand the event sequence s with the generated events that lie in this time window 
void HawkesProcess::simulate(std::vector< ConstantKernel *> & mu, std::vector<std::vector<Kernel *>> & phi, EventSequence &s, double dt){
	
	s.end_t+=dt;//expand the event sequence by dt
	unsigned int K=mu.size(); //nof types
	
	
	//simulate the non-homogeneous poisson processeses of the events that are already in the event sequence, if the event sequence is empty, this step is skipped and the process is simulated from scratch
	for(auto e_iter=std::next(s.full.begin()); e_iter!=s.full.end(); e_iter++){
		if(!phi.empty()){
			//std::cout<<"start generating the clusters\n";
			for(unsigned int k2=0;k2<K;k2++)
				HawkesProcess::simulate(phi, e_iter->second, k2, s.end_t-dt,s.end_t,s);
		}
	}
	Event *ve=s.full.begin()->second;
	
	for(unsigned int k=0; k<K; k++){
		//generate arrival times in the background process
		std::vector<double> arrival_times=simulateHPoisson(mu[k]->p[0],s.end_t-dt,s.end_t);
		long unsigned int N=arrival_times.size();		
		for(long unsigned int i=0;i<N;i++){
			Event *e=new Event{s.type[k].size(),(int)k,arrival_times[i],K+1, ve};//create an event that belongs to the background process (trigerred by the exogenous intensity)
			s.addEvent(e,ve);
			// if the process is mutually exciting, add events triggered by the event e
			if(!phi.empty()){
				for(unsigned int k2=0;k2<K;k2++) //the descendants of type k2 are generated
					HawkesProcess::simulate(phi, e, k2, e->time, s.end_t, s);
			}
		}
	}	
}


//get descendants of type k of the event e that lie in the time interval [ts,te] and add them to sequence hp_seq
void HawkesProcess::simulate(std::vector<std::vector<Kernel *>>  & phi, Event *e, int k, double ts, double te, EventSequence &hp_seq) {//TODO: change it phi doesn't have to be 2d!!!
	unsigned int K=phi.size();
	//generate the arrival times of the events generated by event e - get the immediate offsprings
	if(!phi[e->type][k])
		std::cerr<<"invalid trigerring kernel\n";
	//std::vector<double> arrival_times=simulateNHPoisson(*phi[e->type][k],ts, ts, te);
	std::vector<double> arrival_times=simulateNHPoisson(*phi[e->type][k],e->time, ts, te);
	long unsigned int N=arrival_times.size();

	for(long unsigned int i=0;i<N;i++){ //get the descendents of event e at level greater than one
		Event *oe=new Event{hp_seq.type[k].size(),k,arrival_times[i],K+1,e};//add the immediate offsprings in the sequence
		hp_seq.addEvent(oe,e);
		for(long unsigned int k2=0; k2<K; k2++){//generate the offsprings of offsprings
			HawkesProcess::simulate(phi,oe,k2,arrival_times[i],te, hp_seq);
		}
	}
}

void HawkesProcess::simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s,  double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N){
	simulateNxt(mu, phi, s, start_t,  start_t, end_t, nxt_events, N);
}

void HawkesProcess::simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s, double t0, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N){
	
	if(N==0 || end_t<=start_t)//no more events to generate, stop the recursion
		return;
	
	unsigned int K=mu.size(); //nof types

	//simulate the non-homogeneous poisson processeses of the events that are already in the event sequence, if the event sequence is empty, this step is skipped  to get their next event
	std::map<double,std::pair< int, Event *>> nxt_e; //it contains the arrival time of the next event, and the event that it trigerred it , and the type of the event : the key is the arrival time of the event
	for(auto e_iter=std::next(s.full.begin()); e_iter!=s.full.end(); e_iter++){
		if(e_iter->second->time>=t0)
			break;
		if(!phi.empty()){
			for(unsigned int k=0;k!=K;k++){
				double start_e=start_t;
				if(e_iter->second->offsprings){
					
					for(auto e_iter2=e_iter->second->offsprings->type[k].begin();e_iter2!=e_iter->second->offsprings->type[k].end();e_iter2++)
						if(e_iter2->second->time<=start_t){
							start_e=e_iter2->second->time;
						}
						else break;
				}
					
				std::vector<double> arrival_times(1,start_e);
	
					arrival_times=simulateNHPoisson(*phi[e_iter->second->type][k],e_iter->second->time, arrival_times[0], end_t, 1);//todo: replace the start_t 
		
				if(!arrival_times.empty() && arrival_times[0]>start_t ){
					nxt_e[arrival_times[0]]=std::make_pair(k, e_iter->second);
				
				}
				
			}
		}
	}
	
	//simulate the non-homogeneous poisson processeses of the events from the previous model simulation steps
	for(auto e_iter=nxt_events.begin();e_iter!=nxt_events.end();e_iter++){
		if((*e_iter)->time>=t0)
			break;
		if(!phi.empty()){
			for(unsigned int k=0;k!=K;k++){
				
				//the simulation should start from the last event generated by the event e_iter before time start_t
				double start_e=start_t;
				if((*e_iter)->offsprings){
					for(auto e_iter2=(*e_iter)->offsprings->type[k].begin();e_iter2!=(*e_iter)->offsprings->type[k].end();e_iter2++)
						if(e_iter2->second->time<=start_t){
							start_e=e_iter2->second->time;
						}
						else break;
				}
					
				std::vector<double> arrival_times(1,start_e);
		
					arrival_times=simulateNHPoisson(*phi[(*e_iter)->type][k],(*e_iter)->time, arrival_times[0], end_t, 1);//todo: replace the start_t 
				//

				//if an event was generated in the interval [start_t,end_t] of the process, it may be candidate next event (check that the next event is not before start_t)
				if(!arrival_times.empty() && arrival_times[0]>start_t ){
					nxt_e[arrival_times[0]]=std::make_pair(k, *e_iter);
				
				}
			}
		}
	}
	Event *ve=s.full.begin()->second;
	//simulate the homogeneous poisson processeses of the types to get their next event
	for(unsigned int k=0; k<K; k++){
		//generate arrival times in the background process
			double start_e=start_t;
			for(auto e_iter2=ve->offsprings->type[k].begin();e_iter2!=ve->offsprings->type[k].end();e_iter2++)
				if(e_iter2->second->time<=start_t){
					start_e=e_iter2->second->time;
				}
				else break;
				
			std::vector<double> arrival_times(1,start_e);
			arrival_times=simulateHPoisson(mu[k]->p[0],arrival_times[0],end_t, 1);
			
			if(!arrival_times.empty() && arrival_times[0]>start_t ){
				nxt_e[arrival_times[0]]=std::make_pair(k, ve);
			}
	}	
	//add the next event in the sequence: the map is ordered according to the next event of each poisson process
	if(!nxt_e.empty()){
		int nxt_k=nxt_e.begin()->second.first; //get the type of next event
		Event *nxt_p=nxt_e.begin()->second.second;  //get the parent of next event
		double nxt_t= nxt_e.begin()->first; //get the time of next event
		Event *nxt_event=new Event{s.type[nxt_k].size(),nxt_k,nxt_t,K+1,nxt_p};//todo: add to a vector of events 
		nxt_events.push_back(nxt_event);
	
		//get the next event: continue the recursion
		simulateNxt(mu, phi, s, nxt_t, end_t, nxt_events, N-1);
	}
	else{//no event was generated within the interval, the recursion stops
		return ;
	}

}



/****************************************************************************************************************************************************************************
 * Model Print Methods
******************************************************************************************************************************************************************************/

void HawkesProcess::print(std::ostream &file)const{

	 file<<"Number of event types: "<<K<<std::endl;
	 file<<"Observation  window begin: "<<start_t<<std::endl;
	 file<<"Observation  window end: "<<end_t<<std::endl;

	 if(mu.size()<K) return;
	 file<<"Background Intensity."<<std::endl;
	
	 for(unsigned int k=0;k<K;k++){
		 file<<"Parameters from type  "<<k<<"."<<std::endl;
	
		 if(!mu[k]->p.empty())
			 mu[k]->print(file);
	
		 if(!post_mean_mu.empty()){
			 file<<"Posterior Mean Parameters from type  "<<k<<"."<<std::endl;
			 file<<"Background Intensity."<<std::endl;
			 post_mean_mu[k]->print(file);
		 }
		 if(!post_mode_mu.empty()){
			 file<<"Posterior Mode Parameters from type  "<<k<<"."<<std::endl;
			 file<<"Background Intensity."<<std::endl;
			 post_mode_mu[k]->print(file);
			
		 }
	 }
	 
	 if(phi.size()<K) return; //the process is Poisson
	 file<<"Trigerring Kernels."<<std::endl;
	 for(unsigned int k=0;k<K;k++){
		 if(phi[k].size()<K) return;
		 file<<"Parameters from type  "<<k<<"."<<std::endl;

		 for(unsigned int k2=0;k2<K;k2++){
			 file<<"Kernel parameters to type "<<k2<<std::endl;
			 if(phi[k][k2])
				 phi[k][k2]->print(file);
			 if(post_mode_phi.size()!=K)
				 std::cerr<<"1.unitialized posterior mode model kernel param\n";
			 else{
				 if(post_mode_phi[k].size()!=K)
					 std::cerr<<"2.unitialized posterior mode model kernel param\n";
				 else{
				 	 if(!post_mode_phi[k][k2])
				 	  std::cerr<<"3.unitialized posterior mode model kernel param\n";
				 }
			 }

			 if(post_mode_phi.size()==K && post_mode_phi[k].size( )==K && post_mode_phi[k][k2]){
				 file<<"Posterior Mode Kernel parameters to type "<<k2<<std::endl;
				 post_mode_phi[k][k2]->print(file);
			}
			if(post_mean_phi.size()!=K)
				std::cerr<<"1.unitialized posterior mode model kernel param\n";
			else{
				if(post_mean_phi[k].size()!=K)
					std::cerr<<"2.unitialized posterior mode model kernel param\n";
				else{
					if(!post_mean_phi[k][k2])
						std::cerr<<"3.unitialized posterior mode model kernel param\n";
				}

			}
			if(post_mean_phi.size()==K && post_mean_phi[k].size( )==K && post_mean_phi[k][k2]){
				 file<<"Posterior Mean Kernel parameters to type "<<k2<<std::endl;
				 post_mean_phi[k][k2]->print(file);
		   }
		 }
	 }
}

void HawkesProcess::State::print(std::ostream &file) const{
	
     file<<"----------------New Sample from the Posterior:----------------------\n";
     file<<"\nModel Parameters: \n";
     
     if((unsigned int) mu.size()<K) return;
	 for(unsigned int k=0;k<K;k++){
		 file<<"Parameters from type  "<<k<<"."<<std::endl;
		 file<<"Background Intensity."<<std::endl;
		 mu[k]->print(file);
	 }
	 
	 if((unsigned int) phi.size()<K) return;
	 file<<"Trigerring kernels."<<std::endl;//inference for standard Poisson
	 for(unsigned int k=0;k<K;k++){
		 if(phi[k].size()<K) return;
		 file<<"Parameters from type  "<<k<<"."<<std::endl;
		 for(unsigned int k2=0;k2<K;k2++){
			 file<<"Kernel parameters to type "<<k2<<std::endl;
			 phi[k][k2]->print(file);
		 }
	 }
	 
	 file<<"\nEvent Sequence: \n";
	 for(long unsigned int l=0;l!=seq.size();l++)
		 seq[l].print(file);
}


/****************************************************************************************************************************************************************************
 * Goodness-of-fit Methods
******************************************************************************************************************************************************************************/
void HawkesProcess::goodnessOfFitMatlabScript(const std::string & dir_path_str, const std::string & seq_path_dir, std::string model_name, std::vector<EventSequence> & data, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi) {
	
	std::string matlab_script_name=dir_path_str+"/"+model_name+"_kstest.m";
	std::ofstream matlab_script;
	matlab_script.open(matlab_script_name);

	matlab_script<<"%intensity  functions which correspond to the spike trains and with posterior mode estimates"<<std::endl;

	long unsigned int K=mu.size();
	matlab_script<<"clear all;"<<std::endl;
	//write the anonymous functions which correspond to the event sequence
	for(auto seq_iter=data.begin();seq_iter!=data.end();seq_iter++){
		std::string matlab_lambda_func="lambda_"+seq_iter->name+" = @(t) ";//matlab  anonymous function prefix
		for(unsigned int k=0;k<K;k++){
			std::string matlab_lambda_expr;
			std::vector<Kernel *> phi_k;
			for(auto iter=phi.begin();iter!=phi.end();iter++){
				phi_k.push_back((*iter)[k]);
			}
			if(phi_k.empty())
				printMatlabExpressionIntensity(mu[k], *seq_iter, matlab_lambda_expr); // matlab expression for intensity which corresonds to the sequence 
			else
				printMatlabExpressionIntensity(mu[k], phi_k, *seq_iter, matlab_lambda_expr); // matlab expression for intensity which corresonds to the sequence 
			matlab_lambda_func+=matlab_lambda_expr;
		}
		
		matlab_lambda_func+=";";//terminate matlab command
		matlab_script<<matlab_lambda_func<<std::endl;
	}
	
	matlab_script<<"%interarrivals times of the rescaled spike arrivals across all sequences"<<std::endl;
	matlab_script<<"interarrival_times=[];"<<std::endl;
	
	for(auto seq_iter=data.begin();seq_iter!=data.end();seq_iter++){
		std::string matlab_lambda_func="lambda_"+seq_iter->name;//matlab  anonymous function prefix
		matlab_script<<"%read the spike train file and process for sequence "<<seq_iter->name<<std::endl;
		matlab_script<<"seq = csvread(strcat('"<<seq_path_dir<<seq_iter->name<<".csv'),1,0)";
		matlab_script<<"%get the arrival times"<<std::endl;
		matlab_script<<"seq=seq(:,1);"<<std::endl;
		matlab_script<<"nof_spikes=length(seq);"<<std::endl;
		matlab_script<<"rescaled_seq=[];"<<std::endl;
		matlab_script<<"for i=1:nof_spikes"<<std::endl;
		matlab_script<<"	ti=seq(i)"<<std::endl;
		matlab_script<<"	%use the time-rescaling theorem"<<std::endl;
		matlab_script<<"	%ui should correspond to spike time coming from homogeneous poisson with unit lambda"<<std::endl;
		matlab_script<<"	ui=integral("<<matlab_lambda_func<<",0,ti);"<<std::endl;
		matlab_script<<"	rescaled_seq=[rescaled_seq, ui];"<<std::endl;
		matlab_script<<"	if i>1"<<std::endl;
		matlab_script<<"    	zi=rescaled_seq(i)-rescaled_seq(i-1)"<<std::endl;
		matlab_script<<"   		interarrival_times=[interarrival_times,zi];"<<std::endl;
		matlab_script<<"	end"<<std::endl;
		matlab_script<<"end"<<std::endl;
	 
		matlab_script<<"csvwrite('"<<dir_path_str<<"rescaled_sequence_"<<seq_iter->name<<".csv',rescaled_seq');"<<std::endl;

	}
	matlab_script<<"csvwrite('"<<dir_path_str<<model_name<<"_rescaled_interarrival_times.csv',interarrival_times');"<<std::endl;
	
	matlab_script<<"[h,p,ksstat,cv]=kstest(interarrival_times', [interarrival_times',expcdf(interarrival_times',1)]);"<<std::endl;
	matlab_script.close();
	
}

/****************************************************************************************************************************************************************************
 * Model Learning Methods
******************************************************************************************************************************************************************************/


void HawkesProcess::fit(const MCMCParams *mcmc_params){
	
	HawkesProcessMCMCParams *hp_mcmc_params=(HawkesProcessMCMCParams *)mcmc_params;
	if(hp_mcmc_params->dir_path_str.empty()){
		boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
		std::string curr_path_str=curr_path_boost.string();
		if(!name.empty())
		    hp_mcmc_params->dir_path_str=curr_path_str+"/"+name+"/";
		else
			hp_mcmc_params->dir_path_str=curr_path_str+"/hp/";
		if(!boost::filesystem::is_directory(hp_mcmc_params->dir_path_str) && !boost::filesystem::create_directory(hp_mcmc_params->dir_path_str))
			std::cerr<<"Couldn't create auxiliary folder."<<std::endl;
	}
		
	if(train_data.empty()) return;
	bool empty_sequences_flag=true;
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
		empty_sequences_flag=!(seq_iter->N>2);//even if there's an event it is the virtual event. there should be at least two events
	//all the event sequences provided for training are degenerate
	if(empty_sequences_flag)
		return;

	fit_(hp_mcmc_params);
	
}

void HawkesProcess::fit_(HawkesProcessMCMCParams const * const mcmc_params){
	
	
	//initialize the structures for the posterior samples and mcmc states if they are empty
	if(mcmc_state.empty())
		mcmcInit(mcmc_params);
	
	//initialize the structures for keeping the inference time
	if(mcmc_params->profiling && profiling_mcmc_step.empty()){
		initProfiling(mcmc_params->runs, mcmc_params->max_num_iter);
	}

	
	//initialize the structures for synchronizing the inference threads
	mcmcSynchronize(mcmc_params->mcmc_fit_nof_threads);
	
	pthread_t mcmc_run_threads[mcmc_params->mcmc_fit_nof_threads];
	unsigned int nof_runs_thread[mcmc_params->mcmc_fit_nof_threads];

	unsigned int nof_runs_thread_q=mcmc_params->runs/mcmc_params->mcmc_fit_nof_threads;
	unsigned int nof_runs_thread_m=mcmc_params->runs%mcmc_params->mcmc_fit_nof_threads;
	
	unsigned int fit_nof_threads_created=0;
	for(unsigned int t=0;t<mcmc_params->mcmc_fit_nof_threads;t++){
		nof_runs_thread[t]=nof_runs_thread_q;
		if(t<nof_runs_thread_m)
			nof_runs_thread[t]++;
		if(nof_runs_thread[t])
			fit_nof_threads_created++;
	}
	 
	//split the runs across the threads
	unsigned int run_id_offset=0;
	for(unsigned int thread_id=0;thread_id<mcmc_params->mcmc_fit_nof_threads;thread_id++){
	    if(nof_runs_thread[thread_id]){
	    	break;
	    }

		InferenceThreadParams* p;
		p= new InferenceThreadParams(thread_id, this,run_id_offset,nof_runs_thread[thread_id], mcmc_params->nof_burnin_iters, mcmc_params->max_num_iter,mcmc_params->mcmc_iter_nof_threads, mcmc_params->profiling, fit_nof_threads_created, &nof_fit_threads_done, &fit_mtx, &fit_con);
		
		if(!p)
			std::cerr<<"Unable to allocate memory for the thread\n";
	    int rc = pthread_create(&mcmc_run_threads[thread_id], NULL, HawkesProcess::fit_, (void *)p);
	    if (rc){
	         //std::cout << "Error:unable to create thread," << rc << std::endl;
	         exit(-1);
	    }
	    
	    run_id_offset+=nof_runs_thread[thread_id];	
	}
	
	mcmcPolling(mcmc_params->dir_path_str, fit_nof_threads_created);
	
	
	for (unsigned int i = 0; i < mcmc_params->mcmc_fit_nof_threads; i++){
	    if(nof_runs_thread[i])
	    	pthread_join (mcmc_run_threads [i], NULL);
	}
	
	if(mcmc_params->profiling){
		writeProfiling(mcmc_params->dir_path_str);
	}
}

//periodically save model and posterior samples, while waiting for the generated threads to finish
void HawkesProcess::mcmcPolling(const std::string &dir_path_str, unsigned int fit_nof_threads){
	

	pthread_mutex_lock(&fit_mtx);
	while(nof_fit_threads_done!=fit_nof_threads){
		struct timeval tp;

		gettimeofday(&tp, NULL);
		struct timespec max_wait = {0, 0};
		max_wait.tv_nsec+=tp.tv_usec*1000;
		max_wait.tv_sec=tp.tv_sec;
		max_wait.tv_sec += MCMC_SAVE_PERIOD;
		pthread_cond_timedwait(&fit_con, &fit_mtx, &max_wait);
		std::string model_filename{dir_path_str+name+"_infer_ser.txt"};
		std::ofstream model_file{model_filename};
	
		//requires extra synchronization between the children threads so that mcmc_state and posterior_samples are not changed when they are serialized at the same time
		for(unsigned int i=0;i!=fit_nof_threads;i++)
			pthread_mutex_lock(save_samples_mtx[i]);
		
		save(model_file);
		for(unsigned int i=0;i!=fit_nof_threads;i++)
			pthread_mutex_unlock(save_samples_mtx[i]);
	
	}
	pthread_mutex_unlock(&fit_mtx);


}

void HawkesProcess::initProfiling(unsigned int runs,  unsigned int max_num_iter){
	profiling_mcmc_step.resize(runs);
	profiling_parent_sampling.resize(runs);
	profiling_intensity_sampling.resize(runs);
	for(unsigned int i=0;i<runs;i++){
		profiling_mcmc_step[i].resize(max_num_iter);
		profiling_parent_sampling[i].resize(max_num_iter);
		profiling_intensity_sampling[i].resize(max_num_iter);
	}
}

void HawkesProcess::writeProfiling(const std::string &dir_path_str) const{
	long unsigned int runs=profiling_mcmc_step.size();
	long unsigned int max_num_iter=profiling_mcmc_step[0].size();

	//open profiling files
	std::string profiling_total_str= dir_path_str+"infer_total.csv";
	std::ofstream profiling_total_file{profiling_total_str};
	profiling_total_file<<"mcmc step, learning time"<<std::endl;

	std::string profiling_par_str= dir_path_str+"infer_parent_sampling.csv";
	std::ofstream profiling_par_file{profiling_par_str};
	profiling_par_file<<"mcmc step, learning time"<<std::endl;
	
	std::string profiling_in_str= dir_path_str+"infer_intensity_sampling.csv";
	std::ofstream profiling_in_file{profiling_in_str};
	profiling_in_file<<"mcmc step, learning time"<<std::endl;
	
	//get the mean inference time across the runs
	for(unsigned int iter=0;iter<max_num_iter;iter++){
		double total_mcmc_step=0;
		double total_pr_sampling=0;
		double total_in_sampling=0;
		for(unsigned int r=0;r<runs;r++){
			total_mcmc_step+=profiling_mcmc_step[r][iter];
			total_pr_sampling+= profiling_parent_sampling[r][iter];
			total_in_sampling+=profiling_intensity_sampling[r][iter];
		}
		total_mcmc_step/=runs;
		total_pr_sampling/=runs;
		total_in_sampling/=runs;

		profiling_total_file<<iter<<","<<total_mcmc_step<<std::endl;
		profiling_par_file<<iter<<","<<total_pr_sampling<<std::endl;
		profiling_in_file<<iter<<","<<total_in_sampling<<std::endl;
	}
	
	//close the profiling files
	profiling_total_file.close();
	profiling_par_file.close();
	profiling_in_file.close();
}

void* HawkesProcess::fit_( void *p){
	std::unique_ptr< InferenceThreadParams > params( static_cast< InferenceThreadParams * >( p ) );
	HawkesProcess *hp=params->hp;
	for(unsigned int mcmc_run=params->run_id_offset;mcmc_run<params->run_id_offset+params->runs;mcmc_run++){

		//sample first mcmc state and burn-in
		if(!hp->mcmc_state[mcmc_run].mcmc_iter){//TODO: mcmc states as many as the runs not the threads
			hp->mcmcStartRun(params->thread_id,mcmc_run, params->iter_nof_threads);
			for(unsigned int mcmc_iter=0;mcmc_iter!=params->burnin_iter;mcmc_iter++){
				hp->mcmcStep(params->thread_id,mcmc_run,mcmc_iter,false,params->iter_nof_threads, HawkesProcess::mcmcUpdateExcitationKernelParams_,params->profiling);
			}
		}
		for(unsigned int mcmc_iter=hp->mcmc_state[mcmc_run].mcmc_iter;mcmc_iter!=params->max_num_iter;mcmc_iter++){
			hp->mcmcStep(params->thread_id, mcmc_run,mcmc_iter,true,params->iter_nof_threads,HawkesProcess::mcmcUpdateExcitationKernelParams_, params->profiling);
			hp->mcmc_state[mcmc_run].mcmc_iter++;
		}
	}

	pthread_mutex_lock(params->fit_mtx);
	(*params->nof_fit_threads_done)++;
	if(*params->nof_fit_threads_done==params->fit_nof_threads)//if it's the last inference thread, signal the main thread otherwise leave the unlock for the sibling threads
		pthread_cond_signal(params->fit_con);
	pthread_mutex_unlock(params->fit_mtx);

	return 0;
}

void HawkesProcess::mcmcInit(MCMCParams const * const mcmc_params){
	//initialize the structures for storing the posterior samples
	mu_posterior_samples.resize(K);
	
	if(!phi.empty())
		phi_posterior_samples.resize(K);

	for(unsigned int k=0;k<K;k++){
		mu_posterior_samples[k].resize(mu[k]->nofp);
		for(unsigned int p=0;p<mu[k]->nofp;p++){
			
			mu_posterior_samples[k][p].resize(mcmc_params->runs);
			for(unsigned int r=0;r<mcmc_params->runs;r++)
				mu_posterior_samples[k][p][r].resize(mcmc_params->max_num_iter);
		}
		
		if(!phi.empty()){
			phi_posterior_samples[k].resize(K);
			for(unsigned int k2=0;k2<K;k2++){
				phi_posterior_samples[k][k2].resize(phi[k][k2]->nofp);
				for(unsigned int p=0;p<phi[k][k2]->nofp;p++){
					phi_posterior_samples[k][k2][p].resize(mcmc_params->runs);
					for(unsigned int r=0;r<mcmc_params->runs;r++)
						phi_posterior_samples[k][k2][p][r].resize(mcmc_params->max_num_iter);
				}
			}
		}
	}

	//initialize the mcmc states for each thread of the inference
	unsigned int nof_batches=train_data.size()/mcmc_params->nof_sequences;//the same batch of event sequences may be used by multiplie mcmc runs
	//as two extremes: each run will use all of the available event sequences or each run will use only one event sequence for the inference
	mcmc_state.resize(mcmc_params->runs);
	for(unsigned int run_id=0;run_id<mcmc_params->runs;run_id++){
		    State &s=mcmc_state[run_id];
			s.K=K; //nof types
			s.start_t=start_t;  //beginning of the temporal window 
			s.end_t=end_t; //end of the temporal window 
			s.mcmc_params=(HawkesProcessMCMCParams *) mcmc_params;
			s.seq.resize(mcmc_params->nof_sequences);
			s.observed_seq_v.resize(mcmc_params->nof_sequences);
			s.seq_v.resize(mcmc_params->nof_sequences);
			unsigned int sequence_batch_id=(run_id*nof_batches)/mcmc_params->runs;//the batch of event sequences that will be used by the run
			for(unsigned int l=sequence_batch_id*mcmc_params->nof_sequences;l!=(sequence_batch_id+1)*mcmc_params->nof_sequences;l++){
				s.seq[l-sequence_batch_id*mcmc_params->nof_sequences]=train_data[l];//a deep copy of the sequence is created  	
				s.seq[l-sequence_batch_id*mcmc_params->nof_sequences].observed=true; //the sequence is considered to be fully observed so as to account for the thinned events and the parent structure, needed for the hastings ratios
				//store the realized events in a vector to get O(1) access needed for updating the parent structure
				map_to_vector(s.seq[l-sequence_batch_id*mcmc_params->nof_sequences].full, s.observed_seq_v[l-sequence_batch_id*mcmc_params->nof_sequences]);//the vector contains the virtual event
				//reset the thinned events
				s.seq[l-sequence_batch_id*mcmc_params->nof_sequences].flushThinnedEvents();
				//vectorize both the thinned and the realized events and treat them equally
				//add realized events
				map_to_vector(s.seq[l-sequence_batch_id*mcmc_params->nof_sequences].full, s.seq_v[l-sequence_batch_id*mcmc_params->nof_sequences]);
			}
			//initialize the kernels of the state
			for(auto iter=mu.begin();iter!=mu.end();iter++){
				ConstantKernel *mu=(*iter)->clone();
				mu->reset();
				s.mu.push_back(mu);
			}
			for(auto iter=phi.begin();iter!=phi.end();iter++){
				 std::vector<Kernel *> phi_k;
				for(auto iter2=iter->begin();iter2!=iter->end();iter2++){
					Kernel *phi_k_kp=(*iter2)->clone();
					phi_k_kp->reset();
					phi_k.push_back(phi_k_kp);
				}
				s.phi.push_back(phi_k);
			}
	}
}

void HawkesProcess::mcmcSynchronize(unsigned int fit_nof_threads){
	//initialize the synchronization variables
	
	for(unsigned int thread_id=0;thread_id<fit_nof_threads;thread_id++){
		pthread_mutex_t *m=(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(m,NULL);
		update_event_mutex.push_back(m);
		
		pthread_mutex_t *m2=(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(m2,NULL);
		save_samples_mtx.push_back(m2);
	}
	nof_fit_threads_done=0;
	pthread_cond_init(&fit_con, NULL);
	pthread_mutex_init(&fit_mtx, NULL);
}

void HawkesProcess::mcmcStartRun(unsigned int thread_id, unsigned int run_id,unsigned int iter_nof_threads){

	//initialize number of iterations
	mcmc_state[run_id].mcmc_iter=0;
	
	for(long unsigned int l=0;l!=mcmc_state[run_id].seq.size();l++){
		//the virtual event has no parent
		auto e_iter=mcmc_state[run_id].seq[l].full.begin();
		e_iter->second->parent=0;
	
		//the first event that occurs by default belongs to the background process
		auto e_iter2=std::next(mcmc_state[run_id].seq[l].full.begin());
		e_iter2->second->parent=e_iter->second;
	}
	
	pthread_t mcmc_iter_threads[iter_nof_threads];
	unsigned int nof_types_thread[iter_nof_threads];

	unsigned int nof_types_thread_q=K/iter_nof_threads;
	unsigned int nof_types_thread_m=K%iter_nof_threads;
	
	for(unsigned int t=0;t<iter_nof_threads;t++){
		nof_types_thread[t]=nof_types_thread_q;
		if(t<nof_types_thread_m)
			nof_types_thread[t]++;
	}
	 
	unsigned int type_id_offset=0;
	//split the types across the threads, each thread updates the kernel parameters for a batch of types
	for(unsigned int child_thread_id=0;child_thread_id<iter_nof_threads;child_thread_id++){
	    if(!nof_types_thread[child_thread_id]){
	    	break;
	    }

		InitKernelThreadParams* p;
	    p= new InitKernelThreadParams(thread_id, this, run_id, type_id_offset,nof_types_thread[child_thread_id]);
	    int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, HawkesProcess::mcmcStartRun_, (void *)p);
	   
	    if (rc){
	         //std::cout << "Error:unable to create thread," << rc << std::endl;
	         exit(-1);
	    }
	    type_id_offset+=nof_types_thread[child_thread_id];
	   

	}
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <iter_nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}
	//initialize the parent structure if the process is mutually trigerring
	if(!phi.empty())
		mcmcUpdateParents(thread_id,run_id, iter_nof_threads,false);//TODO: is this correct or should it be totally random????
	
}

void * HawkesProcess::mcmcStartRun_(void *p){
	std::unique_ptr< InitKernelThreadParams > params( static_cast< InitKernelThreadParams * >( p ) );
	HawkesProcess *hp=params->hp;
	unsigned int run_id=params->run_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	State & mcmc_state=hp->mcmc_state[run_id];
	mcmc_state.mcmc_iter=0;
	unsigned int K=hp->K;
	unsigned int thread_id=params->thread_id;

	
	//generate the model parameters from the priors
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){
		//initialize the background intensities
		pthread_mutex_lock(hp->save_samples_mtx[thread_id]);
		mcmc_state.mu[k]->generate();

		pthread_mutex_unlock(hp->save_samples_mtx[thread_id]);
		//initialize the kernel parameters
		
		if(!mcmc_state.phi.empty() && mcmc_state.phi[k].size()==K){
			for(unsigned int k2=0;k2<K;k2++){
				pthread_mutex_lock(hp->save_samples_mtx[thread_id]);
				mcmc_state.phi[k][k2]->generate();
				pthread_mutex_unlock(hp->save_samples_mtx[thread_id]);
			}
		}
	}
	
	//std::cout<<"init excitation done\n";
	return 0;
}
 
void HawkesProcess::mcmcUpdateParents(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads, bool profiling,unsigned int step_id){
    //TODO: check does it account for the virtual event????==>this is why I have two std::next
	boost::timer::cpu_timer infer_timer_1;
	for(long unsigned int l=0;l!=mcmc_state[run_id].seq.size();l++){
		if (mcmc_state[run_id].seq[l].N<=2) return;
		//skip the virtual event (it has no parent), skip the first event (it always belongs to the background process)
		pthread_t mcmc_step_threads[iter_nof_threads];
		unsigned int nof_events_thread[iter_nof_threads];//how many events all but except for the last thread, each thread will update
	
		//update parents of realized events
		
		unsigned int nof_events_thread_q=(mcmc_state[run_id].seq[l].N+mcmc_state[run_id].seq[l].Nt-2)/iter_nof_threads; //for the virtual and the first event (which by default belongs to the background process), we won't update their parents
		unsigned int nof_events_thread_m=(mcmc_state[run_id].seq[l].N+mcmc_state[run_id].seq[l].Nt-2)%iter_nof_threads;
		
		for(unsigned int t=0;t<iter_nof_threads;t++){
			nof_events_thread[t]=nof_events_thread_q;
			if(t<nof_events_thread_m)
				nof_events_thread[t]++;
		}
		
		unsigned int event_id_offset=2;//skip the virtual event (of the exogenous intensity) and the first event (trivially trigerred by the exogenous intensity)
	
		//split the events across the threads
		for(unsigned int child_thread_id=0;child_thread_id<iter_nof_threads;child_thread_id++){
			if(!nof_events_thread[child_thread_id])
				break;
		
			UpdateEventThreadParams* p;
			p= new UpdateEventThreadParams(thread_id, this, run_id, l, event_id_offset,nof_events_thread[child_thread_id],update_event_mutex[thread_id]);
			int rc = pthread_create(&mcmc_step_threads[child_thread_id], NULL, HawkesProcess::mcmcUpdateParents_, (void *)p);
			if (rc){
		
				 exit(-1);
			}
			event_id_offset+=nof_events_thread[child_thread_id];
	
		}
		for (unsigned int i = 0; i < iter_nof_threads; i++){
			if(nof_events_thread[i]){
				pthread_join (mcmc_step_threads [i], NULL);
			}
		}
	}
	auto infer_timer_1_nanoseconds = boost::chrono::nanoseconds(infer_timer_1.elapsed().user + infer_timer_1.elapsed().system);
	
	//report inference time if needed
	if(profiling)
		profiling_parent_sampling[run_id][step_id]=(infer_timer_1_nanoseconds.count()*1e-9)+(step_id>0?profiling_parent_sampling[run_id][step_id-1]:0);
}
	
void * HawkesProcess::mcmcUpdateParents_(void *p){	
	std::unique_ptr< UpdateEventThreadParams > params( static_cast< UpdateEventThreadParams * >( p ) );
	HawkesProcess *hp=params->hp;
	unsigned int thread_id=params->thread_id;
	unsigned int seq_id=params->seq_id;
	unsigned int run_id=params->run_id;
	unsigned int event_id_offset=params->event_id_offset;
	unsigned int nof_events=params->nof_events;
	State & mcmc_state=hp->mcmc_state[run_id];
	

	//sample the parent structure (the latent variables)
	std::random_device rd;
	std::mt19937 gen(rd());
	//TODO: do this loop mulltithreaded, skip virtual event for p->thread_id==0
	for(unsigned int event_id=event_id_offset;event_id<event_id_offset+nof_events;event_id++){

		if(event_id>=mcmc_state.seq_v[seq_id].size()){
			std::cout<<event_id<<std::endl;
			std::cerr<<"invalid event id!\n";
		}
		
		Event * e=mcmc_state.seq_v[seq_id][event_id];
		if(!e)
			std::cerr<<"invalid event\n";
				
		//compute the probabilities for the parent assignment for each event in the observed sequence
		std::vector<double> parent_prob;
     	double norm_constant=0.0;

	   //probability that the event belongs to the background process
     	if((unsigned int)e->type>=mcmc_state.mu.size()){
     		std::cerr<<"invalid event type\n";
     	}
		parent_prob.push_back(mcmc_state.mu[e->type]->p[0]);//TODO: check how the types are stored from 0 for the type of the virtual event!
	    norm_constant+=parent_prob.back();

	    //start parsing the history of events, compute the probability that the event was trigerred by another event
	    if(!hp->phi.empty()){
			for(unsigned int event_id_2=1;event_id_2<mcmc_state.observed_seq_v[seq_id].size() && mcmc_state.observed_seq_v[seq_id][event_id_2]->time<e->time ;event_id_2++){
				Event * e2=mcmc_state.observed_seq_v[seq_id][event_id_2];
				if(!e2)
					std::cerr<<"invalid event\n";
				if(e2->type>=(int)hp->phi.size())
					std::cerr<<"invalid event type\n";
				parent_prob.push_back(mcmc_state.phi[e2->type][e->type]->compute(e->time,e2->time));
				norm_constant+=parent_prob.back();
			}
	    }
	   for(auto iter=parent_prob.begin();iter!=parent_prob.end();iter++){
		  *iter=(*iter)/norm_constant;
	   }

	   std::discrete_distribution<> parent_distribution(parent_prob.begin(), parent_prob.end());

	   //if there is no mutual excitation part in the process, the event is assigned deterministically to the virtual event
	   unsigned int parent=parent_distribution(gen);
	   
	   pthread_mutex_lock((hp->save_samples_mtx[thread_id]));
	   pthread_mutex_lock(params->update_event_mutex);
	   #if DEBUG_LEARNING
	   pthread_mutex_lock(&debug_mtx);
	   debug_learning_file<<"change parents for step "<<run_id<<" and sequence "<<seq_id<<std::endl;

		  debug_learning_file<<"parent for event "<<e->time<<" "<<mcmc_state.observed_seq_v[seq_id][parent]->time<<" "<<mcmc_state.observed_seq_v[seq_id][parent]<<std::endl;
		   if(e->parent)
			   debug_learning_file<<"previous parent for event "<<e->time<<" "<<e->parent->time<<" "<<e->parent<<std::endl;
		   if(mcmc_state.seq[seq_id].full.begin()->second->offsprings)
			   debug_learning_file<<"so far nof events assigned to the background process "<<run_id<<" MCMC step "<<run_id<<" "<<mcmc_state.seq[seq_id].full.begin()->second->offsprings->full.size()<<std::endl;
		   debug_learning_file<<"address of current parent of event in the sequence of the mcmc state "<<e->parent<<std::endl;
		   debug_learning_file<<"address of virtual event in the sequence of the mcmc state "<<mcmc_state.seq[seq_id].full[0]<<std::endl;

	  #endif

	 mcmc_state.seq[seq_id].changeParent(e,mcmc_state.observed_seq_v[seq_id][parent]);
	   
		#if DEBUG_LEARNING
	   	if(mcmc_state.seq[seq_id].full.begin()->second->offsprings)
	   		debug_learning_file<<"so far nof events assigned to the background process after the change "<<run_id<<" MCMC step "<<run_id<<" "<<mcmc_state.seq[seq_id].full.begin()->second->offsprings->full.size()<<std::endl;
	    pthread_mutex_unlock(&debug_mtx);
		#endif
	   pthread_mutex_unlock(params->update_event_mutex);
	   pthread_mutex_unlock((hp->save_samples_mtx[thread_id]));
	}
	
	return 0;
}

void HawkesProcess::mcmcStep(unsigned int thread_id, unsigned int run_id,unsigned int step_id, bool save, unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *), bool profiling /*, GeneralizedHawkesProcess::State *ghs*/){
	//update kernel parameters
	std::cout<<"mcmc step "<<step_id<<std::endl;
	mcmcUpdateExcitationKernelParams(thread_id,run_id,step_id,save,iter_nof_threads, mcmcUpdateExcitationKernelParams_, profiling);
	//update the branching structure
	if(!phi.empty())
		mcmcUpdateParents(thread_id, run_id, iter_nof_threads,profiling,step_id);
	if(profiling){
		profiling_mcmc_step[run_id][step_id]=profiling_parent_sampling[run_id][step_id];
		profiling_mcmc_step[run_id][step_id]+=profiling_intensity_sampling[run_id][step_id];
	}
	
}

void HawkesProcess::mcmcUpdateExcitationKernelParams(unsigned int thread_id,unsigned int run_id,unsigned int step_id, bool save, unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *), bool profiling/*, GeneralizedHawkesProcess::State *ghs*/){
	//TODO: create a separate method updateKernelParam
	//update the model parameters from the posteriors
	
	//TODO: divide the parameters of each type per thread
	pthread_t mcmc_iter_threads[iter_nof_threads];
	unsigned int nof_types_thread[iter_nof_threads];

	//std::cout<<"nof types "<<K<<std::endl;
	unsigned int nof_types_thread_q=K/iter_nof_threads;
	unsigned int nof_types_thread_m=K%iter_nof_threads;
	
	for(unsigned int t=0;t<iter_nof_threads;t++){
		nof_types_thread[t]=nof_types_thread_q;

		if(t<nof_types_thread_m)
			nof_types_thread[t]++;
	}
	  
	
	unsigned int type_id_offset=0;
	boost::timer::cpu_timer infer_timer_1;
	//split the types across the threads, each thread updates the kernel parameters for a batch of types
	for(unsigned int child_thread_id=0;child_thread_id<iter_nof_threads;child_thread_id++){
		//std::cout<<"create thread "<<child_thread_id<<std::endl;
	    if(!nof_types_thread[child_thread_id]){
	    	break;
	    }
		UpdateKernelThreadParams* p;
		p= new UpdateKernelThreadParams(thread_id, this, type_id_offset,nof_types_thread[child_thread_id], run_id, step_id, save);
		if(!p)
			std::cerr<<"unable to allocate memory for thread "<<child_thread_id<<std::endl;
	    int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, mcmcUpdateExcitationKernelParams_, (void *)p);
	    if (rc){
	    	std::cerr << "Error:unable to create thread," << rc << std::endl;
	    	exit(-1);
	    }
	    pthread_join (mcmc_iter_threads [child_thread_id], NULL);
		type_id_offset+=nof_types_thread[child_thread_id];
	}
	//wait for all the kernel parameters to be updated
	for (unsigned int i = 0; i < iter_nof_threads; i++){
		if(nof_types_thread[i]){
			pthread_join (mcmc_iter_threads [i], NULL);
		}
	}

	auto infer_timer_1_nanoseconds = boost::chrono::nanoseconds(infer_timer_1.elapsed().user + infer_timer_1.elapsed().system);

	//report inference time if needed
	if(profiling)
		profiling_intensity_sampling[run_id][step_id]=(infer_timer_1_nanoseconds.count()*1e-9)+(step_id>0?profiling_intensity_sampling[run_id][step_id-1]:0);

}

void * HawkesProcess::mcmcUpdateExcitationKernelParams_(void *p){

	std::unique_ptr< UpdateKernelThreadParams > params( static_cast< UpdateKernelThreadParams * >( p ) );
	HawkesProcess *hp=dynamic_cast<HawkesProcess *>(params->hp);
	unsigned int thread_id=params->thread_id;
	//unsigned int run_id=params->run_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int run_id=params->run_id;
	unsigned int step_id=params->step_id;
	bool save=params->save;
	State & mcmc_state=hp->mcmc_state[run_id];
//
	unsigned int K=hp->K;

	//update the model parameters from the posteriors

//	//TODO: divide the parameters of each type per thread
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){
//		//update the background intensities
		pthread_mutex_lock((hp->save_samples_mtx[thread_id]));
		mcmc_state.mu[k]->mcmcExcitatoryUpdate(k,0,&mcmc_state);
//		std::cout<<"update background intensity done\n";
		pthread_mutex_unlock((hp->save_samples_mtx[thread_id]));
		if(save){
			unsigned int nofp=mcmc_state.mu[k]->nofp;
			std::vector<double> mu_sample(nofp);
			mcmc_state.mu[k]->getParams(mu_sample);
//			//TODO: How can I do it efficiently????
			for(unsigned int p=0;p<nofp;p++){
				pthread_mutex_lock((hp->save_samples_mtx[thread_id]));
				hp->mu_posterior_samples[k][p][run_id](step_id)=mu_sample[p];
				pthread_mutex_unlock((hp->save_samples_mtx[thread_id]));
			}
		}
//
//		//update the kernel parameters
		if(!mcmc_state.phi.empty() && mcmc_state.phi[k].size()==K){
			for(unsigned int k2=0;k2<K;k2++){
				pthread_mutex_lock((hp->save_samples_mtx[thread_id]));

				mcmc_state.phi[k][k2]->mcmcExcitatoryUpdate(k,k2, &mcmc_state);
				pthread_mutex_unlock((hp->save_samples_mtx[thread_id]));
				if(save){
					unsigned int nofp=mcmc_state.phi[k][k2]->nofp;
					std::vector<double> phi_sample(nofp);
					mcmc_state.phi[k][k2]->getParams(phi_sample);
					//pthread_mutex_lock(hp->save_samples_mtx);
					for(unsigned int p=0;p<nofp;p++){
						pthread_mutex_lock((hp->save_samples_mtx[thread_id]));
						hp->phi_posterior_samples[k][k2][p][run_id](step_id)=phi_sample[p];
						pthread_mutex_unlock((hp->save_samples_mtx[thread_id]));
					}
				}
			}
		}
	}
	return 0;
}

void HawkesProcess::computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, 
		                                   std::vector<std::vector<std::vector<double>>> & mean_phi_param, 
										   std::vector<std::vector<double>> & mode_mu_param, 
										   std::vector<std::vector<std::vector<double>>> & mode_phi_param, 
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_posterior_samples, 
										   const std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> & phi_posterior_samples) {
	long unsigned int K=mu_posterior_samples.size();
	for(unsigned int k=0;k<K;k++){
		long unsigned int nofp=mu_posterior_samples[k].size();
	
		for(unsigned int p=0;p<nofp;p++){
			boost::numeric::ublas::vector<double> mu_posterior_samples_all=merge_ublasv(mu_posterior_samples[k][p]);
			mean_mu_param[k][p]=sample_mean(mu_posterior_samples_all);
			double mode_mu_k_p_param=sample_mode(mu_posterior_samples_all);
			mode_mu_param[k][p]=(mode_mu_k_p_param<=0?mean_mu_param[k][p]:mode_mu_k_p_param);//this is due to insufficient precision (nof plot points) in the gaussian kernel regression
		}
	

		std::vector<std::vector<double>> mean_phi_k_param;//it keeps the posterior mean of the parameters of the triggering kernel functions of other types to type k
		std::vector<std::vector<double>> mode_phi_k_param;//it keeps the posterior mode of the parameters of the triggering kernel functions of other types to type k
		for(unsigned int k2=0;k2<K;k2++){
			long unsigned int nofp=phi_posterior_samples[k][k2].size();
			for(unsigned int p=0;p<nofp;p++){
					boost::numeric::ublas::vector<double> phi_posterior_samples_all=merge_ublasv(phi_posterior_samples[k][k2][p]);
					mean_phi_param[k][k2][p]=sample_mean(phi_posterior_samples_all);
					double mode_phi_k_k2_p_param=sample_mode(phi_posterior_samples_all);
					mode_phi_param[k][k2][p]=(mode_phi_k_k2_p_param<=0?mean_phi_param[k][k2][p]:mode_phi_k_k2_p_param);//this is due to insufficient precision (nof plot points) in the gaussian kernel regression
			}
		}
	}
}


void HawkesProcess::computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, 
										   std::vector<std::vector<double>> & mode_mu_param, 
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_posterior_samples) {
	
	long unsigned int K=mu_posterior_samples.size();
	for(unsigned int k=0;k<K;k++){
		long unsigned int nofp=mu_posterior_samples[k].size();
		for(unsigned int p=0;p<nofp;p++){
			boost::numeric::ublas::vector<double> mu_posterior_samples_all=merge_ublasv(mu_posterior_samples[k][p]);
			mean_mu_param[k][p]=sample_mean(mu_posterior_samples_all);
			double mode_mu_k_p_param=sample_mode(mu_posterior_samples_all);
			mode_mu_param[k][p]=(mode_mu_k_p_param<=0?mean_mu_param[k][p]:mode_mu_k_p_param);//this is due to insufficient precision (nof plot points) in the gaussian kernel regression	
		}
	}
}

void HawkesProcess::computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, std::vector<std::vector<std::vector<double>>> & mean_phi_param, 
							std::vector<std::vector<double>> & mode_mu_param, std::vector<std::vector<std::vector<double>>> & mode_phi_param){
	 HawkesProcess::computePosteriorParams(mean_mu_param,mean_phi_param,mode_mu_param,mode_phi_param,mu_posterior_samples,phi_posterior_samples);
}

void HawkesProcess::computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, 
							std::vector<std::vector<double>> & mode_mu_param){
	 HawkesProcess::computePosteriorParams(mean_mu_param,mode_mu_param,mu_posterior_samples);
}

void HawkesProcess::setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
		                       std::vector<std::vector<Kernel *>> & post_mean_phi, 
							   std::vector<ConstantKernel*> & post_mode_mu, 
							   std::vector<std::vector<Kernel *>> & post_mode_phi,
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_posterior_samples,  
							   const std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> & phi_posterior_samples
                               ){
	
	long unsigned int K=phi_posterior_samples.size();
	//compute the posterior point estimations from the samples
	long unsigned int nofp_mu=mu_posterior_samples[0].size();
	std::vector<std::vector<double>> mean_mu_param(K,std::vector<double>(nofp_mu));
	std::vector<std::vector<double>> mode_mu_param(K,std::vector<double>(nofp_mu));
	long unsigned int nofp_phi=phi_posterior_samples[0][0].size();
	std::vector<std::vector<std::vector<double>>> mean_phi_param(K,std::vector<std::vector<double>>(K, std::vector<double>(nofp_phi)));
	std::vector<std::vector<std::vector<double>>> mode_phi_param(K,std::vector<std::vector<double>>(K, std::vector<double>(nofp_phi)));
	
	HawkesProcess::computePosteriorParams(mean_mu_param, mean_phi_param, mode_mu_param, mode_phi_param, mu_posterior_samples,phi_posterior_samples);

	
	//set the kernels to the point estimations
	for(unsigned int k=0;k<K;k++){
		post_mean_mu[k]->setParams(mean_mu_param[k]);
		post_mode_mu[k]->setParams(mode_mu_param[k]);
		for(unsigned int k2=0;k2<K;k2++){
			if(!post_mean_phi[k][k2]){
				std::cerr<<"kernel not properly allocated\n";
				return;//kernel not properly allocated
			}
			post_mean_phi[k][k2]->setParams(mean_phi_param[k][k2]);
			if(!post_mode_phi[k][k2]){
				std::cerr<<"kernel not properly allocated\n";
			    return;
			}
		   post_mode_phi[k][k2]->setParams(mode_phi_param[k][k2]);
		}
	}
}
	
void HawkesProcess::setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
							   std::vector<ConstantKernel*> & post_mode_mu, 
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_posterior_samples 
                               ){
	
	//compute the posterior point estimations from the samples
	long unsigned int K=mu_posterior_samples.size();
	long unsigned int nofp=mu_posterior_samples[0].size();
	std::vector<std::vector<double>> mean_mu_param(K,std::vector<double>(nofp));
	std::vector<std::vector<double>> mode_mu_param(K,std::vector<double>(nofp));
	
	HawkesProcess::computePosteriorParams(mean_mu_param, mode_mu_param, mu_posterior_samples);

	if(post_mode_mu.size()<K)
		std::cerr<<"wrong post mode dimension\n";
	if(mean_mu_param.size()<K)
		std::cerr<<"wrong post mean dimension\n";
	if(mode_mu_param.size()<K)
		std::cerr<<"wrong post mode dimension\n";
	//set the kernels to the point estimations
	for(unsigned int k=0;k<K;k++){
		post_mean_mu[k]->setParams(mean_mu_param[k]);
		post_mode_mu[k]->setParams(mode_mu_param[k]);
	}
}

//TODO: set the hyperparameters from here!
void HawkesProcess::setPosteriorParams(){

	//clear any previous posterior point estimations
	forced_clear(post_mean_mu);
	forced_clear(post_mode_mu);
	//initialize the kernels with the hyperparameters
	for(auto iter=mu.begin();iter!=mu.end();iter++){
		post_mean_mu.push_back(((*iter)->clone()));
		post_mode_mu.push_back(((*iter)->clone()));
	}

	for(long unsigned int k=0;k<post_mode_phi.size();k++){
		forced_clear(post_mean_phi[k]);
		forced_clear(post_mode_phi[k]);
	}

	forced_clear(post_mean_phi);
	forced_clear(post_mode_phi);
	
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		std::vector<Kernel*> mean_phi_k;
		std::vector<Kernel*> mode_phi_k;
		for(auto iter2=iter->begin();iter2!=iter->end();iter2++){
			mean_phi_k.push_back((*iter2)->clone());
			mode_phi_k.push_back((*iter2)->clone());
		}
		post_mean_phi.push_back(mean_phi_k);
		post_mode_phi.push_back(mode_phi_k);
		
	}
	
	//set the parameters
	if(!phi.empty())
		HawkesProcess::setPosteriorParams(post_mean_mu,post_mean_phi,post_mode_mu,post_mode_phi,mu_posterior_samples,phi_posterior_samples);
	else
		HawkesProcess::setPosteriorParams(post_mean_mu,post_mode_mu,mu_posterior_samples);
}

void HawkesProcess::setPosteriorParents(){
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
		setPosteriorParents(*seq_iter, post_mode_mu, post_mode_phi);
}

void HawkesProcess::setPosteriorParents(EventSequence &seq){
	setPosteriorParents(seq, post_mode_mu, post_mode_phi);
}

void HawkesProcess::setPosteriorParents(EventSequence & seq, const std::vector<ConstantKernel*> & post_mode_mu,   const std::vector<std::vector<Kernel *>> & post_mode_phi){

	std::vector<Event *> seq_v;
	map_to_vector(seq.full,seq_v);
	
	for(auto iter=std::next(seq.full.begin());iter!=seq.full.end();iter++){


		Event *e=iter->second;
		std::vector<double> parent_prob;
     	double norm_constant=0.0;
	   //probability that the event belongs to the background process
     	if(e->type>=(int)post_mode_mu.size()){
     		std::cerr<<"invalid event type\n";
     	}
		parent_prob.push_back(post_mode_mu[e->type]->p[0]);//TODO: check how the types are stored from 0 for the type of the virtual event!
	    norm_constant+=parent_prob.back();

	    //start parsing the history of events, compute the probability that the event was trigerred by another event
	    if(!post_mode_phi.empty()){
			for(auto iter2=std::next(seq.full.begin());iter2!=iter;iter2++){

				Event * e2=iter2->second;
				if(!e2)
					std::cerr<<"invalid event\n";
				if(e2->type>=(int)post_mode_phi.size()){
					std::cerr<<"invalid event type\n";
				}
				parent_prob.push_back(post_mode_phi[e2->type][e->type]->compute(e->time,e2->time));
				norm_constant+=parent_prob.back();
			}

	    }

	   std::vector<double>::iterator result = std::max_element(std::begin(parent_prob), std::end(parent_prob));
	   unsigned int parent=std::distance(std::begin(parent_prob), result);
	   seq.changeParent(e,seq_v[parent]);

	}
}

void HawkesProcess::flushBurninSamples(unsigned int nof_burnin){

	for(long unsigned int k=0;k<mu.size();k++){
		for(unsigned int p=0;p<mu[k]->nofp;p++){
			for(long unsigned int r=0;r<mu_posterior_samples[k][p].size();r++){
				boost::numeric::ublas::vector<double> u=boost::numeric::ublas::subrange(mu_posterior_samples[k][p][r], nof_burnin,mu_posterior_samples[k][p][r].size());
				mu_posterior_samples[k][p][r].clear();//<-here to get the subrange
				mu_posterior_samples[k][p][r]=u;
			}
		}
	}

	
	for(unsigned int k=0;k<phi.size();k++){
		for(unsigned int k2=0;k2<phi[k].size();k2++){
			for(unsigned int p=0;p<phi[k][k2]->nofp;p++){
				for(long unsigned int r=0;r<phi_posterior_samples[k][k2][p].size();r++){
					boost::numeric::ublas::vector<double> u=boost::numeric::ublas::subrange(phi_posterior_samples[k][k2][p][r], nof_burnin,phi_posterior_samples[k][k2][p][r].size());
					phi_posterior_samples[k][k2][p][r].clear();
					phi_posterior_samples[k][k2][p][r]=u;
				}
			}
		}
	}
}


/****************************************************************************************************************************************************************************
 * Model Learning Plot Methods
******************************************************************************************************************************************************************************/

void HawkesProcess::generateMCMCplots(unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool write_png_file, bool write_csv_file) {
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
	generateMCMCplots_(dir_path_str,samples_step,true_values, burnin_num_iter, write_png_file, write_csv_file);
}


void HawkesProcess::generateMCMCplots(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool write_png_file, bool write_csv_file){
	if(!write_png_file && !write_csv_file)
		return;
	generateMCMCplots_(dir_path_str,samples_step,true_values, burnin_num_iter, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCplots_(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool write_png_file, bool write_csv_file){
	if(!write_png_file && !write_csv_file)
		return;
//	generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
	//generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);
	//generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);
    generateMCMCTrainLikelihoodPlots(samples_step, true_values,write_png_file, write_csv_file, true,dir_path_str);
	flushBurninSamples(burnin_num_iter);
	//generateMCMCPosteriorPlots_(dir_path_str,true_values,write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCTracePlots(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCTracePlots(bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCTracePlots_(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
	for(long unsigned int k=0;k<mu.size();k++){
		unsigned int nofp=mu[k]->nofp;
		for(unsigned int p=0;p<nofp;p++){
			//plot the trace of the base intensity
			std::string filename=dir_path_str+"trace_lambda_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);
			plotTraces(filename,mu_posterior_samples[k][p], write_png_file,write_csv_file);
		}
	}
	
	for(long unsigned int k=0;k<phi.size();k++){
		for(long unsigned int k2=0;k2<phi[k].size();k2++){
			//plot the trace of the kernel parameters
			unsigned int nofp=phi[k][k2]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"trace_phi_params_type"+std::to_string(k)+"_"+std::to_string(k2)+"_param_"+std::to_string(p);
				plotTraces(filename,phi_posterior_samples[k][k2][p], write_png_file,write_csv_file);
			}
		}
	}
}

void HawkesProcess::generateMCMCMeanPlots(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);
}


void HawkesProcess::generateMCMCMeanPlots(bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);

}

void HawkesProcess::generateMCMCMeanPlots_(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
   
	for(long unsigned int k=0;k<mu.size();k++){
		unsigned int nofp=mu[k]->nofp;
		for(unsigned int p=0;p<nofp;p++){
			//plot the mean of the base intensity
			std::string filename=dir_path_str+"meanplot_lambda_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);
			plotMeans(filename,mu_posterior_samples[k][p], write_png_file, write_csv_file);
		}
	}
	
	for(long unsigned int k=0;k<phi.size();k++){
		for(long unsigned int k2=0;k2<phi[k].size();k2++){
			unsigned int nofp=phi[k][k2]->nofp;
			//plot the mean of the kernel parameters
			for(unsigned int p=0;p<nofp;p++){
				//plot the mean of the kernel parameter
				std::string filename=dir_path_str+"meanplot_phi_params_type"+std::to_string(k)+"_"+std::to_string(k2)+"_param_"+std::to_string(p);
				plotMeans(filename,phi_posterior_samples[k][k2][p],write_png_file, write_csv_file);
			}
		}
	}
	
}

void HawkesProcess::generateMCMCAutocorrelationPlots(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCAutocorrelationPlots(bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);

}
void HawkesProcess::generateMCMCAutocorrelationPlots_(const std::string & dir_path_str, bool write_png_file,  bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
	for(long unsigned int k=0;k<mu.size();k++){
		unsigned int nofp=mu[k]->nofp;
		for(unsigned int p=0;p<nofp;p++){
			//plot the autocorrelation for the base intensity
			std::string filename=dir_path_str+"autocorrelation_lambda_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);
			plotAutocorrelations(filename,mu_posterior_samples[k][p], write_png_file, write_csv_file);
		}
	}	
	
	for(long unsigned int k=0;k<phi.size();k++){
		for(long unsigned int k2=0;k2<phi[k].size();k2++){
			unsigned int nofp=phi[k][k2]->nofp;
			//plot the trace of the kernel parameters
			for(unsigned int p=0;p<nofp;p++){
				//plot the autocorrelation for the kernel parameter
				std::string filename=dir_path_str+"autocorrelation_phi_params_type"+std::to_string(k)+"_"+std::to_string(k2)+"_param_"+std::to_string(p);
				plotAutocorrelations(filename,phi_posterior_samples[k][k2][p], write_png_file, write_csv_file);
			}
		}
	}	
}

void HawkesProcess::generateMCMCPosteriorPlots(const std::string & dir_path_str, bool true_values, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
    generateMCMCPosteriorPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCPosteriorPlots(bool true_values, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCPosteriorPlots_(dir_path_str,true_values, write_png_file, write_csv_file);

}

void  HawkesProcess::generateMCMCPosteriorPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file,  bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
			return;
	for(long unsigned int k=0;k<mu.size();k++){
		unsigned int nofp=mu[k]->nofp;
		std::vector<double> true_mu_v;
		if(true_values && !mu.empty())
			mu[k]->getParams(true_mu_v);
		for(unsigned int p=0;p<nofp;p++){
			//merge the samples across all mcmc runs
			boost::numeric::ublas::vector<double> mu_posterior_samples_all=merge_ublasv(mu_posterior_samples[k][p]);
			//plot the posterior of the base intensity for each run
			std::string filename=dir_path_str+"posterior_lambda_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);
			plotDistributions(filename,mu_posterior_samples[k][p], write_png_file, write_csv_file);
			filename.clear();
			//plot the posterior of the base intensity merging the samples of different runs
			filename=dir_path_str+"all_posterior_lambda_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);

			if(true_values)
				plotDistribution(filename,mu_posterior_samples_all,true_mu_v[p], write_png_file, write_csv_file);//TODO: for real-world data it's not known. make it have a flag for plotting the real value from the kernel
			else
				plotDistribution(filename,mu_posterior_samples_all,write_png_file, write_csv_file);
		}
	}
	
	for(long unsigned int k=0;k<phi.size();k++){
		for(long unsigned int k2=0;k2<phi[k].size();k2++){
			//plot the trace of the kernel parameters
			unsigned int nofp=phi[k][k2]->nofp;
			std::vector<double> true_phi_v;
			if(true_values && !phi.empty() && !phi[k].empty())
				phi[k][k2]->getParams(true_phi_v);
			for(unsigned int p=0;p<nofp;p++){
				//merge the samples across all mcmc runs
				boost::numeric::ublas::vector<double> phi_posterior_samples_all=merge_ublasv(phi_posterior_samples[k][k2][p]);
				//plot the posterior of the kernel parameter
				std::string filename=dir_path_str+"posterior_phi_params_type_"+std::to_string(k)+"_"+std::to_string(k2)+"_param_"+std::to_string(p);
				plotDistributions(filename,phi_posterior_samples[k][k2][p], write_png_file, write_csv_file);
				filename.clear();
				//plot the posterior of the kernel parameters merging the samples of different runs
				filename=dir_path_str+"all_posterior_phi_params_type_"+std::to_string(k)+"_"+std::to_string(k2)+"_param_"+std::to_string(p);

				if(true_values)
					plotDistribution(filename,phi_posterior_samples_all,true_phi_v[p],write_png_file, write_csv_file);
				else
					plotDistribution(filename,phi_posterior_samples_all,write_png_file, write_csv_file);
			}
		}
	}	
}

void HawkesProcess::generateMCMCIntensityPlots(const std::string & dir_path_str, bool true_values, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
    generateMCMCIntensityPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCIntensityPlots(bool true_values,  bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCIntensityPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void HawkesProcess::generateMCMCIntensityPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
	for(unsigned int k=0;k<K;k++){
		for(long unsigned int seq_id=0;seq_id!=train_data.size();seq_id++)
			plotPosteriorIntensity(dir_path_str,k,train_data[seq_id],true_values,seq_id, NOF_PLOT_POINTS,write_png_file, write_csv_file);
	}
}

void HawkesProcess::plotPosteriorIntensity(const std::string & dir_path_str, unsigned int k, const EventSequence &s,  bool true_intensity, unsigned int sequence_id, unsigned int nofps, bool write_png_file,  bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
			return;
	//compute the means and modes of the kernel parameters
	unsigned int nofp=mu[k]->nofp;
	std::vector<double> mean_mu_k_param;//it keeps the posterior mean estimation of the base intensity of type k
	std::vector<double> mode_mu_k_param;//it keeps the posterior mode estimation of the base intensity of type k
	for(unsigned int p=0;p<nofp;p++){
		boost::numeric::ublas::vector<double> mu_posterior_samples_all=merge_ublasv(mu_posterior_samples[k][p]);
		double mean_mu_k_p_param=sample_mean(mu_posterior_samples_all);
		mean_mu_k_param.push_back(mean_mu_k_p_param);
		double mode_mu_k_p_param=sample_mode(mu_posterior_samples_all);
		mode_mu_k_param.push_back(mode_mu_k_p_param);
	}
	std::vector<std::vector<double>> mean_phi_k_param;//it keeps the posterior mean of the parameters of the triggering kernel functions of other types to type k
	std::vector<std::vector<double>> mode_phi_k_param;//it keeps the posterior mode of the parameters of the triggering kernel functions of other types to type k
	for(long unsigned int k2=0;k2<phi.size();k2++){
		std::vector<double> mean_phi_k2_k_param;
		std::vector<double> mode_phi_k2_k_param;
		unsigned int nofp=phi[k2][k]->nofp;
		for(unsigned int p=0;p<nofp;p++){
				boost::numeric::ublas::vector<double> phi_posterior_samples_all=merge_ublasv(phi_posterior_samples[k2][k][p]);
				double mean_phi_k2_k_p_param=sample_mean(phi_posterior_samples_all);
				mean_phi_k2_k_param.push_back(mean_phi_k2_k_p_param);
				double mode_phi_k2_k_p_param=sample_mode(phi_posterior_samples_all);
				mode_phi_k2_k_param.push_back(mode_phi_k2_k_p_param<0?EPS:mode_phi_k2_k_p_param);//negative due to the gaussian kernel density estimation, no posterior sample of mcmc was actually negative
		}
		mean_phi_k_param.push_back(mean_phi_k2_k_param);
		mode_phi_k_param.push_back(mode_phi_k2_k_param);
	}

	
	//set kernels to posterior mode estimations
	ConstantKernel *mode_mu_k=(mu[k]->clone());//TODO: does mode_mu change mu??
	mode_mu_k->setParams(mode_mu_k_param);
	std::vector<Kernel*> mode_phi_k(phi.size());
	for(long unsigned int k2=0;k2<phi.size();k2++){
		mode_phi_k[k2]=phi[k2][k]->clone();
		mode_phi_k[k2]->setParams(mode_phi_k_param[k2]);
	}
	boost::numeric::ublas::vector<double>  mode_tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  mode_lv;//the value of the intensity function
	computeIntensity(mode_mu_k,mode_phi_k,s,mode_tv,mode_lv,nofps);
	

	//create plot terminal and specifications for plotting the posterior mode intensity
	if(write_png_file){
		Gnuplot mode_gp;
		std::vector<std::string> mode_plot_specs;
		mode_plot_specs.push_back("with lines lw 2 title 'Mode Type "+std::to_string(k)+"'");
		std::string mode_filename=dir_path_str+"posterior_mode_intensity_type_"+std::to_string(k)+".png";
		create_gnuplot_script(mode_gp,mode_filename,mode_plot_specs,start_t,end_t);

		mode_gp.send1d(boost::make_tuple(mode_tv,mode_lv));
	}
	if(write_csv_file){
		std::string filename=dir_path_str+"posterior_mode_intensity_type_"+std::to_string(k)+".csv";
		std::ofstream file{filename};
		file<<"time,value"<<std::endl;
		for(long unsigned int i=0;i<mode_tv.size();i++){
			file<<mode_tv[i]<<","<<mode_lv[i]<<std::endl;
		}
		file.close();
	}

	
	//set kernels to posterior mean estimations
	ConstantKernel *mean_mu_k=(mu[k]->clone());//TODO: does mode_mu change mu??
	mean_mu_k->setParams(mean_mu_k_param);
	std::vector<Kernel*> mean_phi_k(phi.size());
	for(long unsigned int k2=0;k2<phi.size();k2++){
		mean_phi_k[k2]=phi[k2][k]->clone();
		mean_phi_k[k2]->setParams(mean_phi_k_param[k2]);
	}
	boost::numeric::ublas::vector<double>  mean_tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  mean_lv;//the value of the intensity function
	computeIntensity(mean_mu_k,mean_phi_k,s,mean_tv,mean_lv,nofps);
	
	

	//create plot terminal and specifications for plotting the posterior mean intensity
	if(write_png_file){
		Gnuplot mean_gp;
		std::vector<std::string> mean_plot_specs;
		mean_plot_specs.push_back("with lines lw 2 title 'Mode Type "+std::to_string(k)+"'");
		std::string mean_filename=dir_path_str+"posterior_mean_intensity_type_"+std::to_string(k)+".png";
		create_gnuplot_script(mean_gp,mean_filename,mean_plot_specs,start_t,end_t);

		mean_gp.send1d(boost::make_tuple(mean_tv,mean_lv));
	}
	if(write_csv_file){
		std::string filename=dir_path_str+"posterior_mean_intensity_"+std::to_string(sequence_id)+"_type_"+std::to_string(k)+".csv";
		std::ofstream file{filename};
		file<<"time,value"<<std::endl;
		for(long unsigned int i=0;i<mean_tv.size();i++){
			file<<mean_tv[i]<<","<<mean_lv[i]<<std::endl;
		}
		file.close();
	}

	//if the true kernel parameters are known, compute the squared error of the estimated intensities
	if(true_intensity){
		boost::numeric::ublas::vector<double>  true_tv;//time points for which the intensity function will be computed
		boost::numeric::ublas::vector<long double>  true_lv;//the value of the intensity function
		std::vector<Kernel *> true_phi_k(phi.size());
		for(long unsigned int k2=0;k2<phi.size();k2++)
			true_phi_k[k2]=phi[k2][k];
		computeIntensity(mu[k],true_phi_k,s,true_tv,true_lv,nofps);

		//compute the absolute relative error of the posterior-mode intensity function
		boost::numeric::ublas::vector<double> error_mode_lv=true_lv-mode_lv;
				boost::numeric::ublas::vector<double> rel_error_mode_lv=element_div(error_mode_lv,true_lv);
				error_mode_lv.clear();
				std::transform(rel_error_mode_lv.begin(),rel_error_mode_lv.end(),rel_error_mode_lv.begin(),[&](double x){
					return std::abs(x);
				}
				);
		if(write_png_file){
			Gnuplot error_mode_gp;
			std::vector<std::string> error_mode_plot_specs;
			error_mode_plot_specs.push_back("with lines lw 2 title 'Mode Type "+std::to_string(k)+"'");
			std::string error_mode_filename=dir_path_str+"mode_posterior_error_intensity_type_"+std::to_string(k)+".png";
			create_gnuplot_script(error_mode_gp,error_mode_filename,error_mode_plot_specs,start_t,end_t);
			
			error_mode_gp.send1d(boost::make_tuple(mode_tv,rel_error_mode_lv));
		}

		if(write_csv_file){
			std::string filename=dir_path_str+"mode_posterior_error_intensity_type_"+std::to_string(k)+".csv";
			std::ofstream file{filename};
			file<<"time,value"<<std::endl;
			for(long unsigned int i=0;i<mode_tv.size();i++){
				file<<mode_tv[i]<<","<<rel_error_mode_lv[i]<<std::endl;
			}
			file.close();
		}
		
		//compute the absolute relative error of the posterior-mean intensity function
		boost::numeric::ublas::vector<double> error_mean_lv=true_lv-mean_lv;
		boost::numeric::ublas::vector<double> rel_error_mean_lv=element_div(error_mean_lv,true_lv);
		std::transform(rel_error_mean_lv.begin(),rel_error_mean_lv.end(),rel_error_mean_lv.begin(),[&](double x){
			return std::abs(x);
		}
		);

		//compute the absolute relative error of the posterior-mean intensity function
		if(write_png_file){
			Gnuplot error_mean_gp;
			std::vector<std::string> error_mean_plot_specs;
			error_mean_plot_specs.push_back("with lines lw 2 title 'Mean Type "+std::to_string(k)+"'");
			std::string error_mean_filename=dir_path_str+"mean_posterior_error_intensity_type_"+std::to_string(k)+".png";
			create_gnuplot_script(error_mean_gp,error_mean_filename,error_mean_plot_specs,start_t,end_t);
	
			error_mean_gp.send1d(boost::make_tuple(mean_tv,rel_error_mean_lv));
		}

		if(write_csv_file){
			std::string filename=dir_path_str+"mean_posterior_error_intensity_type_"+std::to_string(k)+".csv";
			std::ofstream file{filename};
			file<<"time,value"<<std::endl;
			for(long unsigned int i=0;i<mean_tv.size();i++){
				file<<mean_tv[i]<<","<<rel_error_mean_lv[i]<<std::endl;
			}
			file.close();
		}
	}
}


void  HawkesProcess::generateMCMCTrainLikelihoodPlots(unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file, bool normalized, const std::string & dir_path_str) const {
	if(!write_png_file && !write_csv_file)
		return;

	std::string plot_dir_path_str=dir_path_str.empty()?createModelInferenceDirectory():dir_path_str;
	for(long unsigned int seq_id=0;seq_id!=train_data.size();seq_id++){
		generateMCMCLikelihoodPlot(plot_dir_path_str,samples_step, true_values, train_data[seq_id], write_png_file, write_csv_file, normalized);
	}

}
void  HawkesProcess::generateMCMCTestLikelihoodPlots(const std::string &seq_dir_path_str, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file, bool normalized,  double t0, double t1, const std::string & dir_path_str) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string plot_dir_path_str=dir_path_str.empty()?createModelInferenceDirectory():dir_path_str;
	for(long unsigned int seq_id=0;seq_id!=nof_test_sequences;seq_id++){
		
		//open sequence file
		std::string csv_seq_filename{seq_dir_path_str+seq_prefix+std::to_string(seq_id)+".csv"};
		std::ifstream test_seq_file{csv_seq_filename};
		EventSequence test_seq;//{seq_prefix+std::to_string(n)+".csv"};

		
		test_seq.type.resize(K+1);
		test_seq.K=K+1;//the virtual event which corresponds to the background process has an additional type
		test_seq.N=0;
		test_seq.Nt=0;
		test_seq.thinned_type.resize(K+1);

		//load the sequence from csv file
		test_seq.load( test_seq_file,1);
	

		//add virtual event if it doesn't exist
		Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
		test_seq.addEvent(ve,0);
		test_seq.start_t=t0>=0?t0:start_t;//to compute the likelihood in the specificied interval
		double last_arrival_time=test_seq.full.rbegin()->first;
		test_seq.name=seq_prefix+std::to_string(seq_id);
		test_seq.end_t=(t1>test_seq.start_t)?t1:last_arrival_time;
	
		test_seq_file.close();
		
		generateMCMCLikelihoodPlot(plot_dir_path_str,samples_step, true_values, test_seq, write_png_file, write_csv_file, normalized);
	}

}



void HawkesProcess::generateMCMCLikelihoodPlot(const std::string & dir_path_str, unsigned int samples_step, bool true_values, const EventSequence & seq, bool write_png_file,  bool write_csv_file, bool normalized) const{
	if(!write_png_file && !write_csv_file)
			return;
	//compute the likelihood with the true parameters
	double true_logl;
	if(true_values){
		if(!phi.empty())
			true_logl=loglikelihood(seq,mu,phi);
		else
			true_logl=loglikelihood(seq,mu);
	}

	long unsigned int mcmc_samples=mu_posterior_samples[0][0][0].size();//todo: check mu_posterior_samples has at least one dimension
	//vectors which will hold the likelihood and the loglikelihood with the posterior mean estimates
	std::vector<double> post_mean_loglikelihood(mcmc_samples/samples_step);

	//vectors which will hold the likelihood and the loglikelihood with the posterior mode estimates
	std::vector<double> post_mode_loglikelihood(mcmc_samples/samples_step);


	bool endof_samples=false;
	unsigned int nof_samples=samples_step;//TODO: fix it, it should be determined dynamically!!!!!

	//split the computation of successive (with respect to the number of samples) likelihoods in threads
	pthread_t plot_threads[PLOT_NOF_THREADS];
	pthread_mutex_t *mtx=(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(mtx,NULL);
	for(unsigned int thread_id=0;thread_id<PLOT_NOF_THREADS;thread_id++){	
			generateMCMCLikelihoodPlotThreadParams* p;
			p= new generateMCMCLikelihoodPlotThreadParams(this, seq, &nof_samples, &endof_samples,samples_step,post_mean_loglikelihood, post_mode_loglikelihood,mtx, normalized);
			int rc = pthread_create(&plot_threads[thread_id], NULL, generateMCMCLikelihoodPlots__, (void *)p);
			if (rc){
				 ////std::cerr << "Error:unable to create thread," << rc << std::endl;
				 exit(-1);
			}
	}
	//wait for all the loglikelihoods to finish
	for (unsigned int t = 0; t <PLOT_NOF_THREADS; t++){
		pthread_join (plot_threads [t], NULL);
	}

	//plot the loglikelihoods
	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines lw 2 title 'Posterior Mode LogLikelihood' ");
		plot_specs.push_back("with lines lw 2 title 'Posterior Mean LogLikelihood' ");
		if(true_values)
			plot_specs.push_back("with lines lw 2 title 'True LogLikelihood' ");
		std::string filename=dir_path_str+"loglikelihood_"+seq.name+".png";
		create_gnuplot_script(gp,filename,plot_specs,"x"+std::to_string(samples_step)+" samples","Loglikelihood");
		std::vector<double> iter(post_mode_loglikelihood.size());
		std::iota( std::begin(iter), std::end(iter),1);
		gp.send1d(boost::make_tuple(iter,post_mode_loglikelihood));
		gp.send1d(boost::make_tuple(iter,post_mean_loglikelihood));
		std::vector<double> true_loglikelihood(post_mode_loglikelihood.size(),true_logl);
		if(true_values)
			gp.send1d(boost::make_tuple(iter,true_loglikelihood));
	}
	if(write_csv_file){
		std::string filename=dir_path_str+"loglikelihood_"+seq.name+".csv";
		std::ofstream file{filename};
	
		if(true_values)
			file<<"mode likelihood,mean likelihood, true likelihood"<<std::endl;
		else
			file<<"mode likelihood,mean likelihood"<<std::endl;
		for(long unsigned int i=0;i<post_mode_loglikelihood.size();i++){
			if(true_values)
				file<<post_mode_loglikelihood[i]<<","<<post_mean_loglikelihood[i]<<","<<true_logl<<std::endl;
			else
				file<<post_mode_loglikelihood[i]<<","<<post_mean_loglikelihood[i]<<std::endl;
		}
		file.close();
	}
}

void *HawkesProcess::generateMCMCLikelihoodPlots__(void *p){
	
	
	//unwrap thread parameters
	std::unique_ptr<generateMCMCLikelihoodPlotThreadParams> params(static_cast<generateMCMCLikelihoodPlotThreadParams * >(p));
	const HawkesProcess *hp=params->hp;
	const EventSequence &seq=params->seq;
	unsigned int *nof_samples=params->nof_samples;
	bool *endof_samples=params->endof_samples;
	unsigned int samples_step=params->samples_step;

	std::vector<double> & post_mean_loglikelihood=params->post_mean_loglikelihood;
	std::vector<double> & post_mode_loglikelihood=params->post_mode_loglikelihood;
	pthread_mutex_t *mtx=params->mtx;

	bool normalized=params->normalized;
	
	//get number of parameters for each component of the model

	
	//initialize the auxiliary kernels that will hold the current point estimates
	std::vector<ConstantKernel*>  post_mean_mu;
	std::vector<std::vector<Kernel *>>  post_mean_phi;
	
	std::vector<ConstantKernel*>  post_mode_mu;
	std::vector<std::vector<Kernel *>>  post_mode_phi;

	pthread_mutex_lock(mtx);
	//EventSequence data_copy=hp->data;
	unsigned int mu_nofp=hp->mu[0]->nofp;//number of parameters for the base intensity
	long unsigned int runs=hp->mu_posterior_samples[0][0].size(); //number of mcmc runs
	unsigned int phi_nofp=0;
	
	if(!hp->phi.empty())
		phi_nofp=hp->phi[0][0]->nofp;//number of parameters for the mutually triggerred intensity
	
	for(long unsigned int k=0;k<hp->mu.size();k++){
		post_mean_mu.push_back((hp->mu[k]->clone()));//TODO: maybe should I clone it?????
		post_mode_mu.push_back((hp->mu[k]->clone()));
	}
	
	for(long unsigned int k=0;k<hp->phi.size();k++){
		std::vector<Kernel*> mean_phi_k(hp->phi[k].size());
		std::vector<Kernel*> mode_phi_k(hp->phi[k].size());

		for(long unsigned int k2=0;k2<hp->phi[k].size();k2++){
			mean_phi_k[k2]=hp->phi[k][k2]->clone();
			mode_phi_k[k2]=hp->phi[k][k2]->clone();
		}
		post_mean_phi.push_back(mean_phi_k);
		post_mode_phi.push_back(mode_phi_k);
	}
	pthread_mutex_unlock(mtx);
	pthread_mutex_lock(mtx);
	//unsigned int logl_tasks=0;
	while(!*endof_samples){
		//read the end of the next batch of posterior samples
		//update the range of samples to be used
		

		unsigned int nof_samples_t=*nof_samples;
		unsigned int l=nof_samples_t/samples_step-1;
		*nof_samples+=samples_step;//range of samples to be executed by next thread
		
		//check for the end of samples
		if(l>=post_mean_loglikelihood.size())
			*endof_samples=true;

		if(*endof_samples)
			break;
			

		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> mu_samples(hp->mu.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(mu_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs)));
		std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> phi_samples(hp->phi.size(),std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>(hp->phi.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(phi_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs))));
	
		//keep only the first samples  of each mcmc run for each param
		for(long unsigned int k=0;k<hp->mu.size();k++){
			for(unsigned int p=0;p<mu_nofp;p++){
				for(unsigned int r=0;r<runs;r++){
					mu_samples[k][p][r]=subrange(hp->mu_posterior_samples[k][p][r],0,nof_samples_t);//todo: maybe these memorie copies cost that much????

				}
			}
		}
		
		for(long unsigned int k=0;k<hp->phi.size();k++){
			for(long unsigned int k2=0;k2<hp->phi[k].size();k2++){
				for(unsigned int p=0;p<phi_nofp;p++){
					for(unsigned int r=0;r<runs;r++){
						phi_samples[k][k2][p][r]=subrange(hp->phi_posterior_samples[k][k2][p][r],0,nof_samples_t);

					}
				}
			}
		}
		
		pthread_mutex_unlock(mtx);

		//compute and set the kernels to the new posterior point estimates
		if(!hp->phi.empty()){
			HawkesProcess::setPosteriorParams(post_mean_mu,post_mean_phi, post_mode_mu,post_mode_phi, mu_samples,phi_samples);
			//compute the likelihood with the posterior mean estimates		
			double logl=loglikelihood(seq,post_mean_mu,post_mean_phi, normalized);
			post_mean_loglikelihood[l]=logl;
			//compute the likelihood with the posterior mode estimates
			logl=loglikelihood(seq,post_mode_mu,post_mode_phi, normalized);
			post_mode_loglikelihood[l]=logl;

		}
		else{
			//EventSequence data_copy=hp->data;
			HawkesProcess::setPosteriorParams(post_mean_mu, post_mode_mu, mu_samples);
			//compute the likelihood with the posterior mean estimates
			double logl=loglikelihood(seq,post_mean_mu, normalized);
			post_mean_loglikelihood[l]=logl;
			//compute the likelihood with the posterior mode estimates
			logl=loglikelihood(seq,post_mode_mu, normalized);
			post_mode_loglikelihood[l]=logl;
		}
		pthread_mutex_lock(mtx);
	}
	pthread_mutex_unlock(mtx);
	return 0;
}


/****************************************************************************************************************************************************************************
 * Model Likelihood Methods
******************************************************************************************************************************************************************************/
//static methods

double HawkesProcess::likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0){
	return seq.observed ? fullLikelihood(seq, mu, phi, normalized, t0) : partialLikelihood(seq, mu, phi, normalized, t0);
}

double HawkesProcess::likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized,  double t0){
	return seq.observed ? fullLikelihood(seq,mu, normalized, t0) : partialLikelihood(seq,mu, normalized, t0) ;
}

//it computes the part of the likelihood of the seq which involves point processes generated by events of type k of type k2
double HawkesProcess::likelihood(const EventSequence & seq, const Kernel & phi, int k, int k2,  double t0) {
	double l=1.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	if(seq.type.empty()||seq.type[k].empty()){
		return 0.0;
	}
	for(auto e_iter=seq.type[k].begin();e_iter!=seq.type[k].end();e_iter++){//start parsing the events of type k
		double t_n= e_iter->second->time;

		l*=exp(-phi.computeIntegral(t0>t_n?t0:t_n, seq.end_t,t_n));//this corresponds to the exp(-\int(lambda(t))) term in the likelihood
		if(e_iter->second->offsprings){
			for(auto o_iter=e_iter->second->offsprings->type[k2].begin();o_iter!=e_iter->second->offsprings->type[k2].end();o_iter++){
				if(o_iter->second->time>t0 && o_iter->second->time<seq.end_t){
					double hl=phi.compute(o_iter->second->time,t_n);//this corresponds to the product of intensities for the events of type k2 triggered by the event of type k
					l*=hl;
				}
			}
			for(auto o_iter=e_iter->second->offsprings->thinned_type[k2].begin();o_iter!=e_iter->second->offsprings->thinned_type[k2].end();o_iter++){
				if(o_iter->second->time>t0 && o_iter->second->time<seq.end_t)
					l*=phi.compute(o_iter->second->time,t_n);//this corresponds to the product of intensities for the events of type k2 triggered by the event of type k
			}
		}
	}

	return l;
}

double HawkesProcess::loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0) {
	return seq.observed?fullLoglikelihood(seq, mu, phi, normalized, t0):partialLoglikelihood(seq, mu, phi, normalized, t0);
}


double HawkesProcess::loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized,  double t0){
	return seq.observed ? fullLoglikelihood(seq,mu, normalized, t0):partialLoglikelihood(seq,mu, normalized, t0);
}


double HawkesProcess::loglikelihood(const EventSequence & seq, const Kernel & phi, int k, int k2,  double t0){

	double l=0.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	if(seq.type.empty()||seq.type[k].empty()){
		return -1.0;
	}
	for(auto e_iter=seq.type[k].begin();e_iter!=seq.type[k].end();e_iter++){
		double t_n= e_iter->second->time;
		l-=phi.computeIntegral(t0>t_n?t0:t_n, seq.end_t,t_n);
		if(e_iter->second->offsprings){
			for(auto o_iter=e_iter->second->offsprings->type[k2].begin();o_iter!=e_iter->second->offsprings->type[k2].end();o_iter++){
				if(o_iter->second->time>t0 && o_iter->second->time<seq.end_t)
					l+=log(phi.compute(o_iter->second->time,t_n));
			}
			for(auto o_iter=e_iter->second->offsprings->thinned_type[k2].begin();o_iter!=e_iter->second->offsprings->thinned_type[k2].end();o_iter++){
				if(o_iter->second->time>t0 && o_iter->second->time<seq.end_t)
					l+=log(phi.compute(o_iter->second->time,t_n));
			}
		}
	}
	return l;
}
//----- static methods (useful for the mcmc updates from the kernel classes)
//TODO: check for empty offsprings as below etc,,,,
//it computes the full likelihood of the seq
double HawkesProcess::fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0){
	Event *ve=seq.full.begin()->second; //the virtual event which corresponds to the background process
	double l=1.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	long unsigned int K=mu.size();//number of event types
	unsigned int nof_events=0;
	for(unsigned int k=0;k<K;k++){//each loop corresponds to the point processes for events of type k
		l*=exp(-mu[k]->computeIntegral(t0>seq.start_t?t0:seq.start_t, seq.end_t, seq.start_t));//this term corresponds to the background process of type k
		for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){//exclude the virtual event
				int l_n=e_iter->second->type;
				double t_n= e_iter->second->time;
				l*=exp(-phi[l_n][k]->computeIntegral(t0>t_n?t0:t_n, seq.end_t,t_n));//this term corresponds to the point process of type k, that each event triggers
				if(l_n==(int)k && t_n>t0 && t_n<seq.end_t) {//this term corresponds to the product of the intensities for the point processes for events of type k in the likelihood
					nof_events++;
					Event *p_n=e_iter->second->parent;
					if(p_n==ve){
						l*=mu[l_n]->p[0];
					}
					else{
						l*=phi[p_n->type][k]->compute(t0>t_n?t0:t_n,p_n->time);
					}
				}
		}
	}
	//part of the likelihood for the thinned events
	for(auto e_iter=seq.thinned_full.begin();e_iter!=seq.thinned_full.end();e_iter++){
		int l_n=e_iter->second->type;
		double t_n= e_iter->second->time;
		Event *p_n=e_iter->second->parent;
		if(t_n>t0 && t_n<seq.end_t){
			nof_events++;
			if(p_n==ve)
				l*=mu[l_n]->p[0];
			else{
				l*=phi[p_n->type][l_n]->compute(t0>t_n?t0:t_n,p_n->time);
			}
		}
   }
	return normalized?l/nof_events:l;
}


double HawkesProcess::fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized,  double t0){
	
	Event *ve=seq.full.begin()->second; //the virtual event which corresponds to the background process
	double l=1.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	unsigned int nof_events=0;
	long unsigned int K=mu.size();//number of events
	for(unsigned int k=0;k<K;k++){//each loop corresponds to the point processes for events of type k
		l*=exp(-mu[k]->computeIntegral(t0>seq.start_t?t0:seq.start_t, seq.end_t, seq.start_t));//this term corresponds to the background process of type 
		if(t0>0){
			for(auto e_iter=std::next(seq.type[k].begin());e_iter!=seq.type[k].end();e_iter++){
				if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
					nof_events++;
					l*=mu[k]->p[0];
				}
			}
			for(auto e_iter=std::next(seq.thinned_type[k].begin());e_iter!=seq.thinned_type[k].end();e_iter++){
				if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
					nof_events++;
					l*=mu[k]->p[0];
				}
			}
				
		}
		else{
			l*=pow(mu[k]->p[0],ve->offsprings->type[k].size()+ve->offsprings->thinned_type[k].size());
		}			
	}
	return normalized?l/(nof_events):l;
}


double HawkesProcess::fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0) {
	
	Event *ve=seq.full.begin()->second; //the virtual event which corresponds to the background process
	double l=0.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	long unsigned int K=mu.size();//number of events
	unsigned int nof_events=0;
	for(unsigned int k=0;k<K;k++){//each loop corresponds to the point processes for events of type k, don't forget that there's a virtual event of an extra type
		if(!mu[k])
			std::cerr<<"invalid constant kernel function\n";
		double v1=(-mu[k]->computeIntegral(t0>seq.start_t?t0:seq.start_t, seq.end_t, seq.start_t));
		l+=v1;//this term corresponds to the background process of type k

		for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){//exclude the virtual event
			
			int l_n=e_iter->second->type;
			double t_n= e_iter->second->time;
			if(!phi[l_n][k])
				std::cerr<<"invalid constant kernel function\n";
			double v2=(-phi[l_n][k]->computeIntegral(t0>t_n?t0:t_n, seq.end_t,t_n));//each realized event generates a poisson process of events of type k

			l+=v2;//this term corresponds to the point process of type k, that each event triggers
			if(l_n==(int)k && t_n>t0 && t_n<seq.end_t){ //this term corresponds to the product of the intensities for the point processes for events of type k  in the likelihood
				Event *p_n=e_iter->second->parent;
				nof_events++;
				if(p_n==ve ){
					double v3=log(mu[l_n]->p[0]);
					l+=v3;

				}
				else{
					if(!p_n)
						std::cerr<<"undefined event parent. can't compute likelihood\n";
					else{
						double v3=log(phi[p_n->type][k]->compute(t_n,p_n->time));
						l+=v3;
					}
			    }
			}
		}
	}
	for(auto e_iter=seq.thinned_full.begin();e_iter!=seq.thinned_full.end();e_iter++){
		int l_n=e_iter->second->type;
		double t_n= e_iter->second->time;
		Event *p_n=e_iter->second->parent;
		if(t_n>t0 && t_n<seq.end_t){
			nof_events++;
			if(p_n==ve)
				l+=log(mu[l_n]->p[0]);
			else{
				if(!p_n)
					std::cerr<<"undefined event parent. can't compute likelihood\n";
				else
					l+=log(phi[p_n->type][l_n]->compute(t_n,p_n->time));
			}
		}
	}
	
	return normalized?l/nof_events:l;
	
}

double HawkesProcess::fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized,  double t0){

	Event *ve=seq.full.begin()->second; //the virtual event which corresponds to the background process
	double l=0.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;
	unsigned int nof_events=0;
	long unsigned int K=mu.size();//number of events
	for(unsigned int k=0;k<K;k++){//each loop corresponds to the point processes for events of type k
		l+=(-mu[k]->computeIntegral(t0>seq.start_t?t0:seq.start_t, seq.end_t, seq.start_t));//this term corresponds to the background process of type k
		if(t0>0){
			for(auto e_iter=std::next(seq.type[k].begin());e_iter!=seq.type[k].end();e_iter++){
				if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
					nof_events++;
					l+=log(mu[k]->p[0]);
				}
			}
			for(auto e_iter=std::next(seq.thinned_type[k].begin());e_iter!=seq.thinned_type[k].end();e_iter++){
				if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
					nof_events++;
					l+=log(mu[k]->p[0]);
				}
			}
		}
		else
			l+=log(mu[k]->p[0])*(ve->offsprings->type[k].size()+ve->offsprings->thinned_type[k].size());
	}

	return normalized?l/nof_events:l;
}

double HawkesProcess::partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0){

	//reverse the vector so that phi_k[i] refers to the intensity functions to (not from) type k
	long unsigned int K=phi.size();
	unsigned int nof_events=0;
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	for(unsigned int k=0;k<K;k++){
		for(unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];
		}
	}

	double l=0.0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
			l+=log(computeIntensity(mu[e_iter->second->type],phi_k[e_iter->second->type],seq, e_iter->second->time));
			nof_events++;
		}
	}

	//monte carlo integration
	
	auto f=[&](double t)->double{
			double s=0;
			for(unsigned int k=0;k<K;k++)
				s+=computeIntensity(mu[k],phi_k[k], seq, t); 
			
			return s;
	};
	double i=monte_carlo_integral(f, t0>seq.start_t?t0:seq.start_t, seq.end_t);


	return normalized?(l-i)/(nof_events):(l-i);

}

double  HawkesProcess::partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, bool normalized,  double t0){

	const std::vector<Kernel *> phi;
	long unsigned int K=mu.size();
	t0=(t0>seq.start_t)?t0:seq.start_t;
	unsigned int nof_events=0;
	double l=0.0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
			l+=log(computeIntensity(mu[e_iter->second->type],phi,seq, e_iter->second->time));
			nof_events++;
		}
	}

	//TODO: use monte carlo method
	//monte carlo integration
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> uniform_distribution(t0>seq.start_t?t0:seq.start_t,seq.end_t);
	double i=0.0;
	for(unsigned int s=0;s<MC_NOF_SAMPLES;s++){
		double t=uniform_distribution(gen);
		for(unsigned int k=0;k<K;k++)
			i+=computeIntensity(mu[k],phi,seq, t);
	}
	i*=(seq.end_t-t0>seq.start_t?t0:seq.start_t)/MC_NOF_SAMPLES;

	return normalized?(l-i)/(nof_events):(l-i);

}


double HawkesProcess::partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized,  double t0){

	//reverse the vector so that phi_k[i] refers to the intensity functions to (not from) type k
	long unsigned int K=phi.size();
	t0=(t0>seq.start_t)?t0:seq.start_t;
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	unsigned nof_events=0;
	for(unsigned int k=0;k<K;k++){
		for(unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];//TODO: this is not very good design, maybe I should have stored trigering kernels in reversed order in the first place??
		}
	}

	double l=1.0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
			l*=(computeIntensity(mu[e_iter->second->type],phi_k[e_iter->second->type],seq,e_iter->second->time));
			nof_events++;
		}
	}
	//monte carlo integration
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> uniform_distribution(t0>seq.start_t?t0:seq.start_t,seq.end_t);
	double i=0.0;
	for(unsigned int s=0;s<MC_NOF_SAMPLES;s++){
		double t=uniform_distribution(gen);
		for(unsigned int k=0;k<K;k++)
			i+=computeIntensity(mu[k],phi_k[k],seq, t);
	}
	i*=(seq.end_t-t0>seq.start_t?t0:seq.start_t)/MC_NOF_SAMPLES;

	return normalized?l*exp(-i)/(nof_events):(l*exp(-i));

}

double  HawkesProcess::partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, bool normalized,  double t0){

	//assign the events to the virtual event (of the exogenous intensity)
	EventSequence seq_copy=seq;
	Event *ve=seq_copy.full.begin()->second; //the virtual event which corresponds to the background process
	for(auto e_iter=std::next(seq_copy.full.begin());e_iter!=seq_copy.full.end();e_iter++){
		seq_copy.setParent(e_iter->second,ve);
	}
	return fullLikelihood(seq_copy, mu, normalized, t0);
}

//----- non-static methods
double HawkesProcess::likelihood(const EventSequence & seq, bool normalized,  double t0) const{
	if(!phi.empty())//check whether there exists mutual excitation part
		return HawkesProcess::likelihood(seq,mu,phi, normalized, t0); //todo: check here if the library is fully observed
	 return HawkesProcess::likelihood(seq,mu,normalized, t0);
}

//it computes the part of the likelihood of the seq which involves point processes generated from events of type k to type k2
double HawkesProcess::likelihood(const EventSequence & seq, int k, int k2,  double t0) const {
	return HawkesProcess::likelihood(seq, *phi[k][k2], k, k2, t0);
}

double HawkesProcess::loglikelihood(const EventSequence & seq,bool normalized,  double t0) const{
	if(!phi.empty())//check whether there exists mutual excitation part
		return HawkesProcess::loglikelihood(seq,mu,phi, normalized, t0);
	return HawkesProcess::loglikelihood(seq,mu, normalized, t0);
}

double HawkesProcess::loglikelihood(const EventSequence & seq, int k, int k2,  double t0) const{
	return HawkesProcess::loglikelihood(seq, *phi[k][k2], k, k2, t0);
}


void HawkesProcess::posteriorModeLoglikelihood(EventSequence & seq, std::vector<double> & logl, bool normalized,  double t0) {
	setPosteriorParams();
	
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
		logl.push_back(phi.empty()?HawkesProcess::loglikelihood(*seq_iter, post_mode_mu, normalized, t0):HawkesProcess::loglikelihood(seq, post_mode_mu,post_mode_phi, normalized, t0));
 
}
/****************************************************************************************************************************************************************************
 * Model Intensity Methods
******************************************************************************************************************************************************************************/



void HawkesProcess::printMatlabFunctionIntensity(const ConstantKernel * mu,const EventSequence &s, std::string & matlab_lambda_func){
	
	matlab_lambda_func.clear();
	matlab_lambda_func="lambda_"+s.name<+" = @(t) ";  //declare matlab anonymous function
	matlab_lambda_func+=std::to_string(mu->p[0]);	//write constant intensity
	matlab_lambda_func+="+0*t";//this is just to appease matlab
	matlab_lambda_func+=";"; //terminate matlab command
}

void HawkesProcess::printMatlabExpressionIntensity(const ConstantKernel * mu,const EventSequence &s, std::string & matlab_lambda){
	
	matlab_lambda.clear();
	matlab_lambda=std::to_string(mu->p[0]);	//write constant intensity
	matlab_lambda+="+0*t";//this is just to appease matlab
}

void HawkesProcess::printMatlabFunctionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, std::string & matlab_lambda_func){
	
	matlab_lambda_func.clear();
	matlab_lambda_func="lambda_"+s.name+" = @(t) ";//declare matlab anonymous function
	std::string matlab_expr;
	printMatlabExpressionIntensity(mu, phi,s, matlab_expr);
	matlab_lambda_func+=matlab_expr;
	matlab_lambda_func+=";";//terminate matlab command
}

void HawkesProcess::printMatlabExpressionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, std::string & matlab_lambda){
	
	matlab_lambda.clear();
	
	matlab_lambda=std::to_string(mu->p[0])+"+"; //write exogenous intensity

	//write mutually-trigerring intensity
	for(auto iter=std::next(s.full.begin());iter!=s.full.end();iter++){//skip the virtual event
		Kernel *phi_k=phi[iter->second->type];
		if(!phi_k)
			std::cerr<<"unitialized kernel"<<std::endl;
		std::string lambda_e;
		phi_k->printMatlabExpression(iter->second->time, lambda_e);
		matlab_lambda+=lambda_e;
		if(std::next(iter)!=s.full.end())
			matlab_lambda+="+";
	}
}


void HawkesProcess::printMatlabFunctionIntensities(const std::string & dir_path_str, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s){

	long unsigned int K=mu.size();
	for(unsigned int k=0;k!=K;k++){
		//get excitation effect on type k
		std::vector<Kernel *> phi_k;
		for(auto iter=phi.begin();iter!=phi.end();iter++){
			phi_k.push_back((*iter)[k]);
		}
		//get the matlab function expression for the intensity of type k
		std::string matlab_lambda_func;
		HawkesProcess::printMatlabFunctionIntensity(mu[k], phi_k,s, matlab_lambda_func);		
		//open matlab file
		std::string matlab_filename=dir_path_str+"/"+s.name+"_matlab_func_lambda_"+std::to_string(k)+".m";
		std::ofstream matlab_file;
		matlab_file.open(matlab_filename);
		matlab_file<<matlab_lambda_func;
		//close file
		matlab_file.close();
	}
}






void HawkesProcess::computeIntensity(const ConstantKernel *mu, const std::vector<Kernel*> & phi ,const EventSequence &s,  boost::numeric::ublas::vector<double> & time_v, boost::numeric::ublas::vector<long double> & lv,unsigned int nofps){

	//take nofps time-points within the observation window of the process
	double step=(s.end_t-s.start_t)/(nofps-1);

	boost::numeric::ublas::scalar_vector<double> start_time_v(nofps,s.start_t);
	boost::numeric::ublas::vector<double> step_time_v(nofps);
	std::iota(step_time_v.begin(),step_time_v.end(), 0.0);
	if(time_v.size()<nofps)
		time_v.resize(nofps);
	time_v=start_time_v+step_time_v*step;

	computeIntensity(mu,phi,s,const_cast<const boost::numeric::ublas::vector<double>&> (time_v),lv);

}

double HawkesProcess::computeIntensity(const ConstantKernel *mu, const std::vector<Kernel*> & phi ,const EventSequence &s, double t) {

	std::vector<double> mu_params;
	mu->getParams(mu_params);
	double lv=mu_params[0];

	//add the contribution of events of the multiple types that occured in the past.
	for(long unsigned int k=0;k<phi.size();k++){
		boost::numeric::ublas::vector<double> arrival_time_v;
		s.getArrivalTimes(k,arrival_time_v);
		if(arrival_time_v.empty()) {
			continue;
		}
		for(auto t_iter=arrival_time_v.begin();t_iter!=arrival_time_v.end() && *t_iter<t;t_iter++){
			lv+=phi[k]->compute(t-*t_iter);
		}
	}
	return lv;
}



double HawkesProcess::computeIntensity(const ConstantKernel *mu, const std::vector<Kernel*> & phi ,const EventSequence &s, double th, double t) {

	std::vector<double> mu_params;
	mu->getParams(mu_params);
	double lv=mu_params[0];

	//add the contribution of events of the multiple types that occured in the past.
	for(long unsigned int k=0;k<phi.size();k++){
		boost::numeric::ublas::vector<double> arrival_time_v;
		s.getArrivalTimes(k,arrival_time_v);
		if(arrival_time_v.empty()) {
			continue;
		}
		for(auto t_iter=arrival_time_v.begin();t_iter!=arrival_time_v.end() && *t_iter<th;t_iter++){
			lv+=phi[k]->compute(t-*t_iter);
		}
	}
	return lv;
}


//phi holds the trigerring kernel functions from all types to a specific type
void HawkesProcess::computeIntensity(const ConstantKernel *mu, const std::vector<Kernel*> & phi ,const EventSequence &s,  const boost::numeric::ublas::vector<double> & time_v, boost::numeric::ublas::vector<long double> & lv) {
	
	//lv will hold the value of the intensity function for the different timepoints of time_v
	long unsigned int nofps=time_v.size();
	if(lv.size()<nofps)
		lv.resize(nofps);
	std::vector<double> mu_params;

	//add the base intensity
	mu->getParams(mu_params);
	std::fill(lv.begin(),lv.end(),mu_params[0]);

	//add the contribution of events of the multiple types that occured in the past.
	for(long unsigned int k=0;k<phi.size();k++){
		boost::numeric::ublas::vector<double> arrival_time_v;
		s.getArrivalTimes(k,arrival_time_v);
		if(arrival_time_v.empty()) {
			continue;
		}
		long unsigned int nofes=arrival_time_v.size();

		//compute the kernel values for each pair of test point and sample
		boost::numeric::ublas::scalar_vector<long double> ones_nofes(nofes,1.0);
		//it repeats the time points (for which the intensity function will be plotted) in the columns (nof columns==nof events of type k)
		boost::numeric::ublas::matrix<double> time_m=outer_prod(time_v,ones_nofes);

		//it repeats the arrival times in the rows (nof rows=nof plot time points)
		boost::numeric::ublas::scalar_vector<long double> ones_nofps(nofps,1.0);
		boost::numeric::ublas::matrix<long double> arrival_time_m=outer_prod(ones_nofps,arrival_time_v);
		boost::numeric::ublas::matrix<long double> D=time_m-arrival_time_m;

		//create the mask matrix Dpos, with 1 indicating a positive entry of D and 0 otherwise
		boost::numeric::ublas::matrix<long double> Dpos=D;
		std::transform(Dpos.data().begin(),Dpos.data().end(),Dpos.data().begin(),[&](double x){
			return x>=0?1.0:0.0;
		}
		);
		//compute the kernel function for each time difference of the matrix D
		std::transform(D.data().begin(),D.data().end(),D.data().begin(),[&](double x){
			return phi[k]->compute(x);
		}
		);
		//filter out the contribution of future events that didn't occur before the timepoint
		D=element_prod(D,Dpos);
		boost::numeric::ublas::vector<long double> lv_copy=lv;
		//TODO: optimize the inner product, for some reason axpy_prod gives numeric error (overflow!!!)
		for(long unsigned int i=0;i<lv.size();i++){
			long double c=0;
			for(long unsigned int j=0;j<D.size2();j++){
				c+=(ones_nofes(j)*D(i,j));
			}
			lv(i)+=c;
		}
	}	
}

//compute intensity non-static methods
void HawkesProcess::computeIntensity(unsigned int k, const EventSequence &s,  boost::numeric::ublas::vector<double> & time_v, boost::numeric::ublas::vector<long double> & lv,unsigned int nofps) const{
	
	//get the trigerring kernel function of other types to type k
	std::vector<Kernel *> phi_k;
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		phi_k.push_back((*iter)[k]);
	}
	
	computeIntensity(mu[k],phi_k,s,time_v,lv,nofps);
}

void HawkesProcess::computeIntensity(unsigned int k, const EventSequence &s,  const boost::numeric::ublas::vector<double> & time_v, boost::numeric::ublas::vector<long double> & lv) const{
	
	//get the trigerring kernel function of other types to type k
	std::vector<Kernel *> phi_k;
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		phi_k.push_back((*iter)[k]);
	}
	
	computeIntensity(mu[k],phi_k,s,time_v,lv);
}

double  HawkesProcess::computeIntensity(unsigned int k, const EventSequence &s, double t) const{
	std::vector<Kernel *> phi_k;
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		phi_k.push_back((*iter)[k]);
	}
	return computeIntensity(mu[k], phi_k, s,t);
}

void HawkesProcess::plotIntensity(const std::string &filename, unsigned int k, const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	
	boost::numeric::ublas::vector<double>  tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  lv;//the value of the intensity function
	computeIntensity(k,s,tv,lv,nofps);

	if(write_png_file){
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines ls 1 lw 2 title 'Type "+std::to_string(k)+"'");
		Gnuplot gp;
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,start_t,end_t,"Time","Intensity");
		gp.send1d(boost::make_tuple(tv,lv));
	}

	if(write_csv_file){
		std::string csv_filename=filename+".csv";
		std::ofstream file{csv_filename};
		//std
		file<<"time,value"<<std::endl;
		for(long unsigned int i=0;i<tv.size();i++){
			file<<tv[i]<<","<<lv[i]<<std::endl;
		}
		file.close();
	}

}

void HawkesProcess::plotIntensities(const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;

	std::string dir_path_str=createModelDirectory();
			
	dir_path_str+="/intensities/";
	if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the intensity plots."<<std::endl;

	plotIntensities_(dir_path_str,s,nofps, write_png_file, write_csv_file);
}

void HawkesProcess::plotIntensities(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const{
	
	if(!write_png_file && !write_csv_file)
		return;
	plotIntensities_(dir_path_str,s,nofps,write_png_file, write_csv_file);
};

void HawkesProcess::plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	for(unsigned int k=0;k<K;k++){
		//create filename for the plot
		std::string filename=dir_path_str+"intensity_"+s.name+"_"+std::to_string(k);
		plotIntensity(filename,k,s,nofps, write_png_file, write_csv_file);
	}
}
/****************************************************************************************************************************************************************************
		 * Model Testing Methods
******************************************************************************************************************************************************************************/
void HawkesProcess::testLogLikelihood(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples, const std::string & true_logl_filename, bool normalized){
	testLogLikelihood(test_dir_path_str,seq_dir_path_str,seq_prefix, nof_test_sequences, start_t, end_t, burnin_samples, true_logl_filename, normalized);

}

void HawkesProcess::testLogLikelihood(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, double t0, double t1, unsigned int burnin_samples, const std::string & true_logl_filename, bool normalized){

	//flush the burnin samples if needed, compute point posterior estimates (mean and mode) with the rest of the samples
	flushBurninSamples(burnin_samples);
	setPosteriorParams();

	//open file with the logl metrics
	std::string test_logl_filename=test_dir_path_str+"test_sequences_loglikelihood_learned.csv";
	std::ofstream test_logl_file{test_logl_filename};
	if(true_logl_filename.empty())
		test_logl_file<<"sequence, post mean estimation logl ,post mode estimation logl  "<<std::endl;
	else
		test_logl_file<<"sequence, post mean estimation logl, post mean logl abs error ,post mode estimation logl, post mode logl abs error  "<<std::endl;

	//open file with true logls
	std::ifstream *true_logl_file=0;
	if(!true_logl_filename.empty()){
		true_logl_file=new std::ifstream{true_logl_filename};

		std::string line;
		//parse out the headers
		getline(*true_logl_file, line);
	}
	accumulator_set<double, stats<tag::mean>> mode_mean_logl;
	accumulator_set<double, stats<tag::mean>> mean_mean_logl;
	accumulator_set<double, stats<tag::mean,tag::variance>> mode_logl_error;
	accumulator_set<double, stats<tag::mean,tag::variance>> mean_logl_error;


	for(unsigned int n=0;n<nof_test_sequences;n++){
		//open sequence file
		std::string csv_seq_filename{seq_dir_path_str+seq_prefix+std::to_string(n)+".csv"};
		std::ifstream test_seq_file{csv_seq_filename};
		EventSequence test_seq{csv_seq_filename};
		test_seq.type.resize(K+1);
		test_seq.K=K+1;//the virtual event which corresponds to the background process has an additional type
		test_seq.N=0;
		test_seq.Nt=0;
		test_seq.thinned_type.resize(K+1);

		//load the sequence from csv file
		test_seq.load( test_seq_file,1);
		test_seq_file.close();

		//add virtual event if it doesn't exist
		Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
		test_seq.addEvent(ve,0);
		
		//test_seq.start_t=start_t;
		test_seq.start_t=t0>=0?t0:start_t;
		
		double last_arrival_time=test_seq.full.rbegin()->first;
		//test_seq.end_t=(t1>start_t)?t1:last_arrival_time;
		test_seq.end_t=(t1>test_seq.start_t)?t1:last_arrival_time;
		
		test_seq_file.close();
		//read true logl
		double true_logl;
		if(!true_logl_filename.empty()){
			std::string line;
			std::string::size_type sz;
			getline(*true_logl_file, line);
			std::vector<std::string> logl_strs;
			boost::algorithm::split(logl_strs, line, boost::is_any_of(","));
			true_logl=std::stod (logl_strs[1],&sz);
		}

		//compute loglikelihoods
		//with posterior mean
		double post_mean_logl;
		
		if(!phi.empty())
			//post_mean_logl=loglikelihood(test_seq, post_mean_mu, post_mean_phi, normalized);
			post_mean_logl=loglikelihood(test_seq, post_mean_mu, post_mean_phi, normalized, t0);
		else //there is not mutual excitation part
			post_mean_logl=loglikelihood(test_seq, post_mean_mu, normalized, t0);
		mean_mean_logl(post_mean_logl);
		double post_mean_logl_error;
		if(!true_logl_filename.empty()){
			post_mean_logl_error=std::abs(post_mean_logl-true_logl);
			mean_logl_error(post_mean_logl_error);
		}

		//with posterior mode
		double post_mode_logl;
		if(!phi.empty())
			post_mode_logl=loglikelihood(test_seq, post_mode_mu, post_mode_phi, normalized, t0);
		else //there is not mutual excitation part
			post_mode_logl=loglikelihood(test_seq, post_mode_mu, normalized, t0);
		mode_mean_logl(post_mode_logl);
		double post_mode_logl_error;
		if(!true_logl_filename.empty()){
			post_mode_logl_error=std::abs(post_mode_logl-true_logl);
			mode_logl_error(std::abs(post_mode_logl_error));
		}
		if(!true_logl_filename.empty()){
			test_logl_file<<n<<","<<post_mean_logl<<","<<post_mean_logl_error<<","<<post_mode_logl<<","<<post_mode_logl_error<<std::endl;
		}
		else
			test_logl_file<<n<<","<<post_mean_logl<<","<<post_mode_logl<<std::endl;
	}
	if(!true_logl_filename.empty()){
		true_logl_file->close();
	}
	if(!true_logl_filename.empty()){
		test_logl_file<<"all, mean logl with post mean, mean abs error with post mean, std of error with post mean,mean logl with post mode,mean abs error with post mode,std of error with post mode"<<std::endl;
		test_logl_file<<"summary,"<<mean(mean_mean_logl)<<","<<mean(mean_logl_error)<<","<<sqrt(variance(mean_logl_error))<<","<<mean(mode_mean_logl)<<","<<mean(mode_logl_error)<<","<<sqrt(variance(mode_logl_error))<<std::endl;
	}else{
		test_logl_file<<"all, mean logl with post mean, mean logl with post mode"<<std::endl;
		test_logl_file<<"summary,"<<mean(mean_mean_logl)<<","<<mean(mode_mean_logl)<<std::endl;
	}
	test_logl_file.close();
}

void HawkesProcess::testPredictions(EventSequence & seq, const std::string &test_dir_path_str, double & mode_rmse, double & mean_rmse, double & mode_errorrate, double & mean_errorrate, double t0){
	std::string name=seq.name;
	seq.name.clear();
	seq.name="mode_"+name;
	HawkesProcess::predictEventSequence(seq,test_dir_path_str, post_mode_mu, post_mode_phi, &mode_rmse, &mode_errorrate, t0);
	seq.name.clear();
	seq.name="mean_"+name;
	HawkesProcess::predictEventSequence(seq,test_dir_path_str, post_mean_mu, post_mean_phi, &mean_rmse, &mean_errorrate, t0);
	seq.name.clear();
	seq.name=name;
}


/****************************************************************************************************************************************************************************
 * Model Prediction Methods
******************************************************************************************************************************************************************************/
//it computes the probability that the next arrival after tn will happen at t
double HawkesProcess::computeNextArrivalTimeProbability(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, double tn, double t){
	//EventSequence seq_tn=seq.crop(seq.start_t, tn);//keep the events of the event sequence up to time t
	
	long unsigned int K=mu.size();// get the number of types
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	for(unsigned int k=0;k<K;k++){ //get the excitation effect on type k
		for(unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];
		}
	}
	//get the total intensity (arrival of event regardless its type)
	double p=0;
	for(unsigned int k=0;k<K;k++)
		p+=computeIntensity(mu[k],phi_k[k], seq, tn, t); //todo: replace this to use the cropped sequence
	//monte carlo integration
	auto f=[&](double t)->double{
		double s=0;
		for(unsigned int k=0;k<K;k++)
			s+=computeIntensity(mu[k],phi_k[k], seq, tn, t); //todo: replace this to use the cropped sequence
		
		return s;
	};

	double res2=monte_carlo_integral(f, tn, t);
	double res=p*exp(-res2);
	return res;
	
}

double HawkesProcess::predictNextArrivalTime(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, unsigned int nof_samples, unsigned int nof_threads){

	pthread_t sim_threads[nof_threads];
	unsigned int nof_samples_thread[nof_threads];
	double *sim_time_ps=(double*) calloc (nof_threads,sizeof(double));
	
	//distribute the events across the threads
	unsigned int nof_samples_q=nof_samples/nof_threads;
	unsigned int nof_samples_m=nof_samples%nof_threads;
	for(unsigned int t=0;t<nof_threads;t++){
		nof_samples_thread[t]=nof_samples_q;
		if(t<nof_samples_m)
			nof_samples_thread[t]++;
	}
	
	//mu, phi, seq, t_n, seq.end_t
	//create threads
	for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
		if(!nof_samples_thread[thread_id])
			break;
	
		predictNextArrivalTimeThreadParams* p;
		//use the history of realized and thinned events which correspond to type k2
		p= new predictNextArrivalTimeThreadParams(thread_id, nof_samples_thread[thread_id], sim_time_ps[thread_id], mu, phi, seq, t_n);
		int rc = pthread_create(&sim_threads[thread_id], NULL, predictNextArrivalTime_, (void *)p);
		if (rc){
			 std::cerr<< "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
	}
	//wait for all the partial sums to be computed
	for (unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t])
			pthread_join (sim_threads[t], NULL);
	}
	
	double nxt_arrival_time=0.0; 
	for(unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t])
			nxt_arrival_time+=sim_time_ps[t];//TODO, TODO, TODO: uncomment this
	}
	nxt_arrival_time/=nof_threads;
	return nxt_arrival_time;
}

void * HawkesProcess::predictNextArrivalTime_(void *p){
	
	std::unique_ptr<predictNextArrivalTimeThreadParams> params(static_cast< predictNextArrivalTimeThreadParams * >(p));
	unsigned int nof_samples=params->nof_samples;
	double &mean_arrival_time=params->mean_arrival_time;
	const std::vector<ConstantKernel *>  & mu=params->mu;
	const std::vector<std::vector<Kernel*>> & phi=params->phi;
	const EventSequence & seq=params->seq;
	double t_n=params->t_n;
		
	accumulator_set<double, stats<tag::mean>> arrival_time_acc;
	for(unsigned int r=0; r!= nof_samples;r++){
		std::vector<Event *> nxt_events;
		HawkesProcess::simulateNxt(mu, phi, seq, t_n, seq.end_t, nxt_events, 1);
		if(!nxt_events.empty()){
			double t_nxt=nxt_events[0]->time;
			arrival_time_acc(t_nxt);
		}
	}
	mean_arrival_time=mean(arrival_time_acc);
	
	return 0;
}

int HawkesProcess::predictNextType(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, std::map<int, double> & type_prob, unsigned int nof_samples, unsigned int nof_threads){

	pthread_t sim_threads[nof_threads];
	unsigned int nof_samples_thread[nof_threads];
	
	typedef std::map<int, unsigned int> TypeCounts;
	//TypeCounts *sim_type_pc=(TypeCounts*) calloc (nof_threads,sizeof(TypeCounts));
	std::vector<TypeCounts *> sim_type_pc;
	
	//distribute the events across the threads
	unsigned int nof_samples_q=nof_samples/nof_threads;
	unsigned int nof_samples_m=nof_samples%nof_threads;
	for(unsigned int t=0;t<nof_threads;t++){
		sim_type_pc.push_back(new std::map<int, unsigned int>);
		for(long unsigned int k=0;k!=mu.size();k++){
		
			(*(sim_type_pc[t]))[k]=0;
		}
		nof_samples_thread[t]=nof_samples_q;
		if(t<nof_samples_m)
			nof_samples_thread[t]++;
	}
	
	//mu, phi, seq, t_n, seq.end_t
	//create threads
	for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
		if(!nof_samples_thread[thread_id])
			break;
	
		predictNextTypeThreadParams* p;
		//use the history of realized and thinned events which correspond to type k2
		p= new predictNextTypeThreadParams(thread_id, nof_samples_thread[thread_id], *sim_type_pc[thread_id], mu, phi, seq, t_n);
		int rc = pthread_create(&sim_threads[thread_id], NULL, predictNextType_, (void *)p);
		if (rc){
			 std::cerr<< "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
	}
	//wait for all the partial sums to be computed
	for (unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t])
			pthread_join (sim_threads[t], NULL);
	}
	
	//compute the probability of each type
	long unsigned int K=mu.size();
	for(unsigned int k=0;k!=K;k++){
		type_prob[k]=0.0;
	}
	for(unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t]){
			for(unsigned int k=0;k!=K;k++){
				type_prob[k]+=((double)(*(sim_type_pc[t]))[k])/((double)nof_samples_thread[t]*nof_threads);
			}
		}
	}
	
	//compute the type with the highest probability and return it as predicted type
	int max_k = -1;
	double max_prob_k = 0;

	for(auto iter = type_prob.begin(); iter != type_prob.end(); ++iter ) {
		
		if (iter ->second > max_prob_k) {
			max_k = iter->first;
			max_prob_k = iter->second;
		 }
	}

	return max_k;
}

void * HawkesProcess::predictNextType_(void *p){
	
	std::unique_ptr<predictNextTypeThreadParams> params(static_cast< predictNextTypeThreadParams * >(p));
	unsigned int nof_samples=params->nof_samples;
	//double &mean_arrival_time=params->mean_arrival_time;
	const std::vector<ConstantKernel *>  & mu=params->mu;
	const std::vector<std::vector<Kernel*>> & phi=params->phi;
	std::map<int, unsigned int> & type_counts=params->type_counts; 
	const EventSequence & seq=params->seq;
	double t_n=params->t_n;
		

	for(unsigned int r=0; r!= nof_samples;r++){
		std::vector<Event *> nxt_events;
		HawkesProcess::simulateNxt(mu, phi, seq, t_n, seq.end_t, nxt_events, 1);
		if(!nxt_events.empty()){
			int l_nxt=nxt_events[0]->type;
			
			type_counts[l_nxt]++; 
			
		}
	}

	return 0;
}

void HawkesProcess::predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, double t0, unsigned int nof_samples) {
	HawkesProcess::predictEventSequence(seq , seq_dir_path_str, mu,  phi, 0,0, t0, nof_samples);
}


void HawkesProcess::predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, double *rmse, double *error_rate, double t0, unsigned int nof_samples) {
	std::string predict_filename=seq_dir_path_str+seq.name+"_prediction.csv"; //in each row it will hold the mean rmse for the predicition of the arrival time of the next event for each test event sequence, one column for post mean/ post mode
	std::ofstream predict_file{predict_filename};
	
	//seq.print(std::cout);
	//create the headers
	predict_file<<"previous time, real time, predicted time , mse, real type, predicted type, hard error rate ";
	for(long unsigned int k=0;k!=mu.size();k++)
		predict_file<<", prob of type "<<k;
	predict_file<<", soft error rate "<<std::endl;
	
	//accumulators for the metrics across all the test event sequences
	accumulator_set<double, stats<tag::mean, tag::variance>> mse_acc;
	accumulator_set<double, stats<tag::mean>> errorrate_acc;
	accumulator_set<double, stats<tag::mean>> soft_errorrate_acc;

	t0=(t0>seq.start_t)?t0:seq.start_t;
	for(auto e_iter=seq.full.begin(); e_iter!=seq.full.end();e_iter++){
		
			double t_n=e_iter->second->time; //get the current time in the history, for the first step it contains the virtual event
		
			if (t_n > seq.end_t)
				
				break;
			
		
			//check whether t_n is the last evidence in the sequence
			auto nxt_e_iter=std::next(e_iter);
			if(nxt_e_iter==seq.full.end() || !nxt_e_iter->second) //there is no evidence in the sequence after t_n to compare against
				break;
			
			
			//get error for the predicted arrival time at next step
			 //find the next event that occurs after t_n, if t_n is in the time interval under consideration: the next arrival time to be predicted
			double t_np1=nxt_e_iter->second->time;
			//find the arrival time of an event after time t_n that achieves the minimum Bayes risk 
			double th_np1=HawkesProcess::predictNextArrivalTime(seq,t_n,mu,phi, nof_samples);
			//compare the prediction with the real occurence and write in the csv files
			double t_error=(th_np1-t_np1)*(th_np1-t_np1);
			mse_acc(t_error);
			
			//get error for the predicted event type at next step
			 //find the next event type that occurs after t_n, if t_n is in the time interval under consideration: the type of the next event to be predicted
			unsigned int l_np1=nxt_e_iter->second->type;
			//find the type of event after time t_n that achieves the minimum Bayes risk  with the mode point estimates
			std::map<int, double> type_prob;
			unsigned int lh_np1=HawkesProcess::predictNextType(seq,t_n,mu,phi, type_prob, nof_samples);
			//compare the prediction with the real occurence and write in the csv files
			bool l_error=(lh_np1!=l_np1);
			errorrate_acc(l_error);
			
			predict_file<<t_n<<","<<t_np1<<","<<th_np1<<","<<t_error<<","<<l_np1<<","<<lh_np1<<","<<l_error;
			for(auto iter=type_prob.begin();iter!=type_prob.end();iter++)
				predict_file<<", "<<iter->second;
			double soft_error=std::abs(type_prob[l_np1]-1);
			soft_errorrate_acc(soft_error);
			predict_file<<", "<<soft_error<<std::endl;
	}
		
	//gather the statistics across all timesteps
	predict_file<<"all, rmse , std mse, error rate"<<std::endl;
    double rmse_1=std::sqrt(mean(mse_acc));
    if(rmse){
    	*rmse=rmse_1;
    }
    double rmse_std=variance(mse_acc);
    double errorrate_1=mean(errorrate_acc);
    if(error_rate){
    	*error_rate=errorrate_1;
    }
	predict_file<<"summary,"<<rmse_1<<","<<rmse_std<<","<<errorrate_1<<std::endl;
	predict_file.close();
}

