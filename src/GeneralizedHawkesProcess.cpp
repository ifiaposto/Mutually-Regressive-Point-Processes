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


#include "GeneralizedHawkesProcess.hpp"
#include "Kernels.hpp"
#include "stat_utils.hpp"
#include "plot_utils.hpp"
#include "struct_utils.hpp"
#include "debug.hpp"
BOOST_CLASS_EXPORT(GeneralizedHawkesProcess)


/****************************************************************************************************************************************************************************
 *
 * Generalized Hawkes Process (mutual excitation and inhibition)
 *
******************************************************************************************************************************************************************************/


/****************************************************************************************************************************************************************************
 * Model Construction and Destruction Methods
******************************************************************************************************************************************************************************/

/***************************************************   anononymous point processes    *********************************************/
//empty process
GeneralizedHawkesProcess::GeneralizedHawkesProcess()=default;

//empty multivariate process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te): PointProcess(k, ts, te), HawkesProcess(k,ts,te){} 

//multivariate mutually independent poisson process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m): PointProcess(k, ts, te), HawkesProcess(k,ts,te,m){} 

//hawkes process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p ): PointProcess(k, ts, te), HawkesProcess(k,ts,te,m,p){} 

//multivariate multually dependent poisson process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps): PointProcess(k, ts, te), HawkesProcess(k, ts, te, m), pt{h}, psi{ps} {
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}
	}
}
	 
//generalized hawkes process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p, std::vector<LogisticKernel*> h, std::vector<Kernel *> ps): PointProcess(k, ts, te), HawkesProcess(k, ts, te, m, p), pt{h}, psi{ps}{
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}
	}
}

//generalized hawkes process-with normal gamma prior between the excitation and inhibition coefficients
//GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<std::vector<double>> kp): PointProcess(k, ts, te), HawkesProcess(k, ts, te, m, p), pt{h}, kappa{kp} {}

GeneralizedHawkesProcess::GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps, std::vector<std::vector<SparseNormalGammaParameters *>> hp):  PointProcess(k, ts, te), HawkesProcess(k, ts, te, m, p), pt{h}, psi{ps}, pt_hp{hp} {

	//TODO:create a hierarchical logistic kernel
	for(unsigned int k=0;k!=K;k++){//set the hyperpriors of the weights of the kernel of type k
		for(unsigned int k2=1;k2<=K;k2++){//skip the hyperprior for the bias term
			pt[k]->hp[k2]->hp.push_back(new NormalGammaParameters{0,hp[k][k2-1]->lambda,0, hp[k][k2-1]->beta_tau});
		}
	}
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}
	}
	//set the prior of the sparse normal gamma, alpha parameter as the prior of the excitatory kernel
	for(unsigned int k=0;k!=K;k++){
		for(unsigned int k2=1;k2<=K;k2++){
			pt_hp[k][k2]->hp.push_back(p[k][k2]->hp[0]);
		}
	}

};

/***************************************************   named point processes    *********************************************/
//empty process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n): PointProcess(n), HawkesProcess(n){};

//empty multivariate process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n,unsigned int k, double ts, double te): PointProcess(n, k, ts, te), HawkesProcess(n,k,ts,te){}

//multivariate mutually independent poisson process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m): PointProcess(n, k, ts, te), HawkesProcess(n, k,ts,te,m){} 

//hawkes process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p): PointProcess(n, k, ts, te), HawkesProcess(n,k,ts,te,m,p){}

//multivariate multually dependent poisson process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps ): PointProcess(n, k, ts, te), HawkesProcess(n, k, ts, te, m), pt{h}, psi{ps}{
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}
	}
}

//generalized hawkes process
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps): PointProcess(n, k, ts, te), HawkesProcess(n, k, ts, te, m, p), pt{h}, psi{ps}{
	
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}

	}
}

GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::string n, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel*>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps, std::vector<std::vector<SparseNormalGammaParameters *>> hp): PointProcess(n, k, ts, te), HawkesProcess(n, k, ts, te, m, p), pt{h}, psi{ps}, pt_hp{hp}{
	for(unsigned int k=0;k!=K;k++){//set the hyperpriors of the weights of the kernel of type k
		for(unsigned int k2=1;k2<=K;k2++){//skip the hyperprior for the bias term
			pt[k]->hp[k2]->hp.push_back(new NormalGammaParameters{0,hp[k][k2-1]->lambda,0, hp[k][k2-1]->beta_tau});
		}
	}
	//set the history kernel functions in the logistic kernel
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		(*iter)->psi.clear();
		(*iter)->d=K*psi.size();
		for(unsigned int k=0;k!=K;k++){
			for(long unsigned int d=0;d!=psi.size();d++){
				(*iter)->psi.push_back(psi[d]);
			}
		}
	}

	//set the prior of the sparse normal gamma, alpha parameter as the prior of the excitatory kernel
	for(unsigned int k=0;k!=K;k++){
		for(unsigned int k2=0;k2!=K;k2++){
			pt_hp[k][k2]->hp.push_back(p[k][k2]->hp[0]);
		}
	}
};

//load model from file
GeneralizedHawkesProcess::GeneralizedHawkesProcess(std::ifstream &file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

GeneralizedHawkesProcess::~GeneralizedHawkesProcess(){
	//free the posterior samples for the parameters of the history kernels
	if(!psi_posterior_samples.empty()){
		for(long unsigned int d=0;d<psi_posterior_samples.size();d++){	
			if(!psi_posterior_samples[d].empty()){
				for(unsigned int p=0;p<psi[d]->nofp;p++){
					if(!psi_posterior_samples[d][p].empty()){
						for(long unsigned int r=0;r!=psi_posterior_samples[d][p].size();r++){
							if(!psi_posterior_samples[d][p][r].empty()){
								psi_posterior_samples[d][p][r].clear();
							}
						}
						if(!psi_posterior_samples[d][p].empty())
							forced_clear(psi_posterior_samples[d][p]);
					}
				}
			}
			if(!psi_posterior_samples[d].empty())
				forced_clear(psi_posterior_samples[d]);
		}
		if(!psi_posterior_samples.empty())
			forced_clear(psi_posterior_samples);
	}
		
	//free the posterior samples for the precision of the sparse normal gamma hyperprior
	if(!tau_posterior_samples.empty()){
		for(long unsigned int k=0;k<tau_posterior_samples.size();k++){
			if(!tau_posterior_samples[k].empty()){
				for(long unsigned int k2=0;k2!=tau_posterior_samples[k].size();k2++){
					if(!tau_posterior_samples[k][k2].empty()){
						for(long unsigned int r=0;r!=tau_posterior_samples[k][k2].size();r++){
							if(!tau_posterior_samples[k][k2][r].empty()){
								tau_posterior_samples[k][k2][r].clear();
							}
						}
						if(!tau_posterior_samples[k][k2].empty())
							forced_clear(tau_posterior_samples[k][k2]);
					}
				}
			}
			if(!tau_posterior_samples[k].empty())
				forced_clear(tau_posterior_samples[k]);
		}
		if(!tau_posterior_samples.empty())
			forced_clear(tau_posterior_samples);
	}
	//free kernel function of the thining probability
	for(unsigned int k=0;k<pt.size();k++){
		if((pt[k])){
			free(pt[k]);
			pt[k]=0;
		}
	}
	forced_clear(pt);
		//free history kernel function of the thining probability
	for(unsigned int k=0;k<psi.size();k++){
		if((psi[k])){
			free(psi[k]);
			psi[k]=0;
		}
	}
	forced_clear(psi);
	//free the posterior logistic kernels
	for(auto iter=post_mean_pt.begin();iter!=post_mean_pt.end();iter++)
		if(!(*iter))
			free(*iter);
	for(auto iter=post_mode_pt.begin();iter!=post_mode_pt.end();iter++)
		if(!(*iter))
			free(*iter);
	//free the posterior history kernels
	for(auto iter=post_mode_psi.begin();iter!=post_mode_psi.end();iter++)
			if(!(*iter))
				free(*iter);
	for(auto iter=post_mean_psi.begin();iter!=post_mean_psi.end();iter++)
			if(!(*iter))
				free(*iter);
	//free the sparse normal gamma hyperparameters
	for(auto iter=pt_hp.begin();iter!=pt_hp.end();iter++)
			for(auto iter_2=iter->begin(); iter_2!=iter->end();iter_2++)
				if(!(*iter_2))
					free(*iter_2);
}

/****************************************************************************************************************************************************************************
 * Model Utility Methods
******************************************************************************************************************************************************************************/

std::string GeneralizedHawkesProcess::createModelDirectory() const{
	
	boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
    std::string mcmc_curr_path_str=curr_path_boost.string();
    std::string dir_path_str;
    if(!name.empty())
    	dir_path_str=mcmc_curr_path_str+"/"+name+"/";
    else
    	dir_path_str=mcmc_curr_path_str+"/ghp/";
    
    if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the mcmc plots."<<std::endl;
    
    return dir_path_str;
    
}

std::string GeneralizedHawkesProcess::createModelInferenceDirectory() const{
	
	std::string dir_path_str=createModelDirectory();
	dir_path_str+="/inference_plots/";
	if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the mcmc plots."<<std::endl;
	
	return dir_path_str;
}

/****************************************************************************************************************************************************************************
 * Model Serialization/Deserialization Methods
******************************************************************************************************************************************************************************/
void GeneralizedHawkesProcess::State::serialize(boost::archive::text_oarchive &ar, const unsigned int version){
	ar & K;
	ar & pt;
	ar & psi;
	ar & polyagammas;
	ar & thinned_polyagammas;
	ar & mcmc_iter;
	ar & mcmc_params;
}

void GeneralizedHawkesProcess::State::serialize(boost::archive::text_iarchive &ar, const unsigned int version){
	ar & K;
	ar & pt;
	ar & psi;
	ar & polyagammas;
	ar & thinned_polyagammas;
	ar & mcmc_iter;	
	ar & mcmc_params;
}

void GeneralizedHawkesProcess::State::serialize(boost::archive::binary_oarchive &ar, const unsigned int version){
	
	ar & K;
	ar & pt;
	ar & psi;
	ar & polyagammas;
	ar & thinned_polyagammas;
	ar & mcmc_iter;
	ar & mcmc_params;
}

void GeneralizedHawkesProcess::State::serialize(boost::archive::binary_iarchive &ar, const unsigned int version){
	
	ar & K;
	ar & pt;
	ar & psi;
	ar & polyagammas;
	ar & thinned_polyagammas;
	ar & mcmc_iter;
	ar & mcmc_params;
}
//TODO: SERIALIZE THE SPARSE NORMAL GAMMA HYPERPRIOR !!!!!!!

void GeneralizedHawkesProcess::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<GeneralizedHawkesProcess,HawkesProcess>();
	    ar & boost::serialization::base_object<HawkesProcess>(*this);
		ar & pt;
		ar & psi;
		ar & pt_hp;
		ar & post_mean_pt;
		ar & post_mode_pt;
		ar & post_mean_psi;
		ar & post_mode_psi;
		ar & pt_posterior_samples;
		ar & psi_posterior_samples;
		ar & tau_posterior_samples;
		ar & mcmc_state;
		ar & profiling_mcmc_step;
		ar & profiling_compute_history;
		ar & profiling_thinned_events_sampling;
		ar &  profiling_polyagammas_sampling;
		ar & profiling_logistic_sampling;
		ar & profiling_hawkes_sampling;
	
}
void GeneralizedHawkesProcess::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<GeneralizedHawkesProcess,HawkesProcess>();
	    ar & boost::serialization::base_object<HawkesProcess>(*this);
		ar & pt;
		ar & psi;
		ar & pt_hp;
		ar & post_mean_pt;
		ar & post_mode_pt;//todo: add post mode and mean for psi functions
		ar & post_mean_psi;
		ar & post_mode_psi;
		ar & pt_posterior_samples;
		ar & psi_posterior_samples;
		ar & tau_posterior_samples;
		ar & mcmc_state;
		ar & profiling_mcmc_step;
		ar & profiling_compute_history;
		ar & profiling_thinned_events_sampling;
		ar & profiling_polyagammas_sampling;
		ar & profiling_logistic_sampling;
		ar & profiling_hawkes_sampling;
		
}

void GeneralizedHawkesProcess::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<GeneralizedHawkesProcess,HawkesProcess>();
	    ar & boost::serialization::base_object<HawkesProcess>(*this);

		ar & pt;
		ar & psi;
		ar & pt_hp;
		ar & post_mean_pt;
		ar & post_mode_pt;
		ar & post_mean_psi;
		ar & post_mode_psi;
		ar & pt_posterior_samples;
		ar & psi_posterior_samples;
		ar & tau_posterior_samples;
		ar & mcmc_state;
		ar & profiling_mcmc_step;
		ar & profiling_compute_history;
		ar & profiling_thinned_events_sampling;
		ar &  profiling_polyagammas_sampling;
		ar & profiling_logistic_sampling;
		ar & profiling_hawkes_sampling;
		
}
void GeneralizedHawkesProcess::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	    boost::serialization::void_cast_register<GeneralizedHawkesProcess,HawkesProcess>();
	    ar & boost::serialization::base_object<HawkesProcess>(*this);
		ar & pt;
		ar & psi;
		ar & pt_hp;
		ar & post_mean_pt;
		ar & post_mode_pt;
		ar & post_mean_psi;
		ar & post_mode_psi;
		ar & pt_posterior_samples;
		ar & psi_posterior_samples;
		ar & tau_posterior_samples;
	    ar & mcmc_state;
		ar & profiling_mcmc_step;
		ar & profiling_compute_history;
		ar & profiling_thinned_events_sampling;
		ar &  profiling_polyagammas_sampling;
		ar & profiling_logistic_sampling;
		ar & profiling_hawkes_sampling;
	
}
void GeneralizedHawkesProcess::save(std::ofstream & file) const{
	
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void GeneralizedHawkesProcess::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void GeneralizedHawkesProcess::saveParameters(std::ofstream & file) const{
	HawkesProcess::saveParameters(file);
}

/****************************************************************************************************************************************************************************
 * Model Generation Methods
******************************************************************************************************************************************************************************/

//Generates a new model from the priors. it is used for generation of synthetic datasets
void GeneralizedHawkesProcess::generate(){
	//generate the model parameters from the priors for the excitation part
	HawkesProcess::generate();
		
	//update the prior in case of a hierarchical model
	if(!pt_hp.empty())
		setSparseNormalGammas();
	//generate the model parameters from the priors for the thinning part: the history kernel functions
	for(unsigned int d=0;d<psi.size();d++)
			psi[d]->generate();
	
	//generate the model parameters from the priors for the thinning part: the sigmoid kernels
	for(unsigned int k=0;k<pt.size();k++){
			pt[k]->generate();
			pt[k]->psi.clear();
			for(long unsigned int k2=0;k2!=K;k2++){
				for(long unsigned int d=0;d<psi.size();d++)
					pt[k]->psi.push_back(psi[d]);
			}
	}

}

void GeneralizedHawkesProcess::setSparseNormalGamma(const std::vector<Kernel *> & phi, const std::vector<SparseNormalGammaParameters *> & hp, LogisticKernel *pt){

	unsigned int nofhp=pt->hp.size();
	if(phi.size()!=nofhp-1){
		std::cerr<<"wrong number of excitation coeffcients\n";
	}
	if(hp.size()!=nofhp-1)
		std::cerr<<"wrong number of variance regulators\n";
	if(!pt)
		std::cerr<<"uninitialized logistic kernel\n";

	for(unsigned int i=1;i<nofhp;i++){//skip the hyperparameters which correspond to the bias term in the sigmoid

		if(!pt->hp[i])
			std::cerr<<"unitialized hyperparameter for dimension "<<i<<std::endl;
		if(!phi[i-1])
			std::cerr<<"uninitialized excitation kernel\n";
		if(phi[i-1]->p.size()!=2)
			std::cerr<<" wrongly allocated excitation kernel\n";
	}

	for(unsigned int i=1;i<nofhp;i++){//skip the hyperparameters which correspond to the bias term in the sigmoid
		if(!pt->hp[i])
			std::cerr<<"unitialized hyperparameter for dimension "<<i<<std::endl;
		if(!phi[i-1])
			std::cerr<<"uninitialized excitation kernel\n";
		if(pt->hp[i]->type!=Distribution_Normal){
		
			std::cerr<<"wrong hyperprior for the weights of the logistic kernel\n";
		}

		NormalParameters * pt_hp=(NormalParameters *)pt->hp[i];//normal parameters for the weight of the i-th kernel function
		if(pt_hp->hp.size()!=1 || !pt_hp->hp[0] || pt_hp->hp[0]->type!=Distribution_NormalGamma){//a normal gamma parameter is assumed for the precision and the weight of the i-th kernel function
			pt_hp->hp.clear();
			pt_hp->hp.push_back(new NormalGammaParameters{0,0,0, 0});
		}
		NormalGammaParameters  *pt_hhp=(NormalGammaParameters *)pt_hp->hp[0];
		

		pt_hhp->alpha=hp[i-1]->phi_tau->op(hp[i-1]->nu_tau, phi[i-1]->p[0], hp[i-1]->alpha_tau);
		pt_hhp->beta=hp[i-1]->beta_tau;
		pt_hhp->mu=1/(-hp[i-1]->phi_mu->op(hp[i-1]->nu_mu, phi[i-1]->p[0], hp[i-1]->alpha_mu));
		pt_hhp->kappa=hp[i-1]->lambda;
	}
}

void GeneralizedHawkesProcess::setSparseNormalGammas(){

	for(unsigned int k=0;k!=K;k++){//update the normal-gamma prior for each type of the process
		//gather the excitation effect on type k
		std::vector<Kernel *>phi_k;//excitation effects on type k
		std::vector<SparseNormalGammaParameters *> pt_hp_k; //hyperprior for thinning effects on type k
		for(unsigned int k2=0;k2<K;k2++){
			phi_k.push_back(phi[k2][k]);
			pt_hp_k.push_back(pt_hp[k2][k]);
		}
		//update normal gamma for type k
		setSparseNormalGamma(phi_k,pt_hp_k,pt[k]);
	}
}

/****************************************************************************************************************************************************************************
 * Model Simulation Methods
******************************************************************************************************************************************************************************/
EventSequence GeneralizedHawkesProcess::simulate(unsigned int id){
	return GeneralizedHawkesProcess::simulate(mu, phi, pt, start_t, end_t, name, id);
}


void  GeneralizedHawkesProcess::simulate(EventSequence & seq, double dt) {
	return GeneralizedHawkesProcess::simulate(mu, phi, pt, seq, dt);
}

void GeneralizedHawkesProcess::simulateNxt(const EventSequence & seq, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N){
	return(GeneralizedHawkesProcess::simulateNxt(mu, phi, pt, seq, start_t, end_t, nxt_events, N));
	
}

void GeneralizedHawkesProcess::simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, const EventSequence &s, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N){
	
	if(N==0 || end_t<=start_t)//no more events to generate, stop the recursion
		return;
	double th=start_t;
	while(true){
		std::vector<Event *> e_v;
		
		HawkesProcess::simulateNxt(mu, phi, s, th, start_t, end_t, e_v, 1);//simulate the exciting process just to get the next event
		if(!e_v.empty()){
			Event * e=e_v[0]; //get the last event from the excitation part of the process after time start_t, if any
			
			e->observed=true;
			double * h;
			computeKernelHistory(s.type, pt[e->type], start_t, e->time, h);			
			double p_n=pt[e->type]->compute(h);
			free(h);
			
			std::random_device rd;
			std::mt19937 gen(rd());
			std::bernoulli_distribution bern_distribution(p_n);
			bool b_n=bern_distribution(gen);
			if(b_n){//the event is realized
				e->observed=true;
				Event *parent=s.findEvent(e->parent->time);
				nxt_events.push_back(new Event{s.type[e->type].size(),e->type,e->time,e->K,true, parent});
				//get the next event: continue the recursion
				simulateNxt(mu, phi, pt, s, e->time, end_t, nxt_events, N-1);
				free(e);
				return;
			}	
			else {
				start_t=e->time;//shift the interval for model simulation by the arrival time of the thinned event in case it is rejected
				free(e);
			}	
		}
		else{ //no event was trigerred, the recursion has to stop
			return ;
		}
	}
}

void GeneralizedHawkesProcess::simulate(std::vector<ConstantKernel *> & mu, std::vector<std::vector< Kernel *>> & phi, std::vector<LogisticKernel *> & pt, EventSequence &s, double dt){
		

	//simulate thefull sequence (with both the thinned and the observed events)

	EventSequence ghp_seq=s;
	HawkesProcess::simulate(mu, phi, ghp_seq, dt);


	s.end_t+=dt;
	//start thinning for each event generated from the Hawkes process, skip the virtual event
	for(auto e_iter=std::next(ghp_seq.full.begin());e_iter!=ghp_seq.full.end();){
		if(e_iter->second->time>ghp_seq.end_t-dt){
			int l_n=e_iter->second->type;//type of the current event
			double t_n=e_iter->second->time;//time of the current event
			Event *parent=s.findEvent(e_iter->second->parent->time);

			double p_n=computeRealizationProbability(s,pt[l_n],t_n);
			std::random_device rd;
			std::mt19937 gen(rd());
			std::bernoulli_distribution bern_distribution(p_n);
			bool b_n=bern_distribution(gen);
			if(b_n){//the event is realized
				e_iter->second->observed=true;
				s.addEvent(new Event{e_iter->second->id,e_iter->second->type,e_iter->second->time,e_iter->second->K,true, parent}, parent);
				e_iter++;
			}
			else{//the event is thinned
				//thin the event and all of its offsprings from the sequence, start thinning procedure from the next event in the reduced sequence
				s.addEvent(new Event{e_iter->second->id,e_iter->second->type,e_iter->second->time,e_iter->second->K,false, parent}, parent);//create copy of thinned events because it will be released by the pruneCluster
				e_iter=ghp_seq.pruneCluster(e_iter->second);
			}
		}
		else
			e_iter++;
	}
}


EventSequence GeneralizedHawkesProcess::simulate(std::vector<ConstantKernel *> & mu,std::vector<std::vector< Kernel *>> & phi,std::vector<LogisticKernel *> & pt, double start_t, double end_t, std::string name, unsigned int id){
		

	std::string seq_name;
	if(!name.empty())
		seq_name=name+"_sequence_"+std::to_string(id);
	else
		seq_name="ghp_sequence_"+std::to_string(id);
	unsigned int K=mu.size();//number of types
	EventSequence ghp_seq(seq_name, K+1, start_t, start_t);
	Event *ve=new Event{0,(int)K,0.0,K+1,true, (Event *)0};
	ghp_seq.addEvent(ve,0);
	
	GeneralizedHawkesProcess::simulate(mu, phi, pt, ghp_seq,end_t-start_t);
	
	return ghp_seq;
		
}

/****************************************************************************************************************************************************************************
 * Model Print Methods
******************************************************************************************************************************************************************************/

void GeneralizedHawkesProcess::print(std::ostream &file) const{

	std::cout<<"print model\n";
	file<<"Excitation Part \n";
	HawkesProcess::print(file);
	HawkesProcess::print(std::cout);
	file<<"Thinning Part \n";

	for(unsigned int k=0;k<pt.size();k++){
		file<<"effect on type "<<k<<std::endl;
			pt[k]->print(file);
		if(!post_mode_pt.empty()){
			file<<"posterior mode parameters\n";
			post_mode_pt[k]->print(file);
		}
		if(!post_mean_pt.empty()){
			file<<"posterior mean parameters\n";
			post_mean_pt[k]->print(file);
		}
	}
	
	file<<"hyperparameters of the logistic kernels\n";
	for(auto iter1=pt_hp.begin();iter1!=pt_hp.end();iter1++){
		for(auto iter2=iter1->begin();iter2!=iter1->end();iter2++)
			(*iter2)->print(file);
	}
	file<<"history kernel functions\n";
	for(long unsigned int d=0;d!=psi.size();d++){
		psi[d]->print(file);
		if(!post_mode_psi.empty()){
			file<<"posterior mode parameters\n";
			post_mode_psi[d]->print(file);
		}
		if(!post_mean_psi.empty()){
			file<<"posterior mean parameters\n";
			post_mean_psi[d]->print(file);
		}
	}
}


/****************************************************************************************************************************************************************************
 * Model Intensity Methods
******************************************************************************************************************************************************************************/
void GeneralizedHawkesProcess::printMatlabExpressionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda){
	matlab_lambda.clear();
	//expression for the mutually exciting part
	matlab_lambda+="(";
	std::string matlab_hawkes_lambda;
	HawkesProcess::printMatlabExpressionIntensity(mu, phi, s, matlab_hawkes_lambda);
	matlab_lambda+= matlab_hawkes_lambda;
	matlab_lambda+=")";
	//expression for the thinning part
	matlab_lambda+=".*";
	std::string matlab_thin_expr;
	printMatlabExpressionThining(pt,  psi, s, matlab_thin_expr);
	matlab_lambda+=matlab_thin_expr;
}

void GeneralizedHawkesProcess::printMatlabExpressionIntensity(const ConstantKernel * mu,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda){
	matlab_lambda.clear();
	HawkesProcess::printMatlabExpressionIntensity(mu, s, matlab_lambda);
	matlab_lambda+".*";
	std::string matlab_expr;
	printMatlabExpressionThining(pt,  psi, s,  matlab_expr);
	matlab_lambda+= matlab_expr;
}

void GeneralizedHawkesProcess::printMatlabFunctionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda_func){
	matlab_lambda_func.clear();
	matlab_lambda_func="lambda_"+s.name+" = @(t) ";//declare matlab anonymous function
	std::string matlab_lambda_expr;
	printMatlabExpressionIntensity(mu, phi, pt, psi, s, matlab_lambda_expr);
	matlab_lambda_func+=matlab_lambda_expr;
	matlab_lambda_func+=";";//terminate matlab command
}


void GeneralizedHawkesProcess::printMatlabFunctionIntensity(const ConstantKernel * mu,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda_func){
	matlab_lambda_func.clear();
	matlab_lambda_func="lambda_"+s.name+" = @(t) ";//declare matlab anonymous function
	std::string matlab_lambda_expr;
	printMatlabExpressionIntensity( mu,pt, psi, s, matlab_lambda_expr);
	matlab_lambda_func+=matlab_lambda_expr;
	matlab_lambda_func+=";";//terminate matlab command
}

void GeneralizedHawkesProcess::printMatlabExpressionThining(const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda){
	matlab_lambda.clear();
	//number of types in the process
	unsigned int d=psi.size();
	unsigned int K=pt->d/d;
	//TODO: in case of multidimensional history kernel functions, check that the order is correct: first K dimensions for type 0, etc
	matlab_lambda="(1./(1+exp(-("+std::to_string(pt->p[0])+"+";
	for(unsigned int k2=0;k2<K;k2++){//compute history effect coming from events of type k2
		for(unsigned int i=0;i<d;i++){
			matlab_lambda+=std::to_string(pt->p[k2*d+i+1])+"*(";
			for(auto iter=s.type[k2].begin();iter!=s.type[k2].end();iter++){
				std::string matlab_expr;
				psi[i]->printMatlabExpression(iter->second->time, matlab_expr);
				matlab_lambda+=matlab_expr;
				if(std::next(iter)!=s.type[k2].end())
					matlab_lambda+="+";
			}
			matlab_lambda+=")";
		}
	}
	matlab_lambda+="))))";
}


void GeneralizedHawkesProcess::printMatlabFunctionIntensities(const std::string & dir_path_str, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt,  const std::vector<Kernel *> & psi, const EventSequence &s){
	//std::ostream & matlab_file, const ConstantKernel * mu,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s

	unsigned int K=mu.size();
	for(unsigned int k=0;k!=K;k++){
		//get excitation effect on type k
		std::vector<Kernel *> phi_k;
		for(auto iter=phi.begin();iter!=phi.end();iter++){
			phi_k.push_back((*iter)[k]);
		}
		//get matlab function expression for the intensity of type k
		std::string matlab_func_lambda;
		printMatlabFunctionIntensity(mu[k],  phi_k, pt[k], psi, s, matlab_func_lambda);
		
		//create and write the function to a file
		std::string matlab_filename=dir_path_str+"/"+s.name+"_matlab_func_lambda_"+std::to_string(k)+".m";
		std::ofstream matlab_file;
		matlab_file.open(matlab_filename);
		matlab_file<<matlab_func_lambda;
		//close file
		matlab_file.close();
	}
}

double  GeneralizedHawkesProcess::computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, const EventSequence &s, double t){
	return (HawkesProcess::computeIntensity(mu,phi,s, t)*computeRealizationProbability(s, pt, t));
}

double  GeneralizedHawkesProcess::computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, const EventSequence &s, double th, double t){
	return (HawkesProcess::computeIntensity(mu,phi,s, th, t)*computeRealizationProbability(s, pt, t));
}

void GeneralizedHawkesProcess::computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & lv){
	//compute the excitation part of the process
	HawkesProcess::computeIntensity(mu,phi,s,t,lv);
	//compute the thinning probability part of the process
	for(unsigned int n=0;n<t.size();n++){
		lv(n)*=computeRealizationProbability(s, pt, t(n));//multiply the intensity by the thinning probability
     }
}

void GeneralizedHawkesProcess::computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel*  pt, const EventSequence &s, boost::numeric::ublas::vector<double>  &time_v, boost::numeric::ublas::vector<long double> & lv, unsigned int nofps){
	//take nofps time-points within the observation window of the process
	double step=(s.end_t-s.start_t)/(nofps-1);

	boost::numeric::ublas::scalar_vector<double> start_time_v(nofps,s.start_t);
	boost::numeric::ublas::vector<double> step_time_v(nofps);
	std::iota(step_time_v.begin(),step_time_v.end(), 0.0);
	if(time_v.size()<nofps)
		time_v.resize(nofps);
	time_v=start_time_v+step_time_v*step;

	computeIntensity(mu,phi,pt,s,const_cast<const boost::numeric::ublas::vector<double>&> (time_v),lv);
}

void GeneralizedHawkesProcess::computeIntensity(unsigned int k, const EventSequence &s, boost::numeric::ublas::vector<double>  &time_v, boost::numeric::ublas::vector<long double> & lv, unsigned int nofps) const {
	//get the trigerring kernel function of other types to type k
	std::vector<Kernel *> phi_k;
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		phi_k.push_back((*iter)[k]);
	}
	computeIntensity(mu[k],phi_k,pt[k],s,time_v,lv,nofps);	
}

void GeneralizedHawkesProcess::computeIntensity(unsigned int k, const EventSequence &s, const boost::numeric::ublas::vector<double>  &time_v, boost::numeric::ublas::vector<long double> & lv) const {
	//get the trigerring kernel function of other types to type k
	std::vector<Kernel *> phi_k;
	for(auto iter=phi.begin();iter!=phi.end();iter++){
		phi_k.push_back((*iter)[k]);
	}	
	computeIntensity(mu[k],phi_k,pt[k],s,time_v,lv);
}

void GeneralizedHawkesProcess::plotIntensity(const std::string &filename, unsigned int k, const EventSequence &s,  unsigned int nofps, bool  write_png_file, bool write_csv_file) const{
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
		file<<"time,value"<<std::endl;
		for(unsigned int i=0;i<tv.size();i++){
			file<<tv[i]<<","<<lv[i]<<std::endl;
		}
		file.close();
	}
}

void GeneralizedHawkesProcess::plotIntensities(const EventSequence &s,  unsigned int nofps, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	
	std::string dir_path_str=createModelDirectory();
	dir_path_str+="/intensities/";
	if(!boost::filesystem::is_directory(dir_path_str) && !boost::filesystem::create_directory(dir_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the intensity plots."<<std::endl;

	plotIntensities_(dir_path_str,s,nofps, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::plotIntensities(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	
	plotIntensities_(dir_path_str,s,nofps, write_png_file, write_csv_file);
};

void GeneralizedHawkesProcess::plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps, bool  write_png_file, bool write_csv_file) const{
	
	if(!write_png_file && !write_csv_file)
		return;
	
	for(unsigned int k=0;k<K;k++){
		//create filename for the plot
		std::string filename=dir_path_str+"intensity_"+s.name+"_"+std::to_string(k);
		plotIntensity(filename,k,s,nofps, write_png_file, write_csv_file);
	}
	
}

// static intensity methods


void GeneralizedHawkesProcess::plotIntensities(const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector < std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, unsigned int nofps, bool  write_png_file, bool write_csv_file) {
	
	if(!write_png_file && !write_csv_file)
		return;
	
	boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
	std::string curr_path_str=curr_path_boost.string();
	
	curr_path_str+="/intensities/";
	if(!boost::filesystem::is_directory(curr_path_str) && !boost::filesystem::create_directory(curr_path_str))
		std::cerr<<"Couldn't create auxiliary folder for the intensity plots."<<std::endl;

	plotIntensities_(curr_path_str, s, mu, phi, pt, nofps, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::plotIntensities(const std::string & dir_path_str, const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector < std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, unsigned int nofps, bool  write_png_file, bool write_csv_file) {
	if(!write_png_file && !write_csv_file)
		return;
	
	plotIntensities_(dir_path_str,s, mu, phi, pt, nofps, write_png_file, write_csv_file);
};

void GeneralizedHawkesProcess::plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector < std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, unsigned int nofps, bool  write_png_file, bool write_csv_file) {
	
	if(!write_png_file && !write_csv_file)
		return;
	

	unsigned int K=mu.size();
	for(unsigned int k=0;k<K;k++){
		std::vector<Kernel *> phi_k;
		for(auto iter=phi.begin();iter!=phi.end();iter++){
			phi_k.push_back((*iter)[k]);
		}	
		//create filename for the plot
		std::string filename=dir_path_str+"lambda_"+std::to_string(k);
		plotIntensity(filename,s,mu[k], phi_k, pt[k], nofps, write_png_file, write_csv_file);
	}
	
}

void GeneralizedHawkesProcess::plotIntensity(const std::string &filename, const EventSequence &s,  const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, unsigned int nofps, bool write_png_file, bool write_csv_file) {
	boost::numeric::ublas::vector<double>  tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  lv;//the value of the intensity function
	
	computeIntensity(mu,phi,pt,s,tv,lv,nofps);
	
	if(write_png_file){
		std::vector<std::string> plot_specs;
		//plot_specs.push_back("with lines ls 1 lw 2 title 'Type "+std::to_string(k)+"'");
		plot_specs.push_back("with lines ls 1 lw 2");
		Gnuplot gp;
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,s.start_t,s.end_t,"Time","Intensity");
		gp.send1d(boost::make_tuple(tv,lv));
	}

	if(write_csv_file){
		std::string csv_filename=filename+".csv";
		std::ofstream file{csv_filename};
		file<<"time,value"<<std::endl;
		for(unsigned int i=0;i<tv.size();i++){
			file<<tv[i]<<","<<lv[i]<<std::endl;
		}
		file.close();
	}
}


/****************************************************************************************************************************************************************************
 * Model Specific Methods
******************************************************************************************************************************************************************************/


double GeneralizedHawkesProcess::computeRealizationProbability(const EventSequence &s,unsigned int k, double t){
	if(k>pt.size())
		std::cerr<<"invalid type id \n";
	
	if(!pt[k])
		std::cerr<<"invalid kernel function\n";

	 return computeRealizationProbability(s,pt[k],t);
}

double GeneralizedHawkesProcess::computeRealizationProbability(const EventSequence &s,  const LogisticKernel * pt, double t){
	double  *h=0;
	if(!pt)
		std::cerr<<"invalid sigmoid kernel function\n";
	computeKernelHistory(s.type, pt, t, h);//compute the kernel history
	return pt->compute(h);//compute weighted kernel history and pass it through a sigmoid
}

double GeneralizedHawkesProcess::computeRealizationProbability(const EventSequence &s,  const LogisticKernel * pt, double th, double t){
	double  *h=0;
	if(!pt)
		std::cerr<<"invalid sigmoid kernel function\n";
	computeKernelHistory(s.type, pt, th, t, h);//compute the kernel history
	return pt->compute(h);//compute weighted kernel history and pass it through a sigmoid
}

//compute the kernel history effect of the sequence in hs at time t and store it in h
void GeneralizedHawkesProcess::computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, const LogisticKernel *pt, double t, double * &h){
	if(!pt)
		std::cerr<<"invalid sigmoid kernel function\n";
	h=new double[pt->d]();//it will keep the temporal history up to time t passed from the kernel functions
	
	if(!h)
		std::cerr<<"history vector for an event not properly allocated\n";
	unsigned int K=hs.size()-1;//number of event types (-1 due to the virtual event which has an additional type)
	unsigned int d=pt->d/K;//number of kernel functions which correspond to influence of each type

	for(unsigned int k2=0;k2<K;k2++){//compute history effect coming from events of type k2
		for(auto e_iter_2=hs[k2].begin();e_iter_2!=hs[k2].end();e_iter_2++){//find all events of type k2 which occured prior to time t
			if(!e_iter_2->second)
				std::cerr<<"invalid event\n";
			 if(e_iter_2->second->time>=t)
				 break;
			 else{
				 for(unsigned int i=0;i<d;i++){
					if(pt->psi.size()<i)
						std::cerr<<"wrong dimension of basis kernel functions\n";
					if(!pt->psi[i])
						std::cerr<<"invalid basis kernel function\n";
					h[k2*d+i]+=pt->psi[i]->compute(t,e_iter_2->second->time);//in h the first entries correspond to effect from type 0, the next from type 1 etc
				 }
			 }
		}
		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"effect of events of type "<<k2<<" at time "<<t<<" just computed. current history: "<<h[k2]<<std::endl;
		pthread_mutex_unlock(&debug_mtx);
		#endif

	}
}


//compute the kernel history effect of events up to time th of the sequence in hs at time t and store it in h
void GeneralizedHawkesProcess::computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, const LogisticKernel *pt, double th, double t, double * &h){
	if(!pt)
		std::cerr<<"invalid sigmoid kernel function\n";
	h=new double[pt->d]();//it will keep the temporal history up to time t passed from the kernel functions
	
	if(!h)
		std::cerr<<"history vector for an event not properly allocated\n";
	unsigned int K=hs.size()-1;//number of event types (-1 due to the virtual event which has an additional type)
	unsigned int d=pt->d/K;//number of kernel functions which correspond to influence of each type

	for(unsigned int k2=0;k2<K;k2++){//compute history effect coming from events of type k2
		for(auto e_iter_2=hs[k2].begin();e_iter_2!=hs[k2].end();e_iter_2++){//find all events of type k2 which occured prior to time t
			if(!e_iter_2->second)
				std::cerr<<"invalid event\n";
			 if(e_iter_2->second->time>=th)
				 break;
			 else{
	
				 for(unsigned int i=0;i<d;i++){
					if(pt->psi.size()<i)
						std::cerr<<"wrong dimension of basis kernel functions\n";
					if(!pt->psi[i])
						std::cerr<<"invalid basis kernel function\n";
					h[k2*d+i]+=pt->psi[i]->compute(t,e_iter_2->second->time);
	
				 }
		
			 }
		}
		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"effect of events of type "<<k2<<" at time "<<t<<" just computed. current history: "<<h[k2]<<std::endl;
		pthread_mutex_unlock(&debug_mtx);
		#endif
	}
}

//compute the kernel history effect of the sequence in hs on the sequence s of events of type k and store it in h
void GeneralizedHawkesProcess::computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, std::map<double,Event *> & s, unsigned int k, double ** &h, unsigned int nof_threads, int thread_id, int run_id){
	
	if(pt.size()<k||!pt[k])
		std::cerr<<"invalid kernel function to type "<<std::endl;
	
	unsigned int N=s.size();//number of events whose history will be computed TODO:is virtual event included???
	h=new double*[N];
	if(!h)
		std::cerr<<"distribute across events: history vector not properly allocated\n";
	
	pthread_t mcmc_threads[nof_threads];
	unsigned int nof_events_thread[nof_threads];
	
	//distribute the events across threads
	unsigned int nof_events_thread_q=N/nof_threads;
	unsigned int nof_events_thread_m=N%nof_threads;
	for(unsigned int t=0;t<nof_threads;t++){
		nof_events_thread[t]=nof_events_thread_q;
		if(t<nof_events_thread_m)
			nof_events_thread[t]++;
	}
	
	std::vector<Event *>s_v;
	map_to_vector(s,s_v);//vectorize events in s so that the threads get O(1) access
	unsigned int event_id_offset=0;//offset refers to the position in the vector s_v
	
	for(unsigned int child_thread_id=0;child_thread_id<nof_threads;child_thread_id++){
		if(!nof_events_thread[child_thread_id])
			break;
						
		computeKernelHistoryThreadParams_* p;
		//use the history of realized and thinned events which correspond to type k2
		p= new computeKernelHistoryThreadParams_(thread_id, this, run_id, event_id_offset,nof_events_thread[child_thread_id], hs,s_v,k,h);
		int rc = pthread_create(&mcmc_threads[child_thread_id], NULL, computeKernelHistory__, (void *)p);
		if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
		event_id_offset+=nof_events_thread[child_thread_id];
	
	}
	
	//wait for the history of all events to be computed
	for (unsigned int t = 0; t <nof_threads; t++){
		if(nof_events_thread[t])
			pthread_join (mcmc_threads [t], NULL);
	}
}
	
//compute kernel history based on the effect of events of one type for a certain range
void *GeneralizedHawkesProcess::computeKernelHistory__(void *p){
	
	std::unique_ptr<computeKernelHistoryThreadParams_ > params(static_cast< computeKernelHistoryThreadParams_* >(p));
	int thread_id=params->thread_id;
	unsigned int run_id=params->run;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int event_id_offset=params->event_id_offset;
	unsigned int nof_events=params->nof_events;
	const std::vector<std::map<double,Event *>> & hs=params->hs;
	const std::vector<Event *> & s=params->s;
	int k=params->k;
	double **h=params->h;
	if(!h)
		std::cerr<<"inside thread. distribute history across events: history vector not properly allocated\n";
		
	for(unsigned int event_id=event_id_offset;event_id<event_id_offset+nof_events;event_id++){ //todo: per event of type k parallelism
		//compute the history on each event of s separately, TODO: put threads here: convert it to vector to get O(1) access to the events, pass this vector to the threads
		if(!s[event_id])
			std::cerr<<"invalid event\n";
		if(s[event_id]->type!=k)
			std::cerr<<"invalid event type\n";
		if(thread_id>=0)
			pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
		ghp->computeKernelHistory(hs,ghp->mcmc_state[run_id].pt[k],s[event_id]->time,h[event_id]);//TODO: with the true values?????? pass pt as argument in all cases!!!!s
		if(thread_id>=0)
			pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);

	}
	return 0;
}

//compute the kernel history effect of the sequence in hs on the multivariate sequence s and store it in h
void GeneralizedHawkesProcess::computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, std::vector<std::map<double,Event *>> & s, double *** &h, unsigned int nof_threads,  int thread_id, int run){

	//synchronize since this operation changes the mcmc state
	pthread_mutex_lock(save_samples_mtx[thread_id]);
	h= new double **[K];
	if(!h)
		std::cerr<<"history vectory for event sequence not properly allocated\n";
	pthread_mutex_unlock(save_samples_mtx[thread_id]);

	
	//distribute types of events whose history will be computed across threads
	pthread_t mcmc_iter_threads[nof_threads];
	unsigned int nof_types_thread[nof_threads];
	
	unsigned int nof_types_thread_q=K/nof_threads;
	unsigned int nof_types_thread_m=K%nof_threads;
	
	for(unsigned int t=0;t<nof_threads;t++){
		nof_types_thread[t]=nof_types_thread_q;
		if(t<nof_types_thread_m)
			nof_types_thread[t]++;
	}
	 
	unsigned int type_id_offset=0;
	
	for(unsigned int child_thread_id=0;child_thread_id<nof_threads;child_thread_id++){
		if(!nof_types_thread[child_thread_id])
			break;
	
		computeKernelHistoryThreadParams* p;
		p= new computeKernelHistoryThreadParams(thread_id, this, run, type_id_offset,nof_types_thread[child_thread_id], hs, s, h, nof_threads);
		int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, computeKernelHistory_, (void *)p);
		if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
		type_id_offset+=nof_types_thread[child_thread_id];
	
	}
	
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}

}

void *GeneralizedHawkesProcess::computeKernelHistory_(void *p){
	std::unique_ptr<computeKernelHistoryThreadParams > params(static_cast< computeKernelHistoryThreadParams * >(p));
	int thread_id=params->thread_id;
	int run_id=params->run;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int nof_threads=params->nof_threads;
	const std::vector<std::map<double,Event *>> & hs=params->hs;
	std::vector<std::map<double,Event *>> & s=params->s;

	double ***h=params->h;
	
	if(!h)//TODO: better check of the pointer
		std::cerr<<"history vector not properly allocated\n";

	if(s.size()<type_id_offset+nof_types)
		std::cerr<<"invalid sequence\n";

	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){//this drops the virtual event since it has an additional type TODO: put threads here per type of thinned events parallelism
		ghp->computeKernelHistory(hs,s[k],k,h[k], nof_threads, thread_id, run_id);//compute the history of events of each type separately;
	}
	
	return 0;
}


void GeneralizedHawkesProcess:: sampleThinnedEvents(EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt){
	
	unsigned int K=mu.size();
	double start_t=seq.start_t;
	double end_t=seq.end_t;

	seq.flushThinnedEvents();
	Event *ve=seq.full.begin()->second;
	for(unsigned int k=0;k<K;k++){	//each loop samples thinned events of type k
		std::vector<double> arrival_times=simulateHPoisson(mu[k]->p[0],start_t,end_t);
		unsigned int Nt=arrival_times.size();//candidate thinned events of type k which belong to the background process
		//thin the background process 
		for(unsigned i=0;i<Nt;i++){
					
			//decide whether to thin or not the event
			double p_t=computeRealizationProbability(seq,pt[k],arrival_times[i]);
				
			std::random_device rd;
			std::mt19937 gen(rd());
			std::bernoulli_distribution bern_distribution(1-p_t);
			bool b_t=bern_distribution(gen);
			if(b_t){//the event is a thinned event, otherwise it is discarded
				Event * e=new Event{seq.type[k].size(),(int)k,arrival_times[i],K,false, ve};
				seq.addEvent(e,ve);
			}
		}

		if(!phi.empty() &&seq.full.size()>=2 ){//if there is mutual excitation part in the model
			for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){//skip the virtual event which generates the exogenous intensity (whose thinned events were simulated above)
				Event *e=e_iter->second;
				std::vector<double> arrival_times=simulateNHPoisson(*phi[e->type][k],e->time, e->time, end_t);
				unsigned int Nt=arrival_times.size();
				
				for(unsigned i=0;i<Nt;i++){
					//decide whether to thin or not the event
					double p_t=computeRealizationProbability(seq,pt[k],arrival_times[i]);//todo: static method for using a logistic kernel passed as argument!!!
					std::random_device rd;
					std::mt19937 gen(rd());
					std::bernoulli_distribution bern_distribution(1-p_t);
					bool b_t=bern_distribution(gen);
					if(b_t){//the event is a thinned event, otherwise it is discarded
						Event *e2=new Event{seq.type[k].size(),(int)k,arrival_times[i],K,false, e};
						seq.addEvent(e2,e);
					}	
				}
			}
		}
	}
}


/****************************************************************************************************************************************************************************
 * Model Likelihood Methods
******************************************************************************************************************************************************************************/

//static methods
double GeneralizedHawkesProcess::loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	return seq.observed?fullLoglikelihood(seq, mu, phi, pt, normalized, t0):partialLoglikelihood(seq, mu, phi,pt, normalized, t0);
}

////likelihood of mutually regressive poisson process
double GeneralizedHawkesProcess::loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	return seq.observed?fullLoglikelihood(seq, mu, pt, normalized, t0):partialLoglikelihood(seq, mu, pt, normalized, t0);
}

double GeneralizedHawkesProcess::likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	return seq.observed?fullLikelihood(seq, mu, phi,pt, normalized, t0):partialLikelihood(seq, mu, phi,pt, normalized, t0);
}

////likelihood of mutually regressive poisson process
double GeneralizedHawkesProcess::likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	return seq.observed?fullLikelihood(seq, mu, pt, normalized, t0):partialLikelihood(seq, mu, pt, normalized, t0);
}

//likelihood of the thinning procedure
double GeneralizedHawkesProcess:: likelihood(const EventSequence & seq, const std::vector<LogisticKernel*> & pt , double t0){
	double l=1.0;
	unsigned int K=pt.size();
	if(seq.type.size()!=K+1||seq.thinned_type.size()!=K+1){
		return -1;
	}
	t0=(t0>seq.start_t)?t0:seq.start_t;
	
	//each loop adds likelihood of events of type k
	for(unsigned int k=0;k<K;k++){//account for the additional type of the virtual event in the sequence
		//likelihood of the observed events of type k
		for(auto iter=seq.type[k].begin();iter!=seq.type[k].end();iter++){
			if(iter->second->time>t0 && iter->second->time<seq.end_t){
				double p=computeRealizationProbability(seq,pt[k],iter->second->time);
				l*=p;
			}
		}

		for(auto iter=seq.thinned_type[k].begin();iter!=seq.thinned_type[k].end();iter++){
			if(iter->second->time>t0 && iter->second->time<seq.end_t){
				double p=computeRealizationProbability(seq,pt[k],iter->second->time);
				l*=(1-p);
			}
		}
	}

	return l;
}

double GeneralizedHawkesProcess:: likelihood(const EventSequence & seq, unsigned int k, LogisticKernel const * const pt, double t0){
	double l=1.0;
	
	if(seq.type.size()<k||seq.thinned_type.size()<k){
		return -1;
	}
	t0=(t0>seq.start_t)?t0:seq.start_t;
	//each loop adds likelihood of events of type k
	//account for the additional type of the virtual event in the sequence 
	//likelihood of the observed events of type k
	for(auto iter=seq.type[k].begin();iter!=seq.type[k].end();iter++){
		if(iter->second->time>t0 && iter->second->time<seq.end_t){
			double p=computeRealizationProbability(seq,pt,iter->second->time);
			l*=p;
		}
	}

	for(auto iter=seq.thinned_type[k].begin();iter!=seq.thinned_type[k].end();iter++){
		if(iter->second->time>t0 && iter->second->time<seq.end_t){
			double p=computeRealizationProbability(seq,pt,iter->second->time);
			l*=(1-p);
		}
	}

	return l;
}

double GeneralizedHawkesProcess:: likelihood(const EventSequence & seq, unsigned int k, LogisticKernel const * const pt, const std::vector<double> & polyagammas, const std::vector<double> & thinned_polyagammas, double t0){
	double l=1.0;
	
	if(seq.type.size()<k||seq.thinned_type.size()<k){
		return -1;
	}
	t0=(t0>seq.start_t)?t0:seq.start_t;
	//each loop adds likelihood of events of type k
	//account for the additional type of the virtual event in the sequence
	//likelihood of the observed events of type k
	unsigned int event_id=0;
	for(auto iter=seq.type[k].begin();iter!=seq.type[k].end();iter++){
		if(iter->second->time>t0 && iter->second->time<seq.end_t){
			double *h_e=0;
			computeKernelHistory(seq.type, pt, iter->second->time, h_e);//compute the kernel history up to the arrival time of the event
			double o_arg=pt->computeArg(h_e); //compute the argument of the sigmoid function
			l*=exp(0.5*o_arg-0.5*polyagammas[event_id++]*o_arg);
		}
	}
	
	
	event_id=0;
	for(auto iter=seq.thinned_type[k].begin();iter!=seq.thinned_type[k].end();iter++){
		if(iter->second->time>t0 && iter->second->time<seq.end_t){
			double *h_e=0;
			computeKernelHistory(seq.type, pt, iter->second->time, h_e);//compute the kernel history up to the arrival time of the event
			double o_arg=pt->computeArg(h_e);//compute the argument of the sigmoid function
			l*=exp(-0.5*o_arg-0.5*thinned_polyagammas[event_id++]*o_arg);
		}
	}

	return l;
}


//loglikelihood of the thinning procedure
double GeneralizedHawkesProcess:: loglikelihood(const EventSequence & seq, const std::vector<LogisticKernel*> & pt, double t0){
	double l=0.0;
	t0=(t0>seq.start_t)?t0:seq.start_t;

	unsigned int K=pt.size();
	if(seq.type.size()!=K+1||seq.thinned_type.size()!=K+1){
		return -1;
	}
	//each loop adds likelihood of events of type k
	for(unsigned int k=0;k<K;k++){//account for the additional type of the virtual event in the sequence
		for(auto iter=seq.type[k].begin();iter!=seq.type[k].end();iter++){
			if(iter->second->time>t0 && iter->second->time<seq.end_t){
				double p=computeRealizationProbability(seq,pt[k],iter->second->time);
				if(p==0)
					p=EPS;
				l+=log(p);
			}
		}
		
		//likelihood of the thinned events of type k
		if(seq.observed){
			
			for(auto iter=seq.thinned_type[k].begin();iter!=seq.thinned_type[k].end();iter++){
				if(iter->second->time>t0 && iter->second->time<seq.end_t){
					double p=computeRealizationProbability(seq,pt[k],iter->second->time);
					if(p==1)
						p=1-EPS;
					l+=log(1-p);
				}
			}
		}
	}

	return l;
}

//full likelihood methods, the thinned events and the parent structure is known
double GeneralizedHawkesProcess::fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	double l;
	l=HawkesProcess::fullLikelihood(seq,mu,phi, false, t0);

	return normalized?((l*likelihood(seq,pt, t0))/(seq.full.size()+seq.thinned_full.size())):(l*likelihood(seq,pt, t0));
}

//likelihood of mutually regressive poisson process
double GeneralizedHawkesProcess::fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	double l;
	l=HawkesProcess::fullLikelihood(seq,mu, false, t0);
	return normalized? ((l*likelihood(seq,pt, t0))/ (seq.full.size()+seq.thinned_full.size())):(l*likelihood(seq,pt, t0));
}

double GeneralizedHawkesProcess::fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	double l;
	if(!phi.empty())
		l=HawkesProcess::fullLoglikelihood(seq,mu, phi,false, t0);//normalize only once by the number of events in the sequence
	else
		l=HawkesProcess::fullLoglikelihood(seq,mu,false, t0);
	return normalized?(l+loglikelihood(seq,pt, t0))/ (seq.full.size()+seq.thinned_full.size()):(l+loglikelihood(seq,pt, t0));
}


double GeneralizedHawkesProcess::fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	double l;
	l=HawkesProcess::fullLoglikelihood(seq,mu, false, t0);
	return  normalized?(l+loglikelihood(seq,pt, t0))/ (seq.full.size()+seq.thinned_full.size()):(l+loglikelihood(seq,pt, t0));
}

double GeneralizedHawkesProcess::partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){

	#if LOGL_PROFILING
	boost::filesystem::path ghp_curr_path_boost=boost::filesystem::current_path();
	std::string ghp_curr_path_str=ghp_curr_path_boost.string();
	std::ofstream logl_profiling_file{ghp_curr_path_str+"/logl_profiling_"+seq.name+".txt"};
	#endif
	
	t0=(t0>seq.start_t)?t0:seq.start_t;
	//reverse the vector so that phi_k[i] refers to the intensity functions to (not from) type k
	long unsigned int K=phi.size();
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	for(long unsigned int k=0;k<K;k++){
		for(long unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];
		}
	}
	double l=0.0;
	unsigned int nof_events=0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){ 
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
		
			nof_events++;
			double c=log(computeIntensity(mu[e_iter->second->type],phi_k[e_iter->second->type],pt[e_iter->second->type], seq, e_iter->second->time));
			l+=c;
		}
		#if LOGL_PROFILING
		logl_profiling_file<<"hawkes intensity of event "<<HawkesProcess::computeIntensity(mu[e_iter->second->type],phi_k[e_iter->second->type],seq, e_iter->second->time)<<std::endl;
		logl_profiling_file<<"realization probability "<<computeRealizationProbability(seq, pt[e_iter->second->type], e_iter->second->time)<<std::endl;
		logl_profiling_file<<"contribution of event "<<e_iter->second->time<<" of type "<<e_iter->second->type<<" "<<c<<std::endl;
		#endif   
	}
	#if LOGL_PROFILING
	logl_profiling_file<<"contribution of events "<<l<<std::endl;
	#endif

	auto f=[&](double t)->double{
			double s=0;
			for(unsigned int k=0;k<K;k++)
				s+=computeIntensity(mu[k],phi_k[k], pt[k], seq, t); 
			
			return s;
	};
	double i=monte_carlo_integral(f, t0>seq.start_t?t0:seq.start_t, seq.end_t);
	#if LOGL_PROFILING
	logl_profiling_file<<"contribution of the process "<<-i<<std::endl;
	logl_profiling_file<<"normalized log l: "<<(l-i)/(seq.full.size()-1)<<std::endl;
	logl_profiling_file<<"unnomralized log l "<<(l-i)<<std::endl;
	logl_profiling_file.close();
	#endif 

	return normalized?(l-i)/(nof_events):(l-i);
}

double  GeneralizedHawkesProcess::partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	unsigned int K=mu.size();
	#if LOGL_PROFILING
	boost::filesystem::path ghp_curr_path_boost=boost::filesystem::current_path();
	std::string ghp_curr_path_str=ghp_curr_path_boost.string();
	std::ofstream logl_profiling_file{"logl_profiling_"+seq.name+".txt"};
	#endif

	t0=(t0>seq.start_t)?t0:seq.start_t;

	std::vector<Kernel *> phi_k;
	double l=0.0;
	unsigned int nof_events=0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
			double c=log(computeIntensity(mu[e_iter->second->type],phi_k,pt[e_iter->second->type], seq, e_iter->second->time));
			l+=c;
			nof_events++;
		}
		#if LOGL_PROFILING
		std::vector<Kernel *> phi;
		logl_profiling_file<<"contribution of event "<<e_iter->second->time<<" of type "<<e_iter->second->type<<" "<<c<<std::endl;
		logl_profiling_file<<"hawkes intensity of event "<<HawkesProcess::computeIntensity(mu[e_iter->second->type],phi,seq, e_iter->second->time)<<std::endl;
		logl_profiling_file<<"realization probability "<<computeRealizationProbability(seq, pt[e_iter->second->type], e_iter->second->time)<<std::endl;
		#endif   
	}

	#if LOGL_PROFILING
	logl_profiling_file<<"contribution of events "<<l<<std::endl;
	#endif
	
	auto f=[&](double t)->double{
			double s=0;
			for(unsigned int k=0;k<K;k++)
				s+=computeIntensity(mu[k],phi_k, pt[k],seq, t); 
			
			return s;
	};
	double i=monte_carlo_integral(f, t0>seq.start_t?t0:seq.start_t, seq.end_t);
	#if LOGL_PROFILING
	logl_profiling_file<<"contribution of the process "<<-i<<std::endl;
	logl_profiling_file<<"normalized log l: "<<(l-i)/(seq.full.size()-1)<<std::endl;
	logl_profiling_file<<"unnomralized log l "<<(l-i)<<std::endl;
	logl_profiling_file.close();
	#endif 

	return normalized?(l-i)/(nof_events):(l-i);
}


double GeneralizedHawkesProcess::partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){

	t0=(t0>seq.start_t)?t0:seq.start_t;
	
	//reverse the vector so that phi_k[i] refers to the intensity functions to (not from) type k
	long unsigned int K=phi.size();
	unsigned int nof_events=0;
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	for(long unsigned int k=0;k<K;k++){
		for(long unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];//TODO: this is not very good design, maybe I should have stored trigering kernels in reversed order in the first place??
		}
	}

	double l=1.0;
	for(auto e_iter=std::next(seq.full.begin());e_iter!=seq.full.end();e_iter++){//skip the virtual event
		if(e_iter->second->time>t0 && e_iter->second->time<seq.end_t){
			nof_events++;
			l*=(computeIntensity(mu[e_iter->second->type],phi_k[e_iter->second->type],pt[e_iter->second->type], seq,e_iter->second->time));
		}
	}
	
	auto f=[&](double t)->double{
			double s=0;
			for(unsigned int k=0;k<K;k++)
				s+=computeIntensity(mu[k], phi_k[k], pt[k], seq, t); 
			
			return s;
	};
	double i=monte_carlo_integral(f, t0>seq.start_t?t0:seq.start_t, seq.end_t);
	return (normalized?l*exp(-i)/(nof_events):(l*exp(-i)));
}


double  GeneralizedHawkesProcess::partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<LogisticKernel*> & pt, bool normalized, double t0){
	//assign the events to the virtual event (of the exogenous intensity)
	EventSequence seq_copy=seq;
	Event *ve=seq_copy.full.begin()->second; //the virtual event which corresponds to the background process
	for(auto e_iter=std::next(seq_copy.full.begin());e_iter!=seq_copy.full.end();e_iter++){
		seq_copy.setParent(e_iter->second,ve);
	}
	return fullLikelihood(seq_copy, mu, pt, normalized, t0);
}


//non-static methods
double GeneralizedHawkesProcess::likelihood(const EventSequence & seq, bool normalized, double t0) const{
	return  !phi.empty()?likelihood(seq,mu,phi,pt, normalized):likelihood(seq,mu,pt,normalized, t0);
}

double GeneralizedHawkesProcess::loglikelihood(const EventSequence & seq, bool normalized, double t0) const{
	return !phi.empty()?loglikelihood(seq,mu,phi,pt, normalized, t0):loglikelihood(seq,mu,pt,normalized, t0);
}


void GeneralizedHawkesProcess::posteriorModeLoglikelihood(EventSequence & seq, std::vector<double> & logl, bool normalized, double t0) {
	setPosteriorParams();	
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
			logl.push_back(phi.empty()?HawkesProcess::loglikelihood(*seq_iter, post_mode_mu, normalized, t0):HawkesProcess::loglikelihood(seq, post_mode_mu,post_mode_phi, normalized, t0));

 
}
/****************************************************************************************************************************************************************************
 * Goodness-of-fit Methods
******************************************************************************************************************************************************************************/

void GeneralizedHawkesProcess::goodnessOfFitMatlabScript(const std::string & dir_path_str, const std::string & seq_path_dir, std::string model_name, std::vector<EventSequence> & data, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> pt, const std::vector<Kernel *> & psi) {
	
	std::string matlab_script_name=dir_path_str+"/"+model_name+"_kstest.m";
	std::ofstream matlab_script;
	matlab_script.open(matlab_script_name);

	matlab_script<<"%intensity  functions which correspond to the spike trains and with posterior mode estimates"<<std::endl;

	unsigned int K=mu.size();
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
			//printMatlabExpressionIntensity(mu[k], phi_k, pt[k], psi,*seq_iter, matlab_lambda_expr); // matlab expression for intensity which corresonds to the sequence 
			if(phi_k.empty())
				printMatlabExpressionIntensity(mu[k], pt[k], psi, *seq_iter, matlab_lambda_expr); // matlab expression for intensity which corresonds to the sequence 
			else
				printMatlabExpressionIntensity(mu[k], phi_k, pt[k], psi, *seq_iter, matlab_lambda_expr); // matlab expression for intensity which corresonds to the sequence 
			
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


void GeneralizedHawkesProcess::fit(const MCMCParams * mcmc_params){
	GeneralizedHawkesProcessMCMCParams * ghp_mcmc_params=(GeneralizedHawkesProcessMCMCParams *) mcmc_params;
	if(train_data.empty()) return;
	bool empty_sequences_flag=true;
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
		empty_sequences_flag=!(seq_iter->N>2);//even if there's an event it is the virtual event. there should be at least two events
	//all the event sequences provided for training are degenerate
	if(empty_sequences_flag)
		return;
	if(ghp_mcmc_params->dir_path_str.empty()){
		boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
		std::string curr_path_str=curr_path_boost.string();
		if(!name.empty())
			ghp_mcmc_params->dir_path_str=curr_path_str+"/"+name+"/";
		else
			ghp_mcmc_params->dir_path_str=curr_path_str+"/ghp/";
		if(!boost::filesystem::is_directory(ghp_mcmc_params->dir_path_str) && !boost::filesystem::create_directory(ghp_mcmc_params->dir_path_str))
			std::cerr<<"Couldn't create auxiliary folder."<<std::endl;

	}

	fit_(ghp_mcmc_params);
	
}

void GeneralizedHawkesProcess::fit_(GeneralizedHawkesProcessMCMCParams const * const ghp_mcmc_params){
	
	//initialize the mcmc state (the auxiliary structures), generate the initial mcmc states
	if(mcmc_state.empty())
		mcmcInit(ghp_mcmc_params);
	
	//initialize the structures for the inference time if necessary
	if(ghp_mcmc_params->profiling && profiling_mcmc_step.empty()){
		initProfiling(ghp_mcmc_params->runs,ghp_mcmc_params->max_num_iter);
	}
	
	mcmcSynchronize(ghp_mcmc_params->mcmc_fit_nof_threads);

	pthread_t mcmc_run_threads[ghp_mcmc_params->mcmc_fit_nof_threads];
	unsigned int nof_runs_thread[ghp_mcmc_params->mcmc_fit_nof_threads];

	unsigned int nof_runs_thread_q=ghp_mcmc_params->runs/ghp_mcmc_params->mcmc_fit_nof_threads;
	unsigned int nof_runs_thread_m=ghp_mcmc_params->runs%ghp_mcmc_params->mcmc_fit_nof_threads;
	
	
	unsigned int fit_nof_threads_created=0;
	for(unsigned int t=0;t<ghp_mcmc_params->mcmc_fit_nof_threads;t++){
		nof_runs_thread[t]=nof_runs_thread_q;
		if(t<nof_runs_thread_m)
			nof_runs_thread[t]++;
		if(nof_runs_thread[t])
			fit_nof_threads_created++;
	}
	 
	//split the runs across the threads
	unsigned int run_id_offset=0;
	for(unsigned int thread_id=0;thread_id<ghp_mcmc_params->mcmc_fit_nof_threads;thread_id++){
	    if(!nof_runs_thread[thread_id]){
	    	break;
	    }
		InferenceThreadParams* p;
		p= new InferenceThreadParams(thread_id, this,run_id_offset,nof_runs_thread[thread_id], ghp_mcmc_params->nof_burnin_iters, ghp_mcmc_params->max_num_iter,ghp_mcmc_params->mcmc_iter_nof_threads, ghp_mcmc_params->profiling, ghp_mcmc_params->dir_path_str,fit_nof_threads_created, &nof_fit_threads_done, &fit_mtx, &fit_con);
		if(!p)
			std::cerr<<"Unable to allocate memory for the thread\n";
	    int rc = pthread_create(&mcmc_run_threads[thread_id], NULL, GeneralizedHawkesProcess::fit_, (void *)p);
	    if (rc){
	         std::cerr << "Error:unable to create thread," << rc << std::endl;
	         exit(-1);
	    }
	    run_id_offset+=nof_runs_thread[thread_id];	
	}

	
	mcmcPolling(ghp_mcmc_params->dir_path_str, fit_nof_threads_created);	
	//wait until all the mcmc threads have finished
	for (unsigned int i = 0; i < fit_nof_threads; i++){
	    if(nof_runs_thread[i]){
	    	pthread_join (mcmc_run_threads [i], NULL);
	    }
	}
	if(ghp_mcmc_params->profiling){//log the learning time for the various parts of the inference
		writeProfiling(ghp_mcmc_params->dir_path_str);
	}
}

//periodically save model and posterior samples, while waiting for the generated threads to finish
void GeneralizedHawkesProcess::mcmcPolling(const std::string &dir_path_str, unsigned int fit_nof_threads){
	
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
		for(unsigned int i=0;i!=fit_nof_threads;i++){
			pthread_mutex_lock(save_samples_mtx[i]);
			pthread_mutex_lock(HawkesProcess::save_samples_mtx[i]);
		}
		save(model_file);
		for(unsigned int i=0;i!=fit_nof_threads;i++){
			pthread_mutex_unlock(save_samples_mtx[i]);
			pthread_mutex_unlock(HawkesProcess::save_samples_mtx[i]);
		}
	
	}
	pthread_mutex_unlock(&fit_mtx);
}

void GeneralizedHawkesProcess::initProfiling(unsigned int runs,  unsigned int max_num_iter){
	//learning steps for the inhibition part
	profiling_mcmc_step.resize(runs);
	profiling_thinned_events_sampling.resize(runs);
	profiling_polyagammas_sampling.resize(runs);
	profiling_logistic_sampling.resize(runs);
	profiling_hawkes_sampling.resize(runs);
	profiling_compute_history.resize(runs);
	profiling_normal_model_sampling.resize(runs);
	
	//learning steps for the excitation part
	HawkesProcess::profiling_mcmc_step.resize(runs);
	profiling_parent_sampling.resize(runs);
	profiling_intensity_sampling.resize(runs);
	
	for(unsigned int i=0;i<runs;i++){
		profiling_mcmc_step[i].resize(max_num_iter);
		profiling_thinned_events_sampling[i].resize(max_num_iter);
		profiling_polyagammas_sampling[i].resize(max_num_iter);
		profiling_logistic_sampling[i].resize(max_num_iter);
		profiling_hawkes_sampling[i].resize(max_num_iter);
		profiling_compute_history[i].resize(max_num_iter);
		HawkesProcess::profiling_mcmc_step[i].resize(max_num_iter);
		profiling_parent_sampling[i].resize(max_num_iter);
		profiling_intensity_sampling[i].resize(max_num_iter);
		profiling_normal_model_sampling[i].resize(max_num_iter);
	}
}

void GeneralizedHawkesProcess::writeProfiling(const std::string &dir_path_str) const {

	unsigned int runs=profiling_mcmc_step.size();
	unsigned int max_num_iter=profiling_mcmc_step[0].size();

	//open profiling files
	std::string profiling_total_str= dir_path_str+"ghp_infer_total.csv";
	std::ofstream profiling_total_file{profiling_total_str};
	profiling_total_file<<"mcmc step, learning time"<<std::endl;

	std::string profiling_thin_str= dir_path_str+"infer_thinned_events_sampling.csv";
	std::ofstream profiling_thin_file{profiling_thin_str};
	profiling_thin_file<<"mcmc step, learning time"<<std::endl;
	
	std::string profiling_ch_str= dir_path_str+"infer_compute_history.csv";
	std::ofstream profiling_ch_file{profiling_ch_str};
	profiling_ch_file<<"mcmc step, learning time"<<std::endl;

	std::string profiling_pg_str= dir_path_str+"infer_polyagammas_sampling.csv";
	std::ofstream profiling_pg_file{profiling_pg_str};
	profiling_pg_file<<"mcmc step, learning time"<<std::endl;

	std::string profiling_hp_str= dir_path_str+"infer_hawkes_sampling.csv";
	std::ofstream profiling_hp_file{profiling_hp_str};
	profiling_hp_file<<"mcmc step, learning time"<<std::endl;

	std::string profiling_lg_str= dir_path_str+"infer_logistic_sampling.csv";
	std::ofstream profiling_lg_file{profiling_lg_str};
	profiling_lg_file<<"mcmc step, learning time"<<std::endl;
	
	std::string profiling_nm_str= dir_path_str+"infer_normal_model_sampling.csv";
	std::ofstream profiling_nm_file{profiling_nm_str};
	profiling_nm_file<<"mcmc step, learning time"<<std::endl;

	//get the mean inference time across the runs
	for(unsigned int iter=0;iter<max_num_iter;iter++){
		double total_mcmc_step=0;
		double total_th_sampling=0;
		double total_pg_sampling=0;
		double total_hp_sampling=0;
		double total_lg_sampling=0;
		double total_ch_sampling=0;
		double total_nm_sampling=0;
		for(unsigned int r=0;r<runs;r++){
			total_mcmc_step+=profiling_mcmc_step[r][iter];
			total_th_sampling+=profiling_thinned_events_sampling[r][iter];
			total_pg_sampling+=profiling_polyagammas_sampling[r][iter];
			total_hp_sampling+=profiling_hawkes_sampling[r][iter];
			total_lg_sampling+=profiling_logistic_sampling[r][iter];
			total_ch_sampling+=profiling_compute_history[r][iter];
			total_nm_sampling+=profiling_normal_model_sampling[r][iter];
		}
		total_mcmc_step/=runs;
		total_th_sampling/=runs;
		total_pg_sampling/=runs;
		total_hp_sampling/=runs;
		total_lg_sampling/=runs;
		total_ch_sampling/=runs;
		total_nm_sampling/=runs;

		profiling_total_file<<iter<<","<<total_mcmc_step<<std::endl;
		profiling_thin_file<<iter<<","<<total_th_sampling<<std::endl;
		profiling_pg_file<<iter<<","<<total_pg_sampling<<std::endl;
		profiling_hp_file<<iter<<","<<total_hp_sampling<<std::endl;
		profiling_lg_file<<iter<<","<<total_lg_sampling<<std::endl;
		profiling_ch_file<<iter<<","<<total_ch_sampling<<std::endl;
		profiling_nm_file<<iter<<","<<total_nm_sampling<<std::endl;
	}
	
	//close the profiling files
	profiling_total_file.close();
	profiling_thin_file.close();
	profiling_pg_file.close();
	profiling_hp_file.close();
	profiling_lg_file.close();
	profiling_ch_file.close();
	
	HawkesProcess::writeProfiling(dir_path_str);
}

void* GeneralizedHawkesProcess::fit_( void *p){
	std::unique_ptr< InferenceThreadParams > params( static_cast< InferenceThreadParams * >( p ) );
	GeneralizedHawkesProcess *ghp=params->ghp;

//    std::ofstream full_log_seq_file{params->infer_dir_path_str+"train_full_logl_sequence.csv"};//todo:handle more than event sequences
//    std::ofstream partial_log_seq_file{params->infer_dir_path_str+"train_partial_logl_sequence.csv"};
    

    std::ofstream test_log_seq_file;
//	if(!ghp->test_data.empty())
//		test_log_seq_file.open(params->infer_dir_path_str+"test_partial_logl_sequence.csv",std::ofstream::out);//todo:handle more than event sequences
//    
    
	//unsigned int iter_nof_threads=params->fit_nof_threads;
	for(unsigned int mcmc_run=params->run_id_offset;mcmc_run<params->run_id_offset+params->runs;mcmc_run++){
		//sample the first mcmc state and burn-in, if there are no posterior samples already
		if(!ghp->mcmc_state[mcmc_run].mcmc_iter){
			ghp->mcmcStartRun(params->thread_id, mcmc_run, params->iter_nof_threads);

			#if DEBUG_LEARNING
			pthread_mutex_lock(&debug_mtx);
			debug_learning_file<<"thread "<<params->thread_id<<" run "<<mcmc_run<<std::endl;
			pthread_mutex_unlock(&debug_mtx);
			#endif


			for(unsigned int mcmc_iter=0;mcmc_iter!=params->burnin_iter;mcmc_iter++){
				ghp->mcmcStep(params->thread_id,mcmc_run,mcmc_iter,false,params->iter_nof_threads,0, params->profiling);//discard the posterior samples

				ghp->mcmc_state[params->thread_id].mcmc_iter++;
				ghp->HawkesProcess::mcmc_state[params->thread_id].mcmc_iter++;
				
//				for(long unsigned int seq_id=0;seq_id!=ghp->HawkesProcess::mcmc_state[mcmc_run].seq.size();seq_id++){
//					double logl=GeneralizedHawkesProcess::loglikelihood(ghp->HawkesProcess::mcmc_state[mcmc_run].seq[seq_id], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//					full_log_seq_file<<mcmc_iter<<","<<logl<<std::endl;
//					logl=GeneralizedHawkesProcess::loglikelihood(ghp->train_data[seq_id], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//					partial_log_seq_file<<mcmc_iter<<","<<logl<<std::endl;						
//				}
//				for(long unsigned int seq_id=0;seq_id!=ghp->test_data.size();seq_id++){
//					double test_logl=GeneralizedHawkesProcess::loglikelihood(ghp->test_data[0], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//					test_log_seq_file<<mcmc_iter<<","<<test_logl<<std::endl;
//				}
			}
		}
		
		for(unsigned int mcmc_iter=ghp->mcmc_state[mcmc_run].mcmc_iter;mcmc_iter!=params->max_num_iter;mcmc_iter++){
			
			std::cout<<"mcmc iter "<<mcmc_iter<<std::endl;
			ghp->mcmcStep(params->thread_id,mcmc_run,mcmc_iter,true,params->iter_nof_threads, 0, params->profiling);//save the posterior samples
			
			ghp->mcmc_state[mcmc_run].mcmc_iter++;//careful: first save the samples, complete the update of the full mcmc_state then increase number of iter
			//so that if it is loaded from serialized file it starts from the correct iteration number (with the current mcmc state complete)
			ghp->HawkesProcess::mcmc_state[mcmc_run].mcmc_iter++;
			
//			for(long unsigned int seq_id=0;seq_id!=ghp->HawkesProcess::mcmc_state[mcmc_run].seq.size();seq_id++){
//				double logl=GeneralizedHawkesProcess::loglikelihood(ghp->HawkesProcess::mcmc_state[mcmc_run].seq[seq_id], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//				full_log_seq_file<<mcmc_iter<<","<<logl<<std::endl;
//				logl=GeneralizedHawkesProcess::loglikelihood(ghp->train_data[seq_id], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//				partial_log_seq_file<<mcmc_iter<<","<<logl<<std::endl;						
//			}
//			for(long unsigned int seq_id=0;seq_id!=ghp->test_data.size();seq_id++){
//				double test_logl=GeneralizedHawkesProcess::loglikelihood(ghp->test_data[0], ghp->HawkesProcess::mcmc_state[mcmc_run].mu, ghp->HawkesProcess::mcmc_state[mcmc_run].phi, ghp->mcmc_state[mcmc_run].pt, true);
//				test_log_seq_file<<mcmc_iter<<","<<test_logl<<std::endl;
//			}

		}
		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"done. thread "<<params->thread_id<<" run "<<mcmc_run<<std::endl;
		pthread_mutex_unlock(&debug_mtx);
		#endif
	}

	pthread_mutex_lock(params->fit_mtx);
	(*(params->nof_fit_threads_done))++;
	#if DEBUG_LEARNING
	pthread_mutex_lock(&debug_mtx);
	debug_learning_file<<"thread "<<params->thread_id<<"done threads "<<(*(params->nof_fit_threads_done))<<std::endl;
	pthread_mutex_unlock(&debug_mtx);
	#endif
	if((*(params->nof_fit_threads_done))==params->fit_nof_threads){//if it's the last inference thread, signal the main thread otherwise leave the unlock for the sibling threads
	#if DEBUG_LEARNING
	pthread_mutex_lock(&debug_mtx);
	debug_learning_file<<"thread "<<params->thread_id<<" signal main thread."<<std::endl;
	pthread_mutex_unlock(&debug_mtx);
	#endif
		pthread_cond_signal(params->fit_con);
	}
	pthread_mutex_unlock(params->fit_mtx);
	return 0;
}


void GeneralizedHawkesProcess::mcmcInit(MCMCParams const * const mcmc_params/*unsigned int mcmc_runs, unsigned int nof_sequences_per_run, unsigned int mcmc_burnin_iter, unsigned int mcmc_max_num_iter, unsigned int fit_nof_threads*/){
	
	//initialize the structures for storing the posterior samples of the precision
	if(!pt_hp.empty()){
		tau_posterior_samples.resize(K);
		for(unsigned int k=0;k<K;k++){
			unsigned int nofp=pt[k]->nofp;
			tau_posterior_samples[k].resize(nofp);
			for(unsigned int p=0;p<nofp;p++){
				tau_posterior_samples[k][p].resize(mcmc_params->runs);
				for(unsigned int r=0;r<mcmc_params->runs;r++)
					tau_posterior_samples[k][p][r].resize(mcmc_params->max_num_iter);
			}
		}
	}
	
	//initialize the structures for storing the posterior samples of the thinning weights
	pt_posterior_samples.resize(K);

	for(unsigned int k=0;k<K;k++){
		unsigned int nofp=pt[k]->nofp;
		pt_posterior_samples[k].resize(nofp);
		for(unsigned int p=0;p<nofp;p++){
			pt_posterior_samples[k][p].resize(mcmc_params->runs);
			for(unsigned int r=0;r<mcmc_params->runs;r++)
				pt_posterior_samples[k][p][r].resize(mcmc_params->max_num_iter);
		}
	}


	//initialize the structures for storing the posterior samples of the history kernels
	
	psi_posterior_samples.resize(psi.size());
	for(long unsigned int d=0;d<psi.size();d++){
		psi_posterior_samples[d].resize(psi[d]->nofp);
		for(unsigned int p=0;p<psi[d]->nofp;p++){
				psi_posterior_samples[d][p].resize(mcmc_params->runs);
				for(unsigned int r=0;r<mcmc_params->runs;r++)
					psi_posterior_samples[d][p][r].resize(mcmc_params->max_num_iter);
		}
	}

	//initialize the mcmc states for each thread of the inference
	mcmc_state.resize(mcmc_params->runs);
			
			
	for(unsigned int run_id=0;run_id<mcmc_params->runs;run_id++){
		State &s=mcmc_state[run_id];
	    s.K=K;
	    s.pt_hp=pt_hp;
	    s.mcmc_params=(GeneralizedHawkesProcessMCMCParams *)mcmc_params;
	    //resize the vector that will keep the polyagamma variables for each event
	    s.polyagammas.resize(mcmc_params->nof_sequences);
	    s.thinned_polyagammas.resize(mcmc_params->nof_sequences);
	    for(unsigned int l=0;l!=mcmc_params->nof_sequences;l++){
	    	s.polyagammas[l].resize(K);
	    	s.thinned_polyagammas[l].resize(K);
	    }

	    
		for(auto iter=psi.begin();iter!=psi.end();iter++){
			Kernel *psi_d=(*iter)->clone();
			s.psi.push_back(psi_d);
		}
	    
		for(auto iter=pt.begin();iter!=pt.end();iter++){
			LogisticKernel *pt_k=(*iter)->clone();
			pt_k->reset();
			pt_k->psi.clear();
			pt_k->d=K*psi.size();
			for(unsigned int k=0;k!=K;k++){
				for(long unsigned int d=0;d!=psi.size();d++){
					pt_k->psi.push_back(s.psi[d]);
				}
			}
			s.pt.push_back(pt_k);
		}
		
		for(long unsigned int k=0;k<pt_hp.size();k++){
			std::vector<SparseNormalGammaParameters *> pt_hp_k;
			for(long unsigned int k2=0;k2<pt_hp[k].size();k2++){
				SparseNormalGammaParameters *sng=new SparseNormalGammaParameters{*pt_hp[k][k2]};
				GeneralSigmoid * phi_tau=(GeneralSigmoid *) (sng->phi_tau);
				GeneralSigmoid * phi_mu=(GeneralSigmoid *) (sng->phi_mu);
				 sng->phi_tau=new GeneralSigmoid{phi_tau->x0, phi_tau->d0};
				sng->phi_mu=new GeneralSigmoid{phi_mu->x0, phi_mu->d0};
				pt_hp_k.push_back(sng);
			}
			s.pt_hp.push_back(pt_hp_k);
		}
	}
	
	//initialize the mcmc states for inference for hawkes processes, initialize the states with the event sequences
	HawkesProcess::mcmcInit(mcmc_params);

}

void GeneralizedHawkesProcess::mcmcSynchronize(unsigned int fit_nof_threads){
	
	for(unsigned int thread_id=0;thread_id<fit_nof_threads;thread_id++){
		pthread_mutex_t *m=(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(m,NULL);
		sample_thinned_events_mutex.push_back(m);
		
		pthread_mutex_t *m2=(pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
		pthread_mutex_init(m2,NULL);
		save_samples_mtx.push_back(m2);
		
	}
	nof_fit_threads_done=0;
	pthread_cond_init(&fit_con, NULL);
	pthread_mutex_init(&fit_mtx, NULL);
	
	HawkesProcess::mcmcSynchronize(fit_nof_threads);
}


void GeneralizedHawkesProcess::mcmcStartRun(unsigned int thread_id,unsigned int run_id, unsigned int iter_nof_threads){
	//initialize the mcmc for the full hawkes process
	HawkesProcess::mcmcStartRun(thread_id, run_id, iter_nof_threads);
	
	//initialize the precision of the inhibition weights given the initial state of the excitatory coefficients

	if(!pt_hp.empty()){//if the model is hierarchical
		mcmcStartSparseNormalGammas(thread_id, run_id, iter_nof_threads);
	}
	

	//initialize the history kernel functions
	for(auto iter=mcmc_state[run_id].psi.begin();iter!=mcmc_state[run_id].psi.end();iter++){
		pthread_mutex_lock(save_samples_mtx[thread_id]);
		(*iter)->generate();

		pthread_mutex_unlock(save_samples_mtx[thread_id]);
	}
	
	// distribute the initialization of the weights among the threads
	pthread_t mcmc_iter_threads[iter_nof_threads];
    unsigned int nof_types_thread[iter_nof_threads];

    
	unsigned int nof_types_thread_q=K/iter_nof_threads;//split across types
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

		InitThinKernelThreadParams* p;
	    p= new InitThinKernelThreadParams(thread_id, this, run_id, type_id_offset,nof_types_thread[child_thread_id]);
	    int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, GeneralizedHawkesProcess::mcmcStartRun_, (void *)p);
	    if (rc){
	         std::cerr << "Error:unable to create thread," << rc << std::endl;
	         exit(-1);
	    }
	    type_id_offset+=nof_types_thread[child_thread_id];
	}
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <iter_nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}
}

void * GeneralizedHawkesProcess::mcmcStartRun_(void *p){
	//unwrap the run parameters
	std::unique_ptr< InitThinKernelThreadParams > params( static_cast< InitThinKernelThreadParams * >( p ) );
	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int run_id=params->run_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	State & mcmc_state=ghp->mcmc_state[run_id];
	
	//generate the model thinning parameters from the priors
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){ 
		pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
		mcmc_state.pt[k]->generate();

		
		pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
	}

	mcmc_state.mcmc_iter=0;
	return 0;
}

void  GeneralizedHawkesProcess::mcmcStep(unsigned int thread_id, unsigned int run,unsigned int step, bool save , unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *), bool profiling){

	//fill in the observed sequence with sampled thinned events.
	for(unsigned int seq_id=0;seq_id!=HawkesProcess::mcmc_state[run].seq.size();seq_id++){
		mcmcStepSampleThinnedEvents(seq_id, run,step,thread_id,iter_nof_threads,profiling);
		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"RUN "<<run<<" MCMC STEP "<<step<<std::endl;
		debug_learning_file<<"nof thinned events "<<HawkesProcess::mcmc_state[run].seq[seq_id].thinned_full.size()<<std::endl;

		for(unsigned int l=0;l!=HawkesProcess::mcmc_state[run].seq.size();l++){
			for(auto e_iter=HawkesProcess::mcmc_state[run].seq[l].thinned_full.begin();e_iter!=HawkesProcess::mcmc_state[run].seq[l].thinned_full.end();e_iter++){
				debug_learning_file<<"type "<<e_iter->second->type<<" time "<<e_iter->second->time<<" prob "<<e_iter->second->p_observed<<std::endl;
			}
		}
		pthread_mutex_unlock(&debug_mtx);
		#endif
		//pthread_mutex_lock(&debug_mtx);
		//pthread_mutex_unlock(&debug_mtx);
	}
	//given the thinned events, the excitation and inhibition part of the process can be executed in parallel
	
	//update the mutual excitation part of the process (parents of observed events, coefficients of excitatory kernels but not the multiplicative constant in case of hierarchical model)

	pthread_t excitation_thread;
	SampleExcitationPartThreadParams* p2;
	p2= new SampleExcitationPartThreadParams(thread_id, this, run, step,save,iter_nof_threads,profiling);
	int rc2 = pthread_create(&excitation_thread, NULL, mcmcStepSampleExcitationPart_, (void *)p2);
	if (rc2){
			 std::cerr << "Error:unable to create thread," << rc2 << std::endl;
			 exit(-1);
	}
	//wait for the update of the excitation part to be done
	pthread_join (excitation_thread, NULL);
	
	
	//it samples the inhibition weights and the polyagammas needed for that
	pthread_t inhibition_thread;
	SampleInhibitionPartThreadParams* p;
	p= new SampleInhibitionPartThreadParams(thread_id, this, run, step,save,iter_nof_threads,profiling);
	int rc = pthread_create(&inhibition_thread, NULL, mcmcStepSampleInhibitionPart_, (void *)p);
	if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
	}
	
	//wait for the sampling of the excitation and inhibition part to finish

	pthread_join (inhibition_thread, NULL);
	//compute the total inference time so far
	if(profiling){
		profiling_mcmc_step[run][step]=profiling_hawkes_sampling[run][step];
		profiling_mcmc_step[run][step]+=profiling_logistic_sampling[run][step];
		profiling_mcmc_step[run][step]+=profiling_polyagammas_sampling[run][step];
		profiling_mcmc_step[run][step]+=profiling_thinned_events_sampling[run][step];
		profiling_mcmc_step[run][step]+=profiling_compute_history[run][step];
	}
	//flush the current mcmc state from the laten variables
	mcmcStepClear(run);
}


void *GeneralizedHawkesProcess::mcmcStepSampleInhibitionPart_(void *p){
	
	std::unique_ptr<SampleInhibitionPartThreadParams> params(static_cast< SampleInhibitionPartThreadParams * >(p));
	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;//calling object
	unsigned int run=params->run; //run of the mcmc 
	unsigned int step=params->step; //step in the run of the mcmc
	bool save=params->save; //whether the samples of the weights for the inhibition part will be saved
	unsigned int iter_nof_threads=params->iter_nof_threads; //nof threads for the current step/iteration
	bool profiling=params->profiling; //whether it will report inference time in a file

	//compute kernel history for observed and thinned event
	//first dimension refers to the number of event sequences used by the run
	ghp->mcmc_state[run].h=new double***[ghp->HawkesProcess::mcmc_state[run].seq.size()];
	ghp->mcmc_state[run].thinned_h=new double***[ghp->HawkesProcess::mcmc_state[run].seq.size()];
	//initialize timer for the profiling of this step
	boost::timer::cpu_timer infer_timer_1;
	for(unsigned int l=0;l!=ghp->HawkesProcess::mcmc_state[run].seq.size();l++){
		//compute kernel history of observed events
		ghp->computeKernelHistory(ghp->HawkesProcess::mcmc_state[run].seq[l].type, ghp->HawkesProcess::mcmc_state[run].seq[l].thinned_type, ghp->mcmc_state[run].thinned_h[l], iter_nof_threads, thread_id, run);
		ghp->computeKernelHistory(ghp->HawkesProcess::mcmc_state[run].seq[l].type, ghp->HawkesProcess::mcmc_state[run].seq[l].type, ghp->mcmc_state[run].h[l], iter_nof_threads, thread_id, run);
		//compute kernel history of thinned events
		

		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"compute kernel history of thinned events for run "<<run<<" at step "<<step<<std::endl;
		for(unsigned int k=0;k<ghp->K;k++){
			debug_learning_file<<"kernel history for thinned events of type "<<k<<std::endl;
			unsigned int event_id=0;
			for(auto e_iter=ghp->HawkesProcess::mcmc_state[run].seq[l].thinned_type[k].begin();e_iter!=ghp->HawkesProcess::mcmc_state[run].seq[l].thinned_type[k].end();e_iter++){
				debug_learning_file<<"history of event <"<<e_iter->second->time<<","<<e_iter->second->type<<">"<<ghp->mcmc_state[run].thinned_h[l][k][event_id++][0]<<std::endl;
			}
		}
		pthread_mutex_unlock(&debug_mtx);
		#endif
	}
	auto infer_timer_1_nanoseconds = boost::chrono::nanoseconds(infer_timer_1.elapsed().user + infer_timer_1.elapsed().system);
	if(profiling)
		ghp->profiling_compute_history[run][step]=(infer_timer_1_nanoseconds.count()*1e-9)+(step>0?ghp->profiling_compute_history[run][step-1]:0);


	//sample polyagamma variables for both the realized and thinned events
	boost::timer::cpu_timer infer_timer_2;

	for(unsigned int l=0;l!=ghp->HawkesProcess::mcmc_state[run].seq.size();l++)
		ghp->mcmcStepSamplePolyaGammas(thread_id, l, run, iter_nof_threads);
	

	auto infer_timer_2_nanoseconds = boost::chrono::nanoseconds(infer_timer_2.elapsed().user + infer_timer_2.elapsed().system);
	if(profiling)
		ghp->profiling_polyagammas_sampling[run][step]=(infer_timer_2_nanoseconds.count()*1e-9)+(step>0?ghp->profiling_polyagammas_sampling[run][step-1]:0);

	//update the mutual thinning part of the process
	boost::timer::cpu_timer infer_timer_3;

	ghp->mcmcUpdateThinKernelParams(thread_id, run, iter_nof_threads, step,save);
	ghp->mcmcHistoryKernelsUpdate(thread_id, run, iter_nof_threads, step, save);
	
	auto infer_timer_3_nanoseconds = boost::chrono::nanoseconds(infer_timer_3.elapsed().user + infer_timer_3.elapsed().system);
	if(profiling)
		ghp->profiling_logistic_sampling[run][step]=(infer_timer_3_nanoseconds.count()*1e-9)+(step>0?ghp->profiling_logistic_sampling[run][step-1]:0);
	
	return 0;

}

void *GeneralizedHawkesProcess::mcmcStepSampleExcitationPart_(void *p){
	
	std::unique_ptr<SampleExcitationPartThreadParams> params(static_cast< SampleExcitationPartThreadParams * >(p));
	GeneralizedHawkesProcess *ghp=params->ghp;//calling object
	unsigned int run=params->run; //run of the mcmc 
	unsigned int step=params->step; //step in the run of the mcmc
	bool save=params->save; //whether the samples of the weights for the inhibition part will be saved
	unsigned int thread_id=params->thread_id; //thread which executes the run
	unsigned int iter_nof_threads=params->iter_nof_threads; //nof threads for the current step/iteration
	bool profiling=params->profiling; //whether it will report inference time in a file
	
	
	boost::timer::cpu_timer infer_timer_4;
	ghp->HawkesProcess::mcmcStep(thread_id, run, step, save, iter_nof_threads, GeneralizedHawkesProcess:: mcmcUpdateExcitationKernelParams_, profiling);//it may save the sample this is why I need to put the run and step in the arguments
	auto infer_timer_4_nanoseconds = boost::chrono::nanoseconds(infer_timer_4.elapsed().user + infer_timer_4.elapsed().system);
	if(profiling){
		ghp->profiling_hawkes_sampling[run][step]=(infer_timer_4_nanoseconds.count()*1e-9)+(step>0?ghp->profiling_hawkes_sampling[run][step-1]:0);
	}
	
	return 0;
	
}


void * GeneralizedHawkesProcess::mcmcUpdateExcitationKernelParams_(void *p){
	std::unique_ptr< UpdateKernelThreadParams > params( static_cast< UpdateKernelThreadParams * >( p ) );
	GeneralizedHawkesProcess *ghp=dynamic_cast<GeneralizedHawkesProcess *>(params->hp);
	unsigned int thread_id=params->thread_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int run_id=params->run_id;
	unsigned int step_id=params->step_id;
	bool save=params->save;
	HawkesProcess::State & mcmc_hp_state=ghp->HawkesProcess::mcmc_state[run_id];
	//State & mcmc_ghp_state=ghp->mcmc_state[run_id];

	unsigned int K=ghp->K;

	//update the model parameters from the posteriors

	//TODO: divide the parameters of each type per thread
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){
		//update the background intensities
		pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
		mcmc_hp_state.mu[k]->mcmcExcitatoryUpdate(k,0,&mcmc_hp_state);
		pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
		if(save){
			unsigned int nofp=mcmc_hp_state.mu[k]->nofp;
			std::vector<double> mu_sample(nofp);
			mcmc_hp_state.mu[k]->getParams(mu_sample);
			for(unsigned int p=0;p<nofp;p++){
				pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
				ghp->mu_posterior_samples[k][p][run_id](step_id)=mu_sample[p];
				pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
			}
		}

		//TODO: cases for hierarchical vs flat point process
		//update the mutual excitation kernel parameters
		if(!mcmc_hp_state.phi.empty() && mcmc_hp_state.phi[k].size()==K){
			for(unsigned int k2=0;k2<K;k2++){
				pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
				if(!ghp->pt_hp.empty()){
					
					//the multiplicative coefficient will be jointly updated with the prior parameters of the logistic kernel
					switch(mcmc_hp_state.phi[k][k2]->type){
						case Kernel_Exponential:
							//std::cout<<"update decaying coefficient\n";
							((ExponentialKernel *)mcmc_hp_state.phi[k][k2])->mcmcExcitatoryExpUpdate(k,k2, &mcmc_hp_state, 0);
							break;
							
						default:
							break;
					}
					
					pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
					if(save){//todo: move it in the flat-model case???
						unsigned int nofp=mcmc_hp_state.phi[k][k2]->nofp;
						std::vector<double> phi_sample(nofp);
						mcmc_hp_state.phi[k][k2]->getParams(phi_sample);
						pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
						ghp->phi_posterior_samples[k][k2][1][run_id](step_id)=phi_sample[1];//save only the decaying coefficient
						pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
					}
			
				}
				else{//flat model
					mcmc_hp_state.phi[k][k2]->mcmcExcitatoryUpdate(k,k2, &mcmc_hp_state, 0);
					pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
					if(save){//todo: move it in the flat-model case???
						unsigned int nofp=mcmc_hp_state.phi[k][k2]->nofp;
						std::vector<double> phi_sample(nofp);
						mcmc_hp_state.phi[k][k2]->getParams(phi_sample);
						for(unsigned int p=0;p<nofp;p++){
							pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
							ghp->phi_posterior_samples[k][k2][p][run_id](step_id)=phi_sample[p];
							pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
						}
					}
				}
			}
		}
	}
	return 0;
}

void GeneralizedHawkesProcess::mcmcStepClear(unsigned int run_id){
	//delete thinned events of the step once you update the inhibition weights
	for(unsigned int l=0;l!=HawkesProcess::mcmc_state[run_id].seq.size();l++){

		//clear the history of the thinned events of the current mcmc iteration
		for(unsigned int k=0;k<K;k++){
			unsigned int Nt_k=HawkesProcess::mcmc_state[run_id].seq[l].thinned_type[k].size();//nof thinned events of type k
			for(unsigned int e=0;e!=Nt_k;e++){
				delete mcmc_state[run_id].thinned_h[l][k][e];
				 mcmc_state[run_id].thinned_h[l][k][e]=0;
			}
			delete mcmc_state[run_id].thinned_h[l][k];
			mcmc_state[run_id].thinned_h[l][k]=0;
			
			unsigned int N_k=HawkesProcess::mcmc_state[run_id].seq[l].type[k].size();//nof observed events of type k
			for(unsigned int e=0;e!=N_k;e++){
				delete mcmc_state[run_id].h[l][k][e];
				 mcmc_state[run_id].h[l][k][e]=0;
			}
			delete mcmc_state[run_id].h[l][k];
			mcmc_state[run_id].h[l][k]=0;
		}

		delete mcmc_state[run_id].thinned_h[l]; //clear history of thinned events
		mcmc_state[run_id].thinned_h[l]=0;
		
		delete mcmc_state[run_id].h[l]; //clear history of thinned events
		mcmc_state[run_id].h[l]=0;

		//clear polyagamma random variables of observed and thinned events
		if(mcmc_state[run_id].polyagammas.size()<l || mcmc_state[run_id].polyagammas[l].size()<K)
			std::cerr<<"wrong size of structure for polyagammas of observed events\n";

		if(mcmc_state[run_id].thinned_polyagammas.size()<l || mcmc_state[run_id].thinned_polyagammas[l].size()<K)
			std::cerr<<"wrong size of structure for polyagammas of thinned events\n";

		for(unsigned int k=0;k<K;k++){
			forced_clear(mcmc_state[run_id].polyagammas[l][k]);
			forced_clear(mcmc_state[run_id].thinned_polyagammas[l][k]);
		}

		forced_clear(mcmc_state[run_id].polyagammas[l]);
		forced_clear(mcmc_state[run_id].thinned_polyagammas[l]);

		HawkesProcess::mcmc_state[run_id].seq[l].flushThinnedEvents();//flush the thinned events from the event sequence

		//flush thinned events from the vectorized structure. the first entries refer to the realized events that should be preserved
		HawkesProcess::mcmc_state[run_id].seq_v[l].erase(HawkesProcess::mcmc_state[run_id].seq_v[l].begin()+HawkesProcess::mcmc_state[run_id].seq[l].N,HawkesProcess::mcmc_state[run_id].seq_v[l].end());
	}
}
/////////////////////////////////// sample history kernel params ///////////////////////////////////

void GeneralizedHawkesProcess::mcmcHistoryKernelsUpdate(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads,  unsigned int step_id, bool save){
	//update the history kernel functions
	for(unsigned d=0;d!=mcmc_state[run_id].psi.size();d++){
		mcmc_state[run_id].psi[d]->mcmcInhibitoryUpdate(d,-1,&HawkesProcess::mcmc_state[run_id], &mcmc_state[run_id]);
		//todo: gather the psi samples here
	}
	
	//save the posterior samples for the params of the history kernel functions, if necessary
	pthread_mutex_unlock(save_samples_mtx[thread_id]);
	if(save){
		for(long unsigned int d=0;d<mcmc_state[run_id].psi.size();d++){
			unsigned int nofp=mcmc_state[run_id].psi[d]->nofp;
			std::vector<double> psi_sample(nofp);
			mcmc_state[run_id].psi[d]->getParams(psi_sample);
			for(unsigned int p=0;p<nofp;p++){
				pthread_mutex_lock(save_samples_mtx[thread_id]);
				psi_posterior_samples[d][p][run_id](step_id)=psi_sample[p];
				pthread_mutex_unlock(save_samples_mtx[thread_id]);
			}
		}
	}
}


/////////////////////////////////// sample thinned events methods ///////////////////////////////////
void GeneralizedHawkesProcess::mcmcStepSampleThinnedEvents(unsigned int seq_id, unsigned int run,unsigned int step, unsigned int thread_id, unsigned int iter_nof_threads, bool profiling){
	
	//sample thinned events
	boost::timer::cpu_timer infer_timer_1;
	
	sampleThinnedEvents(thread_id, seq_id, run, iter_nof_threads);
	
	auto infer_timer_1_nanoseconds = boost::chrono::nanoseconds(infer_timer_1.elapsed().user + infer_timer_1.elapsed().system);
	
	//report inference time if needed
	if(profiling)
		profiling_thinned_events_sampling[run][step]=(infer_timer_1_nanoseconds.count()*1e-9)+(step>0?profiling_thinned_events_sampling[run][step-1]:0);
	
	
	//vectorize thinned events
	pthread_mutex_lock(save_samples_mtx[thread_id]);
	map_to_vector(HawkesProcess::mcmc_state[run].seq[seq_id].thinned_full, HawkesProcess::mcmc_state[run].seq_v[seq_id]);
	pthread_mutex_unlock(save_samples_mtx[thread_id]);
	
}


void GeneralizedHawkesProcess::sampleThinnedEvents(unsigned int thread_id, unsigned int seq_id, unsigned int run, unsigned int iter_nof_threads){
	
	//distribute types of thinned events to be sampled across threads
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
		if(!nof_types_thread[child_thread_id])
			break;
	
		SampleThinnedEventsParams* p;
		p= new SampleThinnedEventsParams(thread_id, this, seq_id, run, type_id_offset,nof_types_thread[child_thread_id], iter_nof_threads, sample_thinned_events_mutex[thread_id]);
		int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, sampleThinnedEvents_, (void *)p);
		if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
		type_id_offset+=nof_types_thread[child_thread_id];
	
	}
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <iter_nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}
}

void *GeneralizedHawkesProcess::sampleThinnedEvents_(void *p){
	
	std::unique_ptr<SampleThinnedEventsParams > params(static_cast< SampleThinnedEventsParams * >(p));
	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int seq_id=params->seq_id;
	unsigned int run_id=params->run_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int K=ghp->K;
	unsigned int start_t=ghp->start_t;
	unsigned int end_t=ghp->end_t;
	unsigned int nof_threads=params->nof_threads;
	State & mcmc_state=ghp->mcmc_state[run_id];
	HawkesProcess::State & hp_mcmc_state=ghp->HawkesProcess::mcmc_state[run_id];

	Event *ve=hp_mcmc_state.seq[seq_id].full.begin()->second;//virtual event of the process
	//sample thinned events for the types that correspond to the thread
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){	
	
		//sample the background process
		std::vector<double> arrival_times=simulateHPoisson(hp_mcmc_state.mu[k]->p[0],start_t,end_t);
		unsigned int Nt=arrival_times.size();//candidate thinned events of type k which belong to the background process
		//thin the background process 
		for(unsigned int i=0;i<Nt;i++){
					
			//decide whether to thin or not the event
			double p_t=computeRealizationProbability(hp_mcmc_state.seq[seq_id],mcmc_state.pt[k],arrival_times[i]);
				
			std::random_device rd;
			std::mt19937 gen(rd());
			std::bernoulli_distribution bern_distribution(1-p_t);
			bool b_t=bern_distribution(gen);
			if(b_t){//the event is a thinned event, otherwise it is discarded
						
				pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
				pthread_mutex_lock(params->sample_thinned_events_mutex);
				Event * e=new Event{hp_mcmc_state.seq[seq_id].type[k].size(),(int)k,arrival_times[i],K,false, ve};
				hp_mcmc_state.seq[seq_id].addEvent(e,ve);
				pthread_mutex_unlock(params->sample_thinned_events_mutex);
				pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
				#if DEBUG_LEARNING
				pthread_mutex_lock(&debug_mtx);
				e->p_observed=p_t;
				pthread_mutex_unlock(&debug_mtx);
				#endif
			}
			
		}

		if(!ghp->phi.empty()){//if there is mutual excitation part in the model
			
			unsigned int N=hp_mcmc_state.seq[seq_id].full.size();//number of observed events
			
			//distribute the observed events across the threads
			pthread_t mcmc_threads[nof_threads];
			unsigned int nof_events_thread[nof_threads];
			
			unsigned int nof_events_thread_q=(N-1)/nof_threads;	//skip the virtual event which generates the homogeneous poisson
			unsigned int nof_events_thread_m=(N-1)%nof_threads; //skip the virtual event which generates the homogeneous poisson
			for(unsigned int t=0;t<nof_threads;t++){
				nof_events_thread[t]=nof_events_thread_q;
				if(t<nof_events_thread_m)
					nof_events_thread[t]++;
			}
	
			unsigned int event_id_offset=1;	//skip the virtual event which generates the homogeneous poisson
			for(unsigned int child_thread_id=0;child_thread_id<nof_threads;child_thread_id++){
				if(!nof_events_thread[child_thread_id])
					break;
								
				SampleThinnedEventsParams_* p;
				//use the history of realized and thinned events which correspond to type k2
				p= new SampleThinnedEventsParams_(thread_id, ghp, seq_id, run_id, k,nof_events_thread[child_thread_id], event_id_offset,params->sample_thinned_events_mutex);
				int rc = pthread_create(&mcmc_threads[child_thread_id], NULL, sampleThinnedEvents__, (void *)p);
				if (rc){
					 exit(-1);
				}
				event_id_offset+=nof_events_thread[child_thread_id];
			
			}
			
			//wait for all the nonhomogeneous poisson processes to finish
			for (unsigned int t = 0; t <nof_threads; t++){
				if(nof_events_thread[t])
					pthread_join (mcmc_threads [t], NULL);
			}
		}		
	}
	return 0;
}

void *GeneralizedHawkesProcess::sampleThinnedEvents__(void *p){//parallelism across events of a specific type which generate thinned events
	
	std::unique_ptr<SampleThinnedEventsParams_ > params(static_cast< SampleThinnedEventsParams_* >(p));
	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int seq_id=params->seq_id;
	unsigned int run_id=params->run_id;
	int k=params->k;//type of the thinned events that will be generated
	unsigned int nof_events=params->nof_events;
	unsigned int event_id_offset=params->event_id_offset;
	unsigned int K=ghp->K;
	unsigned int end_t=ghp->end_t;
	State & mcmc_state=ghp->mcmc_state[run_id];//mcmc state of the generalized hawkes process (which holds the inhibition weights needed for the computation of thinning probability)
	HawkesProcess::State & hp_mcmc_state=ghp->HawkesProcess::mcmc_state[run_id];//mcmc state of the hawkes process (where the thinned events will be stored)
	
	
	for(auto event_id=event_id_offset;event_id<event_id_offset+nof_events;event_id++){//TODO: parallelize this step
		
		Event * e=hp_mcmc_state.observed_seq_v[seq_id][event_id];
		if(!e)
			std::cerr<<"invalid event!\n";
		
		if(hp_mcmc_state.phi.size()!=K){
			std::cerr<<"invalid trigerring kernel1\n";	
		}
		if(hp_mcmc_state.phi[e->type].size()!=K){
			std::cerr<<"invalid trigerring kernel2\n";
		}
				
		if(!hp_mcmc_state.phi[e->type][k])
			std::cerr<<"invalid trigerring kernel\n";
				
		std::vector<double> arrival_times=simulateNHPoisson(*hp_mcmc_state.phi[e->type][k],e->time, e->time, end_t);
		
	
		unsigned int Nt=arrival_times.size();//candidate thinned events of type k generated by event of iter
				
		//thin offsprings of event e for the mutually exciting processes
		for(unsigned i=0;i<Nt;i++){
		    //decide whether to thin or not the event
			double p_t=computeRealizationProbability(hp_mcmc_state.seq[seq_id],mcmc_state.pt[k],arrival_times[i]);//todo: static method for using a logistic kernel passed as argument!!!
					
			std::random_device rd;
			std::mt19937 gen(rd());
			std::bernoulli_distribution bern_distribution(1-p_t);
			bool b_t=bern_distribution(gen);
			if(b_t){//the event is a thinned event, otherwise it is discarded
				pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
				pthread_mutex_lock(params->sample_thinned_events_mutex);
				Event *e2=new Event{hp_mcmc_state.seq[seq_id].type[k].size(),k,arrival_times[i],K,false, e};
				hp_mcmc_state.seq[seq_id].addEvent(e2,e);
				pthread_mutex_unlock(params->sample_thinned_events_mutex);
				pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
				
				#if DEBUG_LEARNING
				pthread_mutex_lock(&debug_mtx);
				e2->p_observed=p_t;
				pthread_mutex_unlock(&debug_mtx);
				#endif
			}	
		}			
	}
	return 0;
}
/////////////////////////////////// sample polyagammas methods ///////////////////////////////////
void GeneralizedHawkesProcess::mcmcStepSamplePolyaGammas(unsigned int thread_id, unsigned int l, unsigned int run_id, unsigned int iter_nof_threads){


	if(mcmc_state[run_id].polyagammas[l].size()!=K)
		mcmc_state[run_id].polyagammas[l].resize(K);
		
	if(mcmc_state[run_id].thinned_polyagammas[l].size()!=K)
		mcmc_state[run_id].thinned_polyagammas[l].resize(K); 
		
		//reallocate the vectors for the polyagammas. it is needed both for efficiency and for mutlithreaded programming 
	for(unsigned int k=0;k<K;k++){
		mcmc_state[run_id].polyagammas[l][k].resize(HawkesProcess::mcmc_state[run_id].seq[l].type[k].size());
		mcmc_state[run_id].thinned_polyagammas[l][k].resize(HawkesProcess::mcmc_state[run_id].seq[l].thinned_type[k].size());
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
		if(!nof_types_thread[child_thread_id])
			break;
	
		SamplePolyagammasParams* p;
		p= new SamplePolyagammasParams(thread_id, this, l, run_id, type_id_offset,nof_types_thread[child_thread_id], iter_nof_threads);//BUG: this runs if iter_nof_threads>=fit_threads
		int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, mcmcSamplePolyaGammas_, (void *)p);
		if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
		type_id_offset+=nof_types_thread[child_thread_id];
	
	}
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <iter_nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}
}

void* GeneralizedHawkesProcess::mcmcSamplePolyaGammas_( void *p){
	std::unique_ptr<SamplePolyagammasParams > params(static_cast< SamplePolyagammasParams * >(p));

	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int seq_id=params->seq_id;
	unsigned int run_id=params->run_id;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int nof_threads=params->nof_threads;
	//State & mcmc_state=ghp->mcmc_state[run_id];
	
	if(!ghp)
		std::cerr<<"invalid generalzied hawkes process object\n";
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){

		unsigned int N=ghp->HawkesProcess::mcmc_state[run_id].seq[seq_id].type[k].size();
		unsigned int Nt=ghp->HawkesProcess::mcmc_state[run_id].seq[seq_id].thinned_type[k].size();

		pthread_t mcmc_threads[nof_threads];
		unsigned int nof_events_thread[nof_threads];
		
		//distribute the events of type k across the threads
		unsigned int nof_events_thread_q=(N+Nt)/nof_threads;
		unsigned int nof_events_thread_m=(N+Nt)%nof_threads;
		
		for(unsigned int t=0;t<nof_threads;t++){
			nof_events_thread[t]=nof_events_thread_q;
			if(t<nof_events_thread_m)
				nof_events_thread[t]++;
		}
		
		//create threads
		unsigned int event_id_offset=0;
		for(unsigned int child_thread_id=0;child_thread_id<nof_threads;child_thread_id++){
			if(!nof_events_thread[child_thread_id])
				break;
								
			SamplePolyagammasParams_* p;
			//use the history of realized and thinned events which correspond to type k2
			p= new SamplePolyagammasParams_(thread_id, ghp, seq_id, run_id, k, event_id_offset,nof_events_thread[child_thread_id], N, Nt);
			
			int rc = pthread_create(&mcmc_threads[child_thread_id], NULL, mcmcSamplePolyaGammas__, (void *)p);
			if (rc){
				 std::cerr << "Error:unable to create thread," << rc << std::endl;
				 exit(-1);
			}
			event_id_offset+=nof_events_thread[child_thread_id];
		
		}
		//wait for all the partial sums to be computed
		for (unsigned int t = 0; t <nof_threads; t++){
			if(nof_events_thread[t])
				pthread_join (mcmc_threads [t], NULL);
		}
	}
	return 0;
}

void* GeneralizedHawkesProcess::mcmcSamplePolyaGammas__( void *p){

	std::unique_ptr<SamplePolyagammasParams_> params(static_cast< SamplePolyagammasParams_ * >(p));
	unsigned int thread_id=params->thread_id;

	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int run_id=params->run_id;
	unsigned int seq_id=params->seq_id;
	unsigned int k=params->k;
	unsigned int event_id_offset=params->event_id_offset;
	unsigned int nof_events=params->nof_events;
	State & mcmc_state=ghp->mcmc_state[run_id];
	unsigned int N=ghp->HawkesProcess::mcmc_state[run_id].seq[seq_id].type[k].size(); //nof of observed events of type k
	unsigned int Nt=ghp->HawkesProcess::mcmc_state[run_id].seq[seq_id].thinned_type[k].size(); //nof of thinned events of type k

	for(unsigned int i=event_id_offset;i<event_id_offset+nof_events;i++){
		if(i>=Nt+N){
			std::cerr<<"invalid event id!\n";
		}
		
		PolyaGamma pg;
		RNG r(time(NULL));
		
		if(i>=N){//the event is a thinned event
			unsigned int thinned_i=i-N;//shift the index to refer to the index of the event in the structures for the thinned events
			double psi_n=mcmc_state.pt[k]->computeArg(mcmc_state.thinned_h[seq_id][k][thinned_i]);
			double s=pg.draw(1,psi_n,r);
			pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
			mcmc_state.thinned_polyagammas[seq_id][k][thinned_i]=s;
			pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
		}
		else{//the event is an observed event
			//compute the second parameter for the polyagamma--multiply the history by the inhibition weights
			double psi_n=mcmc_state.pt[k]->computeArg(mcmc_state.h[seq_id][k][i]);
			//draw polyagamma for the event
			double s=pg.draw(1,psi_n,r);
			pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
			mcmc_state.polyagammas[seq_id][k][i]=s;
			pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
		}
	}
	return 0;
}


/////////////////////////////////// sample logkernel params methods ///////////////////////////////////
void GeneralizedHawkesProcess::mcmcUpdateThinKernelParams(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads,unsigned int step_id, bool save){
	//the first event that occurs by default belongs to the background process
	#if DEBUG_LEARNING
	pthread_mutex_lock(&debug_mtx);
	debug_learning_file<<"Thinning Kernel Parameters run id "<<run_id<<" step id "<<step_id<<std::endl;
	for(auto iter=mcmc_state[run_id].pt.begin();iter!=mcmc_state[run_id].pt.end();iter++){
		(*iter)->print(debug_learning_file);
	}
	pthread_mutex_unlock(&debug_mtx);
	#endif

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
		if(!nof_types_thread[child_thread_id])
			break;
	
		UpdateThinKernelThreadParams* p;
		p= new UpdateThinKernelThreadParams(thread_id, this, type_id_offset,nof_types_thread[child_thread_id], run_id, step_id, iter_nof_threads, save);
		int rc = pthread_create(&mcmc_iter_threads[child_thread_id], NULL, mcmcUpdateThinKernelParams_, (void *)p);
		if (rc){
			 std::cerr << "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
		type_id_offset+=nof_types_thread[child_thread_id];
	
	}
	//wait for all the kernel parameters to be updated
	for (unsigned int t = 0; t <iter_nof_threads; t++){
		if(nof_types_thread[t])
			pthread_join (mcmc_iter_threads [t], NULL);
	}
}


void* GeneralizedHawkesProcess::mcmcUpdateThinKernelParams_(void *p){

	std::unique_ptr<UpdateThinKernelThreadParams > params(static_cast< UpdateThinKernelThreadParams * >(p));
	unsigned int thread_id=params->thread_id;
	GeneralizedHawkesProcess *ghp=params->ghp;
	unsigned int type_id_offset=params->type_id_offset;
	unsigned int nof_types=params->nof_types;
	unsigned int run_id=params->run_id;
	unsigned int step_id=params->step_id;
	unsigned int nof_threads=params->nof_threads;
	bool save=params->save;
	State & mcmc_state=ghp->mcmc_state[run_id];
	HawkesProcess::State & mcmc_hp_state=ghp->HawkesProcess::mcmc_state[run_id];
	
	//update the model parameters from the posteriors
	#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"update thin kernel param method: run "<<run_id<<" step: "<<step_id<<std::endl;
		pthread_mutex_unlock(&debug_mtx);

	#endif
	//TODO: divide the parameters of each type per thread
	for(unsigned int k=type_id_offset;k<type_id_offset+nof_types;k++){
		//update the thinning kernel parameters
		if(!mcmc_state.pt[k])
			std::cerr<<"invalid inhibition logistic kernel\n";

		pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
		
		mcmc_state.pt[k]->mcmcInhibitoryUpdate(-1,k,&ghp->HawkesProcess::mcmc_state[run_id], &mcmc_state, nof_threads);
		#if DEBUG_LEARNING
		pthread_mutex_lock(&debug_mtx);
		debug_learning_file<<"update thin kernel param method: run "<<run_id<<" step: "<<step_id<<" type "<<k<<std::endl;
		mcmc_state.pt[k]->print(debug_learning_file);
		pthread_mutex_unlock(&debug_mtx);

		#endif
		pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
		if(save){
			unsigned int nofp=mcmc_state.pt[k]->nofp;
			std::vector<double> pt_sample;
			std::vector<std::vector<double>> hp_samples;
			mcmc_state.pt[k]->getParams(pt_sample);
			if(!ghp->pt_hp.empty()){
				mcmc_state.pt[k]->getHyperParams(hp_samples);
			}
						
				for(unsigned int p=0;p<nofp;p++){
					pthread_mutex_lock(ghp->save_samples_mtx[thread_id]);
					ghp->pt_posterior_samples[k][p][run_id](step_id)=pt_sample[p];
					if(!ghp->pt_hp.empty()){
						ghp->tau_posterior_samples[k][p][run_id](step_id)=1/(hp_samples[p][1]*hp_samples[p][1]);
					}

					pthread_mutex_unlock(ghp->save_samples_mtx[thread_id]);
				}
				for(unsigned int k2=0;k2!=ghp->K;k2++){
					
					std::vector<double> phi_sample(nofp);
					mcmc_hp_state.phi[k2][k]->getParams(phi_sample);
					pthread_mutex_lock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
		
					//ghp->phi_posterior_samples[k][k2][0][run_id](step_id)=phi_sample[0];
					ghp->phi_posterior_samples[k2][k][0][run_id](step_id)=phi_sample[0];//save the excitation coefficient
					pthread_mutex_unlock((ghp->HawkesProcess::save_samples_mtx[thread_id]));
				}
				
				
		}	
	}
	return 0;
}

/////////////////////////////////// sample precision of logkernel weights ///////////////////////////////////
//todo: remove this, excitatory kernels and normal gamma paramters are updated jointly
//TODO: generalize for more history dimensions in which case k*D sparse normal gammas should be set 
void GeneralizedHawkesProcess::mcmcStartSparseNormalGammas(unsigned int thread_id, unsigned int run, unsigned int iter_nof_threads){

	for(unsigned int k=0;k!=K;k++){//update the normal-gamma prior for each type of the process
		std::vector<Kernel *>phi_k;//excitation effects on type k
		std::vector<SparseNormalGammaParameters *> pt_hp_k; //hyperprior for thinning effects on type k
		for(unsigned int k2=0;k2<K;k2++){
			phi_k.push_back(HawkesProcess::mcmc_state[run].phi[k2][k]);
			SparseNormalGammaParameters *proposed_sng=new SparseNormalGammaParameters{*pt_hp[k2][k]};
			proposed_sng->phi_tau=new GeneralSigmoid{((GeneralSigmoid *)(proposed_sng->phi_tau))->x0, (((GeneralSigmoid *)(proposed_sng->phi_tau))->d0)};// smooth thresholding, for the normal-gamma proposal
			proposed_sng->phi_mu=new GeneralSigmoid{((GeneralSigmoid *)(proposed_sng->phi_mu))->x0, (((GeneralSigmoid *)(proposed_sng->phi_tau))->d0)};// smooth thresholding
			pt_hp_k.push_back(proposed_sng);
		}
		
		//update normal gamma for type k
		setSparseNormalGamma(phi_k,pt_hp_k,mcmc_state[run].pt[k]);
	}

}





/****************************************************************************************************************************************************************************
 * Processing MCMC samples Methods
******************************************************************************************************************************************************************************/

void GeneralizedHawkesProcess::computePosteriorParams(std::vector<std::vector<double >> & mean_pt_param, 
										   std::vector<std::vector<double >> & mode_pt_param, 
										   std::vector<std::vector<double>> & mean_psi_param,
										   std::vector<std::vector<double>> & mode_psi_param,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_posterior_samples,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_posterior_samples
										   ) {
	//compute posteior mode and mean for the interaction weights
	long unsigned int K=pt_posterior_samples.size();
	for(long unsigned int k=0;k<K;k++){
		unsigned int nofp=pt_posterior_samples[k].size();
		for(unsigned int p=0;p<nofp;p++){
			boost::numeric::ublas::vector<double> pt_posterior_samples_all=merge_ublasv(pt_posterior_samples[k][p]);
			double mean_pt_k_p_param=sample_mean(pt_posterior_samples_all);
			mean_pt_param[k][p]=(mean_pt_k_p_param);//TODO: check for right dimension
			double mode_pt_k_p_param=sample_mode(pt_posterior_samples_all);
			mode_pt_param[k][p]=(mode_pt_k_p_param);
		}
	}
	//compute posteior mode and mean for the history kernel functions

	for(long unsigned int d=0;d<psi_posterior_samples.size();d++){
		unsigned int nofp=psi_posterior_samples[d].size();
		for(unsigned int p=0;p<nofp;p++){
				boost::numeric::ublas::vector<double> psi_posterior_samples_all=merge_ublasv(psi_posterior_samples[d][p]);
				mean_psi_param[d][p]=sample_mean(psi_posterior_samples_all);
				double mode_psi_d_p_param=sample_mode(psi_posterior_samples_all);
				mode_psi_param[d][p]=(mode_psi_d_p_param<=0?mean_psi_param[d][p]:mode_psi_d_p_param);//this is due to insufficient precision (nof plot points) in the gaussian kernel regression
		}
	}
}


void GeneralizedHawkesProcess::computePosteriorParams(std::vector<std::vector<double>> & mean_pt_param, std::vector<std::vector<double>> & mode_pt_param, std::vector<std::vector<double>> & mean_psi_param, std::vector<std::vector<double>> & mode_psi_param){
	GeneralizedHawkesProcess::computePosteriorParams(mean_pt_param, mode_pt_param, mean_psi_param, mode_psi_param, pt_posterior_samples, psi_posterior_samples);
}

//for setting both mutual excitation and inhibition
void GeneralizedHawkesProcess::setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
		                       std::vector<std::vector<Kernel *>> & post_mean_phi, //todo change it to std::vector<kernel *> *
							   std::vector<LogisticKernel *> & post_mean_pt,
							   std::vector<Kernel *> & post_mean_psi,
							   std::vector<ConstantKernel*> & post_mode_mu, 
							   std::vector<std::vector<Kernel *>> & post_mode_phi,
							   std::vector<LogisticKernel *> & post_mode_pt,
							   std::vector<Kernel *> & post_mode_psi,
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples,  
							   const std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> & phi_posterior_samples,      
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_samples,
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_samples
							   ){
	setPosteriorParams(post_mean_pt, post_mean_psi, post_mode_pt, post_mode_psi, pt_samples,psi_samples);
	HawkesProcess::setPosteriorParams(post_mean_mu, post_mean_phi, post_mode_mu, post_mode_phi, mu_samples, phi_posterior_samples);
}


//for setting only inhibition and constant excitation
void GeneralizedHawkesProcess::setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
							   std::vector<LogisticKernel *> & post_mean_pt,
							   std::vector<Kernel *> & post_mean_psi,
							   std::vector<ConstantKernel*> & post_mode_mu,
							   std::vector<LogisticKernel *> & post_mode_pt,
							   std::vector<Kernel *> & post_mode_psi,
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples,       
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_samples,
							   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_samples
							   ){
	setPosteriorParams(post_mean_pt, post_mean_psi, post_mode_pt,  post_mode_psi, pt_samples, psi_samples);
	HawkesProcess::setPosteriorParams(post_mean_mu, post_mode_mu,mu_samples );
}


void GeneralizedHawkesProcess::setPosteriorParams(
        					std::vector<LogisticKernel *> & post_mean_pt, 
							std::vector<Kernel *> & post_mean_psi,
							std::vector<LogisticKernel *> & post_mode_pt, 
							std::vector<Kernel *> & post_mode_psi,
							const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_posterior_samples,
							const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_posterior_samples
 ){
	
	//compute the posterior point estimations from the samples for the logistic kernels
	long unsigned int K=pt_posterior_samples.size();
	long unsigned int nofp=pt_posterior_samples[0].size();
	std::vector<std::vector<double>> mean_pt_param(K,std::vector<double>(nofp));
	std::vector<std::vector<double>> mode_pt_param(K,std::vector<double>(nofp));

	long unsigned int nofp_psi=psi_posterior_samples[0].size();
	long unsigned int d=psi_posterior_samples.size();

	
	std::vector<std::vector<double>> mean_psi_param(d,std::vector<double>(nofp_psi));
	std::vector<std::vector<double>> mode_psi_param(d,std::vector<double>(nofp_psi));

	computePosteriorParams(mean_pt_param, mode_pt_param, mean_psi_param, mode_psi_param, pt_posterior_samples, psi_posterior_samples);//todo change it so that mean_pt_param, mean_pt_param are initialized before the call
	
	//set history kernels to the point estimations
	for(long unsigned int i=0;i!=d;i++){
		if(!post_mean_psi[i]) return;
		post_mean_psi[i]->setParams(mean_psi_param[i]);
		
		if(!post_mode_psi[i]) return;
		post_mode_psi[i]->setParams(mode_psi_param[i]);
		
		
	}
	
	//set the sigmoid kernels to the point estimations
	for(long unsigned int k=0;k<K;k++){
		if(!post_mean_pt[k])
			return;//kernel not properly allocated
		post_mean_pt[k]->setParams(mean_pt_param[k]);
		post_mean_pt[k]->d=K*d;
		post_mean_pt[k]->psi.clear();
		for(unsigned int k2=0;k2!=K;k2++){
			for(unsigned int i=0;i!=d;i++){
			
				post_mean_pt[k]->psi.push_back(post_mean_psi[i]);
			}
		}
		//post_mode_pt[k]->setParams(mean_pt_param[k]);
		if(!post_mode_pt[k])
			return;//kernel not properly allocated
		post_mode_pt[k]->setParams(mode_pt_param[k]);
		post_mode_pt[k]->d=K*d;
		post_mode_pt[k]->psi.clear();
		for(unsigned int k2=0;k2!=K;k2++){
			for(unsigned int i=0;i!=d;i++){
				post_mode_pt[k]->psi.push_back(post_mode_psi[i]);
			}
		}
	}
}
	

//TODO: set the hyperparameters from here!
void GeneralizedHawkesProcess::setPosteriorParams(){
	forced_clear(post_mean_pt);
	forced_clear(post_mode_pt);
	//initialize the logistic kernels of the inhibition part with the hyperparameters
	for(auto iter=pt.begin();iter!=pt.end();iter++){
		post_mean_pt.push_back((*iter)->clone());
		post_mode_pt.push_back((*iter)->clone());
	}
	forced_clear(post_mean_psi);
	forced_clear(post_mode_psi);
	//initialize the logistic kernels of the inhibition part with the hyperparameters
	for(auto iter=psi.begin();iter!=psi.end();iter++){
		post_mean_psi.push_back((*iter)->clone());
		post_mode_psi.push_back((*iter)->clone());
	}
	
	//set the posterior params for the inhibition part of the process
	setPosteriorParams(post_mean_pt,post_mean_psi, post_mode_pt,post_mode_psi, pt_posterior_samples,psi_posterior_samples);
	//set posterior parameters for the excitation part
	HawkesProcess::setPosteriorParams();

}

void GeneralizedHawkesProcess::flushBurninSamples(unsigned int nof_burnin){

	//flush the samples for the interaction weights
	for(unsigned int k=0;k<pt.size();k++){
		if(pt_posterior_samples.size()<k)
			std::cerr<<"wrong dimension for posterior samples\n";
		
		for(unsigned int p=0;p<pt[k]->nofp;p++){
			if(pt_posterior_samples.size()<k)
				std::cerr<<"wrong dimension for posterior samples\n";
			if(pt_posterior_samples[k].size()<p)
				std::cerr<<"wrong dimension for posterior samples\n";

			for(unsigned int r=0;r<pt_posterior_samples[k][p].size();r++){//this dimension refers to the mcmc runs
					boost::numeric::ublas::vector<double> u=boost::numeric::ublas::subrange(pt_posterior_samples[k][p][r], nof_burnin,pt_posterior_samples[k][p][r].size());
					pt_posterior_samples[k][p][r].clear();
					pt_posterior_samples[k][p][r]=u;
			}
		}
	}
	
	//flush the samples for the history kernel functions
	for(long unsigned int d=0;d<psi.size();d++){
		for(unsigned int p=0;p<psi[d]->nofp;p++){
				for(long unsigned int r=0;r<psi_posterior_samples[d][p].size();r++){//this dimension refers to the mcmc runs
					boost::numeric::ublas::vector<double> u=boost::numeric::ublas::subrange(psi_posterior_samples[d][p][r], nof_burnin,psi_posterior_samples[d][p][r].size());
					psi_posterior_samples[d][p][r].clear();
					psi_posterior_samples[d][p][r]=u;
				}
		}
	}

	
	
	//flush the burnin samples for the parameters in the excitation part
	HawkesProcess::flushBurninSamples(nof_burnin);
}

/****************************************************************************************************************************************************************************
 * Model Learning Plot Methods
******************************************************************************************************************************************************************************/

void GeneralizedHawkesProcess::generateMCMCplots(unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool  write_png_file, bool write_csv_file){
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
	generateMCMCplots_(dir_path_str,samples_step,true_values, burnin_num_iter, write_png_file, write_csv_file);
}


void GeneralizedHawkesProcess::generateMCMCplots(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool  write_png_file, bool write_csv_file){
	if(!write_png_file && !write_csv_file)
		return;
	
	generateMCMCplots_(dir_path_str,samples_step,true_values, burnin_num_iter, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCplots_(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool  write_png_file, bool write_csv_file) {
	if(!write_png_file && !write_csv_file)
		return;
	generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
	generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);
	generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);
	generateMCMCTrainLikelihoodPlots(samples_step, true_values, write_png_file, write_csv_file, true,dir_path_str);
	flushBurninSamples(burnin_num_iter);
	generateMCMCPosteriorPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCTracePlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCTracePlots(bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCTracePlots_(const std::string & dir_path_str, bool write_png_file , bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
	//plot the trace of the parameters in the inhibition part


	//plot the trace of the interaction weights
	for(unsigned int k=0;k<pt.size();k++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[k]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				//std::string filename=dir_path_str+"trace_pt_params_type"+std::to_string(k)+"_param_"+std::to_string(p)+".png";
				std::string filename=dir_path_str+"trace_pt_params_type"+std::to_string(k)+"_param_"+std::to_string(p);
				plotTraces(filename,pt_posterior_samples[k][p], write_png_file, write_csv_file);
			}
	}

	//plot the trace of the history kernel functions

		//plot the trace of the kernel parameters
	for(unsigned int d=0;d<psi.size();d++){
			//plot the trace of the kernel parameters
		unsigned int nofp=psi[d]->nofp;
		for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
			std::string filename=dir_path_str+"trace_psi_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
			plotTraces(filename,psi_posterior_samples[d][p], write_png_file, write_csv_file);
		}
	}
	
	//plot the trace of the precision
	if(!pt_hp.empty()){
		for(unsigned int d=0;d<pt.size();d++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[d]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"trace_tau_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
				plotTraces(filename,tau_posterior_samples[d][p], write_png_file, write_csv_file);
			}
		}
	}
	

	//plot the trace of the paramters in the excitation part
	HawkesProcess::generateMCMCTracePlots_(dir_path_str, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCMeanPlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);
}


void GeneralizedHawkesProcess::generateMCMCMeanPlots(bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);

}

void GeneralizedHawkesProcess::generateMCMCMeanPlots_(const std::string & dir_path_str,bool write_png_file, bool write_csv_file) const {
	//meanplot of the parameters in the inhibition part
	if(!write_png_file && !write_csv_file)
		return;

	//meanplot of the interaction weights
	for(unsigned int k=0;k<pt.size();k++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[k]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"meanplot_pt_params_type"+std::to_string(k)+"_param_"+std::to_string(p);
				plotMeans(filename,pt_posterior_samples[k][p], write_png_file, write_csv_file);
			}
	}


	//meanplot of the parameters in the history kernel functions
	for(unsigned int d=0;d<psi.size();d++){
		unsigned int nofp=psi[d]->nofp;
			//plot the mean of the kernel parameters
		for(unsigned int p=0;p<nofp;p++){
				//plot the mean of the kernel parameter
			std::string filename=dir_path_str+"meanplot_psi_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
			plotMeans(filename,psi_posterior_samples[d][p], write_png_file, write_csv_file);
		}
	}
	
	//meanplot of the precision in the interaction weights
	if(!pt_hp.empty()){
		for(unsigned int d=0;d<pt.size();d++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[d]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"meanplot_tau_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
				plotMeans(filename,tau_posterior_samples[d][p], write_png_file, write_csv_file);
			}
		}
	}
	
	
	//meanplot of the parameters in the excitation part
	HawkesProcess::generateMCMCMeanPlots_(dir_path_str, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCAutocorrelationPlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCAutocorrelationPlots(bool write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);

}
void GeneralizedHawkesProcess::generateMCMCAutocorrelationPlots_(const std::string & dir_path_str,bool write_png_file, bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
	//autocorrelation plots of the parameters in the inhibition part
	for(unsigned int k=0;k<pt.size();k++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[k]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"autocorrelation_pt_params_type"+std::to_string(k)+"_param_"+std::to_string(p);
				plotAutocorrelations(filename,pt_posterior_samples[k][p], write_png_file, write_csv_file);
			}
	}

	//autocorrelation plots in the history kernel functions
	
	for(unsigned int d=0;d<psi.size();d++){
		unsigned int nofp=psi[d]->nofp;
			//plot the mean of the kernel parameters
		for(unsigned int p=0;p<nofp;p++){
				//plot the mean of the kernel parameter
				std::string filename=dir_path_str+"autocorrelation_psi_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
				plotAutocorrelations(filename,psi_posterior_samples[d][p], write_png_file, write_csv_file);
		}
	}
	if(!pt_hp.empty()){
		for(unsigned int d=0;d<pt.size();d++){
			//plot the trace of the kernel parameters
			unsigned int nofp=pt[d]->nofp;
			for(unsigned int p=0;p<nofp;p++){
				//plot the trace of the kernel parameter
				std::string filename=dir_path_str+"autocorrelation_tau_params_dim"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
				plotAutocorrelations(filename,tau_posterior_samples[d][p], write_png_file, write_csv_file);
			}
		}
	}
	
	//meanplot of the parameters in the excitation part
	HawkesProcess::generateMCMCAutocorrelationPlots_(dir_path_str, write_png_file, write_csv_file);	
}

void  GeneralizedHawkesProcess::generateMCMCPosteriorPlots(const std::string & dir_path_str, bool true_values, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCPosteriorPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void  GeneralizedHawkesProcess::generateMCMCPosteriorPlots(bool true_values, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCPosteriorPlots_(dir_path_str,true_values, write_png_file, write_csv_file);

}

void  GeneralizedHawkesProcess::generateMCMCPosteriorPlots_(const std::string & dir_path_str, bool true_values, bool  write_png_file, bool write_csv_file) const {
	if(!write_png_file && !write_csv_file)
		return;
	for(unsigned int k=0;k<pt.size();k++){
			//plot the posterior of the interaction weights
			unsigned int nofp=pt[k]->nofp;
			std::vector<double> true_pt_v;
			if(true_values && !pt.empty())
				pt[k]->getParams(true_pt_v);
			for(unsigned int p=0;p<nofp;p++){
				//merge the samples across all mcmc runs
				boost::numeric::ublas::vector<double> pt_posterior_samples_all=merge_ublasv(pt_posterior_samples[k][p]);
				//plot the posterior of the kernel parameter
				std::string filename=dir_path_str+"posterior_pt_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);
				plotDistributions(filename,pt_posterior_samples[k][p], write_png_file, write_csv_file);
				filename.clear();
				//plot the posterior of the kernel parameters merging the samples of different runs
				filename=dir_path_str+"all_posterior_pt_params_type_"+std::to_string(k)+"_param_"+std::to_string(p);

				if(true_values)
					plotDistribution(filename,pt_posterior_samples_all,true_pt_v[p], write_png_file, write_csv_file);//TODO: for real-world data it's not known. make it have a flag for plotting the real value from the kernel
				else
					plotDistribution(filename,pt_posterior_samples_all,write_png_file, write_csv_file);
			}
	}
	
	//plot the posterior of the parameters in the hsitory kernel functions
	for(long unsigned int d=0;d!=psi.size();d++){
		unsigned int nofp=psi[d]->nofp;
		std::vector<double> true_psi_v;
		if(true_values && !psi.empty())
			psi[d]->getParams(true_psi_v);
		for(unsigned int p=0;p<nofp;p++){
			//merge the samples across all mcmc runs
			boost::numeric::ublas::vector<double> psi_posterior_samples_all=merge_ublasv(psi_posterior_samples[d][p]);
			//plot the posterior of the kernel parameter
			std::string filename=dir_path_str+"posterior_psi_params_dim_"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
			plotDistributions(filename,psi_posterior_samples[d][p], write_png_file, write_csv_file);
			filename.clear();
			//plot the posterior of the kernel parameters merging the samples of different runs
			filename=dir_path_str+"all_posterior_psi_params_dim_"+"_"+std::to_string(d)+"_param_"+std::to_string(p);

			if(true_values)
				plotDistribution(filename,psi_posterior_samples_all,true_psi_v[p], write_png_file, write_csv_file);//TODO: for real-world data it's not known. make it have a flag for plotting the real value from the kernel
			else
				plotDistribution(filename,psi_posterior_samples_all, write_png_file, write_csv_file);
		}
	}

//	if(!pt_hp.empty()){
//		for(long unsigned int d=0;d!=pt.size();d++){
//			unsigned int nofp=pt[d]->nofp;
//			std::vector<std::vector<double>> true_tau_v;
//			pt[d]->getHyperParams(true_tau_v);
//			
//			for(unsigned int p=0;p<nofp;p++){
//				//merge the samples across all mcmc runs
//				boost::numeric::ublas::vector<double> tau_posterior_samples_all=merge_ublasv(tau_posterior_samples[d][p]);
//				//plot the posterior of the kernel parameter
//				std::string filename=dir_path_str+"posterior_tau_params_dim_"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
//				plotDistributions(filename,tau_posterior_samples[d][p], write_png_file, write_csv_file);
//				filename.clear();
//				//plot the posterior of the kernel parameters merging the samples of different runs
//				filename=dir_path_str+"all_posterior_tau_params_dim_"+"_"+std::to_string(d)+"_param_"+std::to_string(p);
//	
//				if(true_values)
//					plotDistribution(filename,tau_posterior_samples_all,1/true_tau_v[p][1], write_png_file, write_csv_file);//TODO: for real-world data it's not known. make it have a flag for plotting the real value from the kernel
//				else
//					plotDistribution(filename,tau_posterior_samples_all, write_png_file, write_csv_file);
//			}
//		}
//	}

	HawkesProcess::generateMCMCPosteriorPlots_(dir_path_str, true_values, write_png_file, write_csv_file);
}


void GeneralizedHawkesProcess::generateMCMCIntensityPlots(const std::string & dir_path_str, bool true_values, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
    generateMCMCIntensityPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCIntensityPlots(bool true_values, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	std::string dir_path_str=createModelInferenceDirectory();
    generateMCMCIntensityPlots_(dir_path_str,true_values, write_png_file, write_csv_file);
}

void GeneralizedHawkesProcess::generateMCMCIntensityPlots_(const std::string & dir_path_str, bool true_values, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	for(unsigned int k=0;k<K;k++){
		for(unsigned int seq_id=0;seq_id!=train_data.size();seq_id++)
			//plotPosteriorIntensity(dir_path_str,k,data[seq_id],true_values,seq_id, NOF_PLOT_POINTS, write_png_file, write_csv_file);
			plotPosteriorIntensity(dir_path_str,k,train_data[seq_id],false,seq_id, NOF_PLOT_POINTS, write_png_file, write_csv_file);
	}
}

void GeneralizedHawkesProcess::plotPosteriorIntensity(const std::string & dir_path_str, unsigned int k, const EventSequence &s,  bool true_intensity, unsigned int sequence_id, unsigned int nofps, bool  write_png_file, bool write_csv_file) const{
	if(!write_png_file && !write_csv_file)
		return;
	
	//compute the means and modes of the kernel parameters from the mcmc samples
	//posterior mean and mode for the constant intensity part
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
	//TODO: check if there is mutual excitation part
	//posterior mean and mode for the mutual excitation intensity part
	std::vector<std::vector<double>> mean_phi_k_param;//it keeps the posterior mean of the parameters of the triggering kernel functions of other types to type k
	std::vector<std::vector<double>> mode_phi_k_param;//it keeps the posterior mode of the parameters of the triggering kernel functions of other types to type k
	for(unsigned int k2=0;k2<phi.size();k2++){

		std::vector<double> mean_phi_k2_k_param;
		std::vector<double> mode_phi_k2_k_param;
		unsigned int nofp=phi[k2][k]->nofp;
		for(unsigned int p=0;p<nofp;p++){
				
				boost::numeric::ublas::vector<double> phi_posterior_samples_all=merge_ublasv(phi_posterior_samples[k2][k][p]);
				double mean_phi_k2_k_p_param=sample_mean(phi_posterior_samples_all);
				mean_phi_k2_k_param.push_back(mean_phi_k2_k_p_param);
				double mode_phi_k2_k_p_param=sample_mode(phi_posterior_samples_all);
				mode_phi_k2_k_param.push_back(mode_phi_k2_k_p_param);
		}
		mean_phi_k_param.push_back(mean_phi_k2_k_param);
		mode_phi_k_param.push_back(mode_phi_k2_k_param);
	}
	//posterior mean and mode for the mutual inhibition part
	nofp=pt[k]->nofp;

	std::vector<double> mean_pt_k_param;//it keeps the posterior mean estimation of the base intensity of type k
	std::vector<double> mode_pt_k_param;//it keeps the posterior mode estimation of the base intensity of type k
	for(unsigned int p=0;p<nofp;p++){
		
		boost::numeric::ublas::vector<double> pt_posterior_samples_all=merge_ublasv(pt_posterior_samples[k][p]);
		double mean_pt_k_p_param=sample_mean(pt_posterior_samples_all);
		mean_pt_k_param.push_back(mean_pt_k_p_param);
		double mode_pt_k_p_param=sample_mode(pt_posterior_samples_all);
		mode_pt_k_param.push_back(mode_pt_k_p_param);
	}


	std::vector<std::vector<double>> mean_psi_k_param;//it keeps the posterior mean of the parameters of the history kernel functions of other types to type k
	std::vector<std::vector<double>> mode_psi_k_param;//it keeps the posterior mode of the parameters of the history kernel functions of other types to type k
	for(unsigned int d=0;d<psi.size();d++){
	
		std::vector<double> mean_psi_k_d_param;
		std::vector<double> mode_psi_k_d_param;
		unsigned int nofp=psi[d]->nofp;
		for(unsigned int p=0;p<nofp;p++){
	
				boost::numeric::ublas::vector<double> psi_posterior_samples_all=merge_ublasv(psi_posterior_samples[d][p]);
				double mean_psi_k_d_p_param=sample_mean(psi_posterior_samples_all);
				mean_psi_k_d_param.push_back(mean_psi_k_d_p_param);
				double mode_psi_k_d_p_param=sample_mode(psi_posterior_samples_all);
				mode_psi_k_d_param.push_back(mode_psi_k_d_p_param);
		}
		mean_psi_k_param.push_back(mean_psi_k_d_param);
		mode_psi_k_param.push_back(mode_psi_k_d_param);
	}


	//posterior mean kernels
	
	ConstantKernel *mean_mu_k=(mu[k]->clone()); //constant poisson part

	mean_mu_k->setParams(mean_mu_k_param);

	std::vector<Kernel*> mean_phi_k(phi.size()); //mutual excitation part
	for(unsigned int k2=0;k2<phi.size();k2++){
	
		mean_phi_k[k2]=phi[k2][k]->clone();
		
		mean_phi_k[k2]->setParams(mean_phi_k_param[k2]);
	}
	LogisticKernel *mean_pt_k=pt[k]->clone(); //mutual inhibition part

	mean_pt_k->setParams(mean_pt_k_param);


	//posterior mode kernels
	ConstantKernel *mode_mu_k=(mu[k]->clone()); //constant poisson part
	mode_mu_k->setParams(mode_mu_k_param);
	std::vector<Kernel*> mode_phi_k(phi.size()); //mutual excitation part
	for(unsigned int k2=0;k2<phi.size();k2++){
		mode_phi_k[k2]=phi[k2][k]->clone();
		mode_phi_k[k2]->setParams(mode_phi_k_param[k2]);
	}
	LogisticKernel *mode_pt_k=pt[k]->clone(); //mutual inhibition part
	mode_pt_k->setParams(mode_pt_k_param);//set the inhibition weights

	//compute intensity of the posterior mode parameters
	boost::numeric::ublas::vector<double>  mode_tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  mode_lv;//the value of the intensity function
	computeIntensity(mode_mu_k,mode_phi_k,mode_pt_k,s,mode_tv,mode_lv,nofps);
	
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
		for(unsigned int i=0;i<mode_tv.size();i++){
			file<<mode_tv[i]<<","<<mode_lv[i]<<std::endl;
		}
		file.close();
	}
	
	//compute intensity of the posterior mean parameters
	boost::numeric::ublas::vector<double>  mean_tv;//time points for which the intensity function will be computed
	boost::numeric::ublas::vector<long double>  mean_lv;//the value of the intensity function
	computeIntensity(mean_mu_k,mean_phi_k,mean_pt_k, s,mean_tv,mean_lv,nofps);
	
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
		std::string filename=dir_path_str+"posterior_mean_intensity_type_"+std::to_string(k)+".csv";
		std::ofstream file{filename};
		file<<"time,value"<<std::endl;
		for(unsigned int i=0;i<mean_tv.size();i++){
			file<<mean_tv[i]<<","<<mean_lv[i]<<std::endl;
		}
		file.close();
	}


	//if the true kernel parameters are known, compute the squared error of the estimated intensities
	if(true_intensity){
		boost::numeric::ublas::vector<double>  true_tv;//time points for which the intensity function will be computed
		boost::numeric::ublas::vector<long double>  true_lv;//the value of the intensity function
		std::vector<Kernel *> true_phi_k(phi.size());
		for(unsigned int k2=0;k2<phi.size();k2++)
			true_phi_k[k2]=phi[k2][k];
		computeIntensity(mu[k],true_phi_k,pt[k], s,true_tv,true_lv,nofps);

		//compute the absolute relative error of the posterior mode estimated intensity
		boost::numeric::ublas::vector<double> error_mode_lv=true_lv-mode_lv;
		boost::numeric::ublas::vector<double> rel_error_mode_lv=element_div(error_mode_lv,true_lv);
		forced_clear(error_mode_lv);
		std::transform(rel_error_mode_lv.begin(),rel_error_mode_lv.end(),rel_error_mode_lv.begin(),[&](double x){
			return std::abs(x);
		}
		);
		
		//compute the absolute relative error of the posterior-mode intensity function
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
			for(unsigned int i=0;i<mode_tv.size();i++){
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

void  GeneralizedHawkesProcess::generateMCMCTrainLikelihoodPlots(unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file, bool normalized, const std::string & dir_path_str) const{
	if(!write_png_file && !write_csv_file)
		return;

	std::string plot_dir_path_str=dir_path_str.empty()?createModelInferenceDirectory():dir_path_str;
	for(unsigned int seq_id=0;seq_id!=train_data.size();seq_id++){
		generateMCMCLikelihoodPlot(plot_dir_path_str,samples_step, true_values, train_data[seq_id], write_png_file, write_csv_file, normalized);
	}

}

void  GeneralizedHawkesProcess::generateMCMCTestLikelihoodPlots(const std::string &seq_dir_path_str, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file, bool normalized,  double t0, double t1, const std::string & dir_path_str) const{
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
		test_seq.end_t=(t1>test_seq.start_t)?t1:last_arrival_time;
		test_seq.name=seq_prefix+std::to_string(seq_id);
		test_seq_file.close();
		
		generateMCMCLikelihoodPlot(plot_dir_path_str,samples_step, true_values, test_seq, write_png_file, write_csv_file, normalized);
	}

}


void GeneralizedHawkesProcess::generateMCMCLikelihoodPlot(const std::string & plot_dir_path_str, unsigned int samples_step, bool true_values, const EventSequence & seq, bool  write_png_file, bool write_csv_file, bool normalized) const{
	if(!write_png_file && !write_csv_file)
		return;
	//compute the likelihood with the true parameters
	double true_logl;
	if(true_values){
		if(!phi.empty())
			true_logl=loglikelihood(seq,mu,phi,pt, normalized);
		else
			true_logl=loglikelihood(seq,mu,pt, normalized);
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
		std::string filename=plot_dir_path_str+"loglikelihood_"+seq.name+".png";
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
		std::string filename=plot_dir_path_str+"loglikelihood_"+seq.name+".csv";
		std::ofstream file{filename};
		if(true_values)
			file<<"mode likelihood,mean likelihood, true likelihood"<<std::endl;
		else
			file<<"mode likelihood,mean likelihood"<<std::endl;
		for(unsigned int i=0;i<post_mode_loglikelihood.size();i++){
			if(true_values)
				file<<post_mode_loglikelihood[i]<<","<<post_mean_loglikelihood[i]<<","<<true_logl<<std::endl;
			else
				file<<post_mode_loglikelihood[i]<<","<<post_mean_loglikelihood[i]<<std::endl;
		}
		file.close();
	}
}

void *GeneralizedHawkesProcess::generateMCMCLikelihoodPlots__(void *p){
	
	//unwrap thread parameters
	std::unique_ptr<generateMCMCLikelihoodPlotThreadParams> params(static_cast<generateMCMCLikelihoodPlotThreadParams * >(p));
	const GeneralizedHawkesProcess *ghp=params->ghp;
	const EventSequence &seq=params->seq;
	unsigned int *nof_samples=params->nof_samples;
	bool *endof_samples=params->endof_samples;
	unsigned int samples_step=params->samples_step;
	std::vector<double> & post_mean_loglikelihood=params->post_mean_loglikelihood;
	std::vector<double> & post_mode_loglikelihood=params->post_mode_loglikelihood;
	pthread_mutex_t *mtx=params->mtx;
	bool normalized=params->normalized;
	
	//get number of parameters for each component of the model
	pthread_mutex_lock(mtx);
	
	unsigned int mu_nofp=ghp->mu[0]->nofp;//number of parameters for the base intensity
	unsigned int runs=ghp->mu_posterior_samples[0][0].size(); //number of mcmc runs
	unsigned int phi_nofp=0;
	
	if(!ghp->phi.empty())
		phi_nofp=ghp->phi[0][0]->nofp;//number of parameters for the mutually triggerred intensity
	
	unsigned int pt_nofp=0;
	if(!ghp->pt.empty())
		pt_nofp=ghp->pt[0]->nofp;//number of parameters for the inhibition part
	
	unsigned int psi_nofp=0;
	psi_nofp=ghp->pt[0]->psi[0]->nofp;//number of parameters for the history kernel functions

	
	//initialize the auxiliary kernels that will hold the current point estimates
	std::vector<ConstantKernel*>  post_mean_mu;
	std::vector<std::vector<Kernel *>>  post_mean_phi;
	std::vector<LogisticKernel *>  post_mean_pt;
	std::vector<Kernel *>  post_mean_psi;
	
	std::vector<ConstantKernel*>  post_mode_mu;
	std::vector<std::vector<Kernel *>>  post_mode_phi;
	std::vector<LogisticKernel *>  post_mode_pt;
	std::vector<Kernel *>  post_mode_psi;


	for(unsigned int k=0;k<ghp->mu.size();k++){
		post_mean_mu.push_back((ghp->mu[k]->clone()));
		post_mode_mu.push_back((ghp->mu[k]->clone()));
	}
	
	for(unsigned int k=0;k<ghp->phi.size();k++){
		std::vector<Kernel*> mean_phi_k(ghp->phi[k].size());
		std::vector<Kernel*> mode_phi_k(ghp->phi[k].size());

		for(unsigned int k2=0;k2<ghp->phi[k].size();k2++){
			mean_phi_k[k2]=ghp->phi[k][k2]->clone();
			mode_phi_k[k2]=ghp->phi[k][k2]->clone();
		}
		post_mean_phi.push_back(mean_phi_k);
		post_mode_phi.push_back(mode_phi_k);
	}
	
	for(long unsigned int k=0;k<ghp->pt.size();k++){
		post_mean_pt.push_back(ghp->pt[k]->clone());
		post_mode_pt.push_back(ghp->pt[k]->clone());
	}
	for(long unsigned int d=0;d<ghp->psi.size();d++){
		post_mean_psi.push_back(ghp->psi[d]->clone());
		post_mode_psi.push_back(ghp->psi[d]->clone());
	}

	pthread_mutex_unlock(mtx);
	pthread_mutex_lock(mtx);
	
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

		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> mu_samples(ghp->mu.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(mu_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs)));
		std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> phi_samples(ghp->phi.size(),std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>(ghp->phi.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(phi_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs))));
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> pt_samples(ghp->pt.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(pt_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs)));
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> psi_samples(ghp->psi.size(),std::vector<std::vector<boost::numeric::ublas::vector<double>>>(psi_nofp,std::vector<boost::numeric::ublas::vector<double>>(runs)));
		//keep only the first samples  of each mcmc run for each param
		for(long unsigned int k=0;k<ghp->mu.size();k++){
			for(unsigned int p=0;p<mu_nofp;p++){
				for(unsigned int r=0;r<runs;r++){
					mu_samples[k][p][r]=subrange(ghp->mu_posterior_samples[k][p][r],0,nof_samples_t);//todo: maybe these memorie copies cost that much????
				}
			}
		}
		for(long unsigned int k=0;k<ghp->phi.size();k++){
			for(long unsigned int k2=0;k2<ghp->phi[k].size();k2++){
				for(unsigned int p=0;p<phi_nofp;p++){
					for(unsigned int r=0;r<runs;r++){
						phi_samples[k][k2][p][r]=subrange(ghp->phi_posterior_samples[k][k2][p][r],0,nof_samples_t);

					}
				}
			}
		}
		for(long unsigned int k=0;k<ghp->pt.size();k++){
			for(unsigned int p=0;p<pt_nofp;p++){
				for(unsigned int r=0;r<runs;r++){
					pt_samples[k][p][r]=subrange(ghp->pt_posterior_samples[k][p][r],0,nof_samples_t);
				}
			}
		}
			for(long unsigned int d=0;d<ghp->psi.size();d++){
				for(unsigned int p=0;p<ghp->psi[d]->nofp;p++){
					for(unsigned int r=0;r<runs;r++){
						psi_samples[d][p][r]=subrange(ghp->psi_posterior_samples[d][p][r],0,nof_samples_t);

					}
				}
			}
		pthread_mutex_unlock(mtx);
		//compute and set the kernels to the new posterior point estimates
		if(!ghp->phi.empty()){
			GeneralizedHawkesProcess::setPosteriorParams(post_mean_mu,post_mean_phi,post_mean_pt, post_mean_psi, post_mode_mu,post_mode_phi, post_mode_pt, post_mode_psi, mu_samples,phi_samples, pt_samples, psi_samples);
			double logl=loglikelihood(seq,post_mean_mu,post_mean_phi, post_mean_pt, normalized);
			pthread_mutex_lock(mtx);
			post_mean_loglikelihood[l]=logl;
			pthread_mutex_unlock(mtx);
			logl=loglikelihood(seq,post_mode_mu,post_mode_phi, post_mode_pt, normalized);
			pthread_mutex_lock(mtx);
			post_mode_loglikelihood[l]=logl;
			pthread_mutex_unlock(mtx);
		}
		else{
			GeneralizedHawkesProcess::setPosteriorParams(post_mean_mu, post_mean_pt, post_mean_psi, post_mode_mu, post_mode_pt, post_mode_psi, mu_samples, pt_samples, psi_samples);
		
			//compute the likelihood with the posterior mean estimates
			double logl=loglikelihood(seq,post_mean_mu,post_mean_pt, normalized);
			pthread_mutex_lock(mtx);
			post_mean_loglikelihood[l]=logl;
			pthread_mutex_unlock(mtx);

			//compute the likelihood with the posterior mode estimates
			logl=loglikelihood(seq,post_mode_mu,post_mode_pt, normalized);

			pthread_mutex_lock(mtx);
			post_mode_loglikelihood[l]=logl;
			pthread_mutex_unlock(mtx);

		}
		pthread_mutex_lock(mtx);
	}
	pthread_mutex_unlock(mtx);
	return 0;
}

/****************************************************************************************************************************************************************************
 * Model Testing Methods
******************************************************************************************************************************************************************************/


void GeneralizedHawkesProcess::testLogLikelihood(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples, const std::string & true_logl_filename, bool normalized){
	testLogLikelihood(test_dir_path_str, seq_dir_path_str,seq_prefix,  nof_test_sequences, start_t, end_t, burnin_samples,  true_logl_filename, normalized);
		
}

void GeneralizedHawkesProcess::testLogLikelihood(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, double t0, double t1, unsigned int burnin_samples, const std::string & true_logl_filename, bool normalized){

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
	//accumulators for the logl metrics
	accumulator_set<double, stats<tag::mean>> mode_mean_logl;
	accumulator_set<double, stats<tag::mean>> mean_mean_logl;
	accumulator_set<double, stats<tag::mean,tag::variance>> mode_logl_error;
	accumulator_set<double, stats<tag::mean,tag::variance>> mean_logl_error;


	for(unsigned int n=0;n<nof_test_sequences;n++){
	
		//open sequence file
		std::string csv_seq_filename{seq_dir_path_str+seq_prefix+std::to_string(n)+".csv"};
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
		//test_seq.start_t=start_t;
		test_seq.start_t=t0>=0?t0:start_t;
		double last_arrival_time=test_seq.full.rbegin()->first;
		test_seq.end_t=(t1>test_seq.start_t)?t1:last_arrival_time;
		
		//test_seq.end_t=(t1>start_t)?t1:last_arrival_time;
	
		test_seq_file.close();

		//compute posterior parent for events
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
		if(!phi.empty()){
			test_seq.name="mean_"+seq_prefix+std::to_string(n)+".csv";
			post_mean_logl=loglikelihood(test_seq, post_mean_mu, post_mean_phi, post_mean_pt,normalized, t0);
		}
		else{ //there is not mutual excitation part
			test_seq.name="mean_"+seq_prefix+std::to_string(n)+".csv";
			post_mean_logl=loglikelihood(test_seq, post_mean_mu, post_mean_pt,normalized, t0);
		}
		mean_mean_logl(post_mean_logl);
		double post_mean_logl_error;
		if(!true_logl_filename.empty()){
			post_mean_logl_error=std::abs(post_mean_logl-true_logl);
			mean_logl_error(post_mean_logl_error);
		}

		//with posterior mode
		double post_mode_logl;
		if(!phi.empty()){
			test_seq.name="mode_"+seq_prefix+std::to_string(n)+".csv";
	
			post_mode_logl=loglikelihood(test_seq, post_mode_mu, post_mode_phi, post_mode_pt, normalized, t0);
		}
		else{ //there is not mutual excitation part
			test_seq.name="mode_"+seq_prefix+std::to_string(n)+".csv";
			post_mode_logl=loglikelihood(test_seq, post_mode_mu, post_mode_pt, normalized, t0);
		}
		mode_mean_logl(post_mode_logl);
		
		double post_mode_logl_error;
		if(!true_logl_filename.empty()){
			post_mode_logl_error=std::abs(post_mode_logl-true_logl);
			mode_logl_error(std::abs(post_mode_logl_error));
		}
		//print metrics to file
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


void GeneralizedHawkesProcess::testPredictions(EventSequence & seq, const std::string &test_dir_path_str, double & mode_rmse, double & mean_rmse, double & mode_errorrate, double & mean_errorrate, double t0){
	std::string name=seq.name;
	seq.name.clear();
	seq.name="mode_"+name;
	GeneralizedHawkesProcess::predictEventSequence(seq,test_dir_path_str, post_mode_mu, post_mode_phi, post_mode_pt, &mode_rmse, &mode_errorrate, t0);
	seq.name.clear();
	seq.name="mean_"+name;
	GeneralizedHawkesProcess::predictEventSequence(seq,test_dir_path_str, post_mean_mu, post_mean_phi, post_mean_pt, &mean_rmse, &mean_errorrate, t0);
	seq.name.clear();
	seq.name=name;
}

/****************************************************************************************************************************************************************************
 * Model Prediction Methods
******************************************************************************************************************************************************************************/
//it computes the probability that the next arrival after tn will happen at t
double GeneralizedHawkesProcess::computeNextArrivalTimeProbability(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi,  const std::vector<LogisticKernel*> & pt, double tn, double t){

	unsigned int K=mu.size();
	std::vector<std::vector<Kernel*>> phi_k(K,std::vector<Kernel *>(K));
	for(unsigned int k=0;k<K;k++){
		for(unsigned int k2=0;k2<K;k2++){
			phi_k[k][k2]=phi[k2][k];
		}
	}
	
	double p=0;
	for(unsigned int k=0;k<K;k++)
		p+=computeIntensity(mu[k],phi_k[k], pt[k], seq, tn, t); 
	//monte carlo integration
	auto f=[&](double t)->double{
		double s=0;
		for(unsigned int k=0;k<K;k++)
			s+=computeIntensity(mu[k],phi_k[k], pt[k], seq, tn, t); 
		
		return s;
	};

	double res2=monte_carlo_integral(f, tn, t);
	double res=p*exp(-res2);
	
	return res;
	
}


double GeneralizedHawkesProcess::predictNextArrivalTime(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, unsigned int nof_samples, unsigned int nof_threads){

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
	
	//create threads
	for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
		if(!nof_samples_thread[thread_id])
			break;
	
		predictNextArrivalTimeThreadParams* p;
		//use the history of realized and thinned events which correspond to type k2
		p= new predictNextArrivalTimeThreadParams(thread_id, nof_samples_thread[thread_id], sim_time_ps[thread_id], mu, phi, pt, seq, t_n);
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

void * GeneralizedHawkesProcess::predictNextArrivalTime_(void *p){
	
	std::unique_ptr<predictNextArrivalTimeThreadParams> params(static_cast< predictNextArrivalTimeThreadParams * >(p));
	unsigned int nof_samples=params->nof_samples;
	double &mean_arrival_time=params->mean_arrival_time;
	const std::vector<ConstantKernel *>  & mu=params->mu;
	const std::vector<std::vector<Kernel*>> & phi=params->phi;
	const std::vector<LogisticKernel*> & pt=params->pt;
	const EventSequence & seq=params->seq;
	double t_n=params->t_n;
	accumulator_set<double, stats<tag::mean>> arrival_time_acc;
	for(unsigned int r=0; r!= nof_samples;r++){
		std::vector<Event *> nxt_events;
		GeneralizedHawkesProcess::simulateNxt(mu, phi, pt, seq, t_n, seq.end_t, nxt_events, 1);
		if(!nxt_events.empty()){
			double t_nxt=nxt_events[0]->time;
			
			arrival_time_acc(t_nxt);
		}
	}
	mean_arrival_time=mean(arrival_time_acc);
	
	return 0;
}


int GeneralizedHawkesProcess::predictNextType(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, std::map<int, double> & type_prob, unsigned int nof_samples, unsigned int nof_threads){

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
		p= new predictNextTypeThreadParams(thread_id, nof_samples_thread[thread_id], *sim_type_pc[thread_id], mu, phi, pt, seq, t_n);
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
	unsigned int K=mu.size();
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

void * GeneralizedHawkesProcess::predictNextType_(void *p){
	
	std::unique_ptr<predictNextTypeThreadParams> params(static_cast< predictNextTypeThreadParams * >(p));
	unsigned int nof_samples=params->nof_samples;
	//double &mean_arrival_time=params->mean_arrival_time;
	const std::vector<ConstantKernel *>  & mu=params->mu;
	const std::vector<std::vector<Kernel*>> & phi=params->phi;
	const std::vector<LogisticKernel*> & pt=params->pt;
	std::map<int, unsigned int> & type_counts=params->type_counts; 
	const EventSequence & seq=params->seq;
	double t_n=params->t_n;
		

	for(unsigned int r=0; r!= nof_samples;r++){
		std::vector<Event *> nxt_events;
		GeneralizedHawkesProcess::simulateNxt(mu, phi, pt, seq, t_n, seq.end_t, nxt_events, 1);
		if(!nxt_events.empty()){
			int l_nxt=nxt_events[0]->type;
		
			type_counts[l_nxt]++; 
		}
	}

	return 0;
}

void GeneralizedHawkesProcess::predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, double t0, unsigned int nof_samples) {
	GeneralizedHawkesProcess::predictEventSequence(seq , seq_dir_path_str, mu,  phi, pt, 0, 0, t0);
}
//TODO: prediction tasks incomplete

void GeneralizedHawkesProcess::predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, double *rmse, double *error_rate, double t0, unsigned int nof_samples) {
	std::string predict_filename=seq_dir_path_str+seq.name+"_prediction.csv"; //in each row it will hold the mean rmse for the predicition of the arrival time of the next event for each test event sequence, one column for post mean/ post mode
	std::ofstream predict_file{predict_filename};
	
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
			if (t_n < t0 || t_n > seq.end_t)  //time step is outside the interval under consideration
				break;
		
			//check whether t_n is the last evidence in the sequence
			auto nxt_e_iter=std::next(e_iter);
			if(nxt_e_iter==seq.full.end() || !nxt_e_iter->second) //there is no evidence in the sequence after t_n to compare against
				break;
			
			
			//get error for the predicted arrival time at next step
			 //find the next event that occurs after t_n, if t_n is in the time interval under consideration: the next arrival time to be predicted
			double t_np1=nxt_e_iter->second->time;
			//find the arrival time of an event after time t_n that achieves the minimum Bayes risk 
			double th_np1=GeneralizedHawkesProcess::predictNextArrivalTime(seq,t_n,mu,phi, pt, nof_samples);
			//compare the prediction with the real occurence and write in the csv files
			double t_error=(th_np1-t_np1)*(th_np1-t_np1);
			mse_acc(t_error);
			
			//get error for the predicted event type at next step
			 //find the next event type that occurs after t_n, if t_n is in the time interval under consideration: the type of the next event to be predicted
			unsigned int l_np1=nxt_e_iter->second->type;
			//find the type of event after time t_n that achieves the minimum Bayes risk  with the mode point estimates
			std::map<int, double> type_prob;
			unsigned int lh_np1=GeneralizedHawkesProcess::predictNextType(seq,t_n,mu,phi, pt, type_prob, nof_samples);
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
	predict_file<<"all, rmse , std mse, hard error rate, soft error rate"<<std::endl;
    double rmse_1=std::sqrt(mean(mse_acc));
    if(rmse){
    	*rmse=rmse_1;
    }
    double rmse_std=variance(mse_acc);
    double error_rate_1=mean(errorrate_acc);
    if(error_rate!=0){
    	*error_rate=error_rate_1;
	predict_file<<"summary,"<<rmse_1<<","<<rmse_std<<","<<error_rate_1<<mean(soft_errorrate_acc)<<std::endl;
    }
	predict_file.close();
}


