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

#include "Kernels.hpp"
#include "stat_utils.hpp"
#include "struct_utils.hpp"

#define DEBUG 0


BOOST_CLASS_EXPORT(Kernel)
BOOST_CLASS_EXPORT(ConstantKernel)
BOOST_CLASS_EXPORT(ExponentialKernel)
BOOST_CLASS_EXPORT(RayleighKernel)

/****************************************************************************************************************************************************************************
 *
 * Kernel
 *
******************************************************************************************************************************************************************************/
//empty constructor
Kernel::Kernel():nofp(0),nofhp(0){};

Kernel::Kernel(const Kernel &s): type{s.type},nofp(s.nofp), nofhp(s.nofhp), p(s.p){
	
	//deep copy for the hyperparameters
	for(auto iter=s.hp.begin();iter!=s.hp.end();iter++){
		if((*iter))
			hp.push_back((*iter)->clone());
		else
			std::cerr<<"hyperparameters of the kernel are not properly allocated\n";
	}
	
	//deep copy for the metropolis hastings
	for(auto iter=s.mh_vars.begin();iter!=s.mh_vars.end();iter++){
		if((*iter)){
			mh_vars.push_back((*iter)->clone());
		}
		else
			mh_vars.push_back(0);
	}
}

//constructor which sets the hyperparameters of the parameters of the kernel
Kernel::Kernel(KernelType t, std::vector< const DistributionParameters *> const  & hp2): type{t}{
	nofp=hp2.size();
	nofhp=0;
	for(auto iter=hp2.begin();iter!=hp2.end();iter++){
		nofhp+=(*iter)->nofp;
		hp.push_back((*iter)->clone());
	}
}

//constructor which sets the parameters of the kernel
Kernel::Kernel(KernelType t, const std::vector<double>  & p2):  type{t}, nofp(p2.size()), p(p2){};

Kernel::Kernel(KernelType t, const std::vector< const DistributionParameters *>   & hp2,  const std::vector<double>  & p2): type{t}, p(p2){

	//check if there is a set of hyperparameters for each parameter of the kernel
	if(p2.size()!=hp2.size())
		std::cerr<<"wrong nof parameters or hyperparameters given in the constructor\n";

	nofp=hp.size();
	nofhp=0;
	for(auto iter=hp2.begin();iter!=hp2.end();iter++){
		nofhp+=(*iter)->nofp;
		hp.push_back((*iter)->clone());
	}
}

//get the hyperparameters for the parameters of the kernel
void Kernel::getHyperParams(std::vector<DistributionParameters *> & hparams) const{
	forced_clear(hparams);
	for(auto iter=hp.begin();iter!=hp.end();iter++){
		hparams.push_back((*iter)->clone());
	}
}

void Kernel::getHyperParams(std::vector<std::vector<double>> & hparams) const{
	forced_clear(hparams);
	hparams.resize(hp.size());
	for(unsigned int i=0;i!=hp.size();i++){
		hp[i]->getParams(hparams[i]);
	}
}


//set the hyperparameters for the parameters of the kernel
void Kernel::setHyperParams(const std::vector< DistributionParameters *> & hparams){
	forced_clear(hp);
	for(auto iter=hparams.begin();iter!=hparams.end();iter++){
		hp.push_back((*iter)->clone());
	}
}


//get the parameters for the parameters of the kernel
void Kernel::getParams(std::vector<double> & params) const{
	forced_clear(params);
	for(auto iter=p.begin();iter!=p.end();iter++)
		params.push_back(*iter);
}


//set the parameters of the kernel
void Kernel::setParams(const std::vector<double> & params){
	forced_clear(p);
	for(auto iter=params.begin();iter!=params.end();iter++)
		p.push_back(*iter);
}


void Kernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}


void Kernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}



void Kernel::reset(){
	forced_clear(p);
}

void Kernel::generate(){
	std::random_device rd;
	std::mt19937 gen(rd());
	forced_clear(p);
	//draw a value for each parameter of the kernel from its prior distribution
	for(auto iter=hp.begin();iter!=hp.end();iter++){
		if(!(*iter))
			std::cerr<<"prior uninitialized\n";
		switch((*iter)->type){
			case Distribution_Gamma:{
				// todo: check if the kernel is hierarchical
				//sample from a gamma distribution

				std::gamma_distribution<double> gamma_distribution(((GammaParameters *)(*iter))->alpha, 1/((GammaParameters *)(*iter))->beta);
				double p2=gamma_distribution(gen);
		
				p.push_back(p2);
				break;
			}
			case Distribution_Exponential:{
				// todo: check if the kernel is hierarchical
				//sample from a exponential distribution
				std::exponential_distribution<double> exp_distribution(((ExponentialParameters *)(*iter))->lambda);
				p.push_back(exp_distribution(gen));
				break;
			}
			case Distribution_Normal:{
				NormalParameters *nhp=(NormalParameters *)(*iter);
				if(nhp->hp.size()==1){	//check if the kernel is hierarchical, constant (zero mean) and random variance for the parameter
					nhp->mu=0.0;
					if(!nhp->hp[0])
						std::cerr<<"uninitialized hyperprior\n";
					switch(nhp->hp[0]->type){//chose the proper type of the prior for the variance
						case Distribution_NormalGamma:{//normal-gamma prior for the precision-weight of the log kernel
							normal_gamma(*((NormalGammaParameters *)nhp->hp[0]), nhp->mu, nhp->tau);
							nhp->sigma=sqrt(1/nhp->tau);
						}
						break;
						default:
							break;
					}
				}
				//sample from a normal distribution
				std::normal_distribution<double> norm_distribution(nhp->mu,nhp->sigma);
				double w=norm_distribution(gen);
				p.push_back(w);

				break;
			}
			default:
				break;
		}

	}
}

void Kernel::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	ar & nofp;
	ar & nofhp;
	ar & hp;
	ar & p;
	ar & mh_vars;
}

void Kernel::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	ar & nofp;
	ar & nofhp;
	ar & hp;
	ar & p;
	ar & mh_vars;
}

void Kernel::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	ar & nofp;
	ar & nofhp;
	ar & hp;
	ar & p;
	ar & mh_vars;
}

void Kernel::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	ar & nofp;
	ar & nofhp;
	ar & hp;
	ar & p;
	ar & mh_vars;
}


Kernel::~Kernel(){
	//release the parameters
	forced_clear(p);
	//release the priors of the parameters
	for(auto iter=hp.begin();iter!=hp.end();iter++){
		delete (*iter);
	}
	forced_clear(hp);
	//release metropolis hastings parameters
	for(auto iter=mh_vars.begin();iter!=mh_vars.end();iter++){
		delete (*iter);
		*iter=0;
	}
	forced_clear(mh_vars);
}

/****************************************************************************************************************************************************************************
 *
 * Constant Kernel
 *
******************************************************************************************************************************************************************************/

ConstantKernel::ConstantKernel(){}

//constructor which sets the hyperparameters of the kernel
ConstantKernel::ConstantKernel(const std::vector<const DistributionParameters *>  & hp):Kernel(Kernel_Constant, hp){
	if(hp.size()!=1){
		std::cerr<<"wrong number of hyperparameters in the constructor for the constant kernel\n";
	} 
}

//constructor which sets the point parameters of the kernel
ConstantKernel::ConstantKernel(const std::vector<double>   & p):Kernel(Kernel_Constant, p){
	if(p.size()!=1)
		std::cerr<<"wrong number of hyperparameters in the constructor for the constant kernel\n";
}

//constructor which sets both the hyperparameters and the parameters of the kernel
ConstantKernel::ConstantKernel(const std::vector<const DistributionParameters *>   & hp,  const std::vector<double>  & p):Kernel(Kernel_Constant, hp,p){
	if(hp.size()!=1|| p.size()!=1)
		std::cerr<<"wrong number of hyperparameters or parameters in the constructor for the constant kernel\n";
}


ConstantKernel::ConstantKernel(const ConstantKernel &s):Kernel(s){}

ConstantKernel *ConstantKernel::clone() const{
	 return new ConstantKernel(*this);
}

void ConstantKernel::print(std::ostream & file) const{

	if(hp.size()!=1 || p.size()!=1){
		std::cerr<<"constant kernel: wrong number of parameters or hyperparameters\n";
		return ;
	}
	if(!hp[0]){
		std::cerr<<"uninitialized prior for the kernel parameters\n";
		return ;
	}
	
	file<<"prior of constant\n";
	hp[0]->print(file);
	file<<"c: "<<p[0]<<std::endl;
}

void ConstantKernel::printMatlabExpression(double t0, std::string & matlab_expr) const{
	matlab_expr.clear();
	matlab_expr=std::to_string(p[0]);
}


double ConstantKernel::compute(double t, double t0) const{
	return p[0];
}

double ConstantKernel::compute(double t) const{
	return p[0];
}


double ConstantKernel::computeMaxValue(double s, double e, double t0) const{
	return p[0];
}

double ConstantKernel::computeIntegral(double ts, double te, double t0) const{ 
	if (ts>te)
		throw std::exception();
	return p[0]*(te-ts);
}

//todo: create similar overloaded functions for mcmc states of new models....
void ConstantKernel::mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads ) {
	
	switch (hp[0]->type){
		case Distribution_Gamma:{//gamma prior is assumed for the constant kernel
			std::random_device rd;
			std::mt19937 gen(rd());
			unsigned int N_0_k=0;

			for(unsigned int l=0;l!=s->seq.size();l++){
				auto ve=s->seq[l].type[s->K].begin();//access the virtual event of the process, which corresponds to the background process

				if(ve->second->offsprings){
					N_0_k+=ve->second->offsprings->type[k].size();
					N_0_k+=ve->second->offsprings->thinned_type[k].size();
				}
			}
			std::gamma_distribution<double> gamma_distribution(((GammaParameters *)hp[0])->alpha+N_0_k,1/(((GammaParameters *)hp[0])->beta+(s->end_t-s->start_t)*s->seq.size()));

			p[0]=gamma_distribution(gen);
			break;
		}
		default:
			break;
	}
}

void ConstantKernel::mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads ) {}


void ConstantKernel::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ConstantKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ConstantKernel::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ConstantKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ConstantKernel::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ConstantKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ConstantKernel::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ConstantKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ConstantKernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void ConstantKernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

/****************************************************************************************************************************************************************************
 *
 * Exponential Kernel
 *
******************************************************************************************************************************************************************************/

ExponentialKernel::ExponentialKernel(){}

ExponentialKernel::ExponentialKernel(const std::vector<const DistributionParameters *>  & hp2): Kernel(Kernel_Exponential, hp2){
	if(hp2.size()!=2)
		std::cerr<<"wrong number of hyperparameters\n";

};

ExponentialKernel::ExponentialKernel( const std::vector<const DistributionParameters *>   & hp2,  const std::vector<double>  & p2): Kernel(Kernel_Exponential, hp2,p2){
	if(hp2.size()!=2 || p2.size()!=2){
		std::cerr<<"exp kernel: wrong number of parameters or hyperparameters\n";
		return;
	}

};


ExponentialKernel::ExponentialKernel( const std::vector<double>  & p2): Kernel(Kernel_Exponential,p2){
	if(p2.size()!=2)
		std::cerr<<"wrong number of parameters \n";
};


ExponentialKernel::ExponentialKernel(const ExponentialKernel &s):Kernel(s){
}

void ExponentialKernel::print(std::ostream & file) const{
	file<<"exponential kernel\n";
	file<<"prior for the multiplicative coefficient\n";
	hp[0]->print(file);
	if(p.size()>=1)
		file<<"value for the multiplicative coefficient "<<p[0]<<std::endl;
	file<<"prior for the decaying coefficient\n";
	hp[1]->print(file);
	if(p.size()>=2)
		file<<"value for the decaying coefficient "<<p[1]<<std::endl;
	if(!mh_vars.empty()){
		if(mh_vars[0])
			mh_vars[0]->print(file);
		
		if(mh_vars[1])
			mh_vars[1]->print(file);
	}

};

void ExponentialKernel::printMatlabExpression(double t0, std::string & matlab_expr) const{
	if(p.size()!=2)
		std::cerr<<"wrong number of kernel parameters\n";

	matlab_expr.clear();
	matlab_expr=std::to_string(p[0])+"*exp(-"+std::to_string(p[1])+".*(heaviside(t-"+std::to_string(t0)+").*(t-"+std::to_string(t0)+"))).*heaviside(t-"+std::to_string(t0)+")";
}

double ExponentialKernel::compute(double t, double t0) const{
	if(p.size()<2){
		std::cerr<<"wrong number of parameters for the kernel "<<p.size();
	}
	
	double v=p[0]*exp(-p[1]*(t-t0));
	
	
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double ExponentialKernel::compute(double t) const{
	double v=p[0]*exp(-p[1]*t);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double ExponentialKernel::computeMaxValue(double s, double e, double t0) const{
	if (s>e)
		throw std::exception();
	 double v=p[0]*exp(-p[1]*(s-t0));
	 return v;
}

double ExponentialKernel::computeIntegral(double ts, double te, double t0) const{ 
	if (ts>te)
		throw std::exception();
	double v=(p[0]/p[1])*(exp(-p[1]*(ts-t0))-exp(-p[1]*(te-t0)));
	return v>KERNEL_EPS?v:KERNEL_EPS;
}


void ExponentialKernel::mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads){


	//todo: check the type of the prior for both cases
		
		switch(hp[0]->type){//type of prior for the excitatory coefficient
			case Distribution_Gamma:{ //conjugate update is feasible
				unsigned int N_k_k2=0;
				double e=0.0;
				double d_inv=1/p[1];
				for(unsigned int l=0;l!=s->seq.size();l++){
					N_k_k2+=s->seq[l].countTriggeredEvents(k,k2);//number of events of type k2 trigerred by event of type k
					for(auto e_iter=s->seq[l].type[k].begin();e_iter!=s->seq[l].type[k].end();e_iter++)
						e+=d_inv*(1-exp(-p[1]*(s->end_t-e_iter->second->time)));
				}
				
				double alpha=((GammaParameters *)hp[0])->alpha+N_k_k2;
				double beta=((GammaParameters *)hp[0])->beta+e;
				
				std::random_device rd;
				std::mt19937 gen(rd());
				std::gamma_distribution<double> gamma_distribution(alpha,1/beta);
				p[0]=gamma_distribution(gen);
				break;
			}
			case Distribution_Exponential:{ //metropolis-hastings update with prior as proposal
				
			
				auto likelihood_ratio_0=[&](double new_p)->double{
					
					Kernel *phi_new=clone();
					phi_new->p[0]=new_p;
					
					double newl=1.0;
					for(unsigned int l=0;l!=s->seq.size();l++){
						newl*=HawkesProcess::likelihood(s->seq[l], *phi_new,k,k2);
					}
					
					double oldl=1.0;
					for(unsigned int l=0;l!=s->seq.size();l++){
						oldl*=HawkesProcess::likelihood(s->seq[l], *this,k,k2);
					}	
					
					return newl/oldl;
					
				};
				
				bool accept_sample=false;
				double sample;
				accept_sample=MetropolisHastings(new ExponentialParameters(((ExponentialParameters *)hp[0])->lambda), likelihood_ratio_0, sample);
				p[0]=accept_sample?sample:p[0];
				
				break;
			}
			default:
				std::cerr<<"unspecified type of prior for the multiplicative coefficient of the excitatory kernel\n";
				break;	
		}

	
}

//it updates the decaying coeficient of the exponential kernel
void ExponentialKernel::mcmcExcitatoryExpUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads){


	//if the it is the first iteration, allocate the auxiliary variables for the metropolis hastings
	if(!s->mcmc_iter){
		if(mh_vars.size()!=p.size())
			mh_vars.resize(p.size());
		mh_vars[1]=new AdaptiveMetropolisHastingsState(s->mcmc_params->d_t_am_sd, s->mcmc_params->d_t_am_t0,  s->mcmc_params->d_t_am_c0, AM_EPSILON);
		mh_vars[1]->mu_t=p[1];
	}
	//lambda expression for computing the likelihood ratio, with the new and old decaying coefficient for the excitatory kernels
	unsigned int pid=1;
	
	auto likelihood_ratio_1=[&](double new_p)->double{
		//create a kernel with the proposed value new_p as decaying coefficient
		Kernel *phi_new=clone();
		phi_new->p[pid]=new_p;
		//ratio of the likelihood for the new and old sample
		double l_ratio=1.0;
	
		

		//ratio of the hawkes process
		for(unsigned int l=0;l!=s->seq.size();l++){
			l_ratio*=(HawkesProcess::likelihood(s->seq[l], *phi_new,k,k2)/HawkesProcess::likelihood(s->seq[l], *this,k,k2));
		}

				
		switch(hp[pid]->type){
			case Distribution_Exponential: {//assume an exponential prior for the decaying coefficient
				
				double new_prior=exp_pdf(((ExponentialParameters *)hp[pid])->lambda,phi_new->p[pid]); //todo: switch cases for the type of the prior
				double old_prior=exp_pdf(((ExponentialParameters *)hp[pid])->lambda,p[pid]);

				if(new_prior<EPS) return EPS;
				l_ratio*=new_prior/old_prior;
				break;
			}
			case Distribution_Gamma:{ //assume a gamma prior for the decaying coefficient
				
				double new_prior=gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, phi_new->p[pid]);
				if(new_prior<EPS) return EPS;
				l_ratio*=new_prior/gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, p[pid]);
				break;
			}
			default:
				break;
		}
	
		
		
		delete phi_new;
		return l_ratio;
	};

	mh_vars[pid]->mcmcUpdate();
	double sample;
	double sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	bool accept=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t, sigma), likelihood_ratio_1, sample);


	p[pid]=accept?sample:p[pid];

	mh_vars[pid]->mu_t=p[pid];
}

//mcmc update for the exponential kernel when it is used in the excitatory part of the process
void ExponentialKernel::mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads){
	//update the multiplicative coefficient of the exponential kernel
	mcmcExcitatoryCoeffUpdate(k, k2, s, nof_threads);
	//update the exponential coefficient of the exponential kernel
	mcmcExcitatoryExpUpdate(k, k2, s, nof_threads);
}

//mcmc update for the exponential kernel when it is used in the inhibition part of the process as history kernel
void ExponentialKernel::mcmcInhibitoryUpdate(int l, int k, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads) {
	
	//if it is the first mcmc step, allocate the metropolis hastings variables for the multiplicative and decaying coefficient
	if(!ghs->mcmc_iter){
		mh_vars.resize(p.size());
		mh_vars[0]=new AdaptiveMetropolisHastingsState(ghs->mcmc_params->c_h_am_sd, ghs->mcmc_params->c_h_am_t0, ghs->mcmc_params->c_h_am_c0, AM_EPSILON);
		mh_vars[0]->mu_t=p[0];
		mh_vars[1]=new AdaptiveMetropolisHastingsState(ghs->mcmc_params->d_h_am_sd, ghs->mcmc_params->d_h_am_t0, ghs->mcmc_params->d_h_am_c0, AM_EPSILON);
		mh_vars[1]->mu_t=p[1];
	}
	
	unsigned int pid=0;
	auto likelihood_ratio=[&](double new_p)->double{
		//allocate a sigmoid kernel, with a history kernel function for dimension l with value new_p for the parameter pid
		Kernel *psi_new=this->clone();
		psi_new->p[pid]=new_p;
		
		//assign the new history kernel function to each sigmoid kernel of the process
		std::vector<LogisticKernel *> pt_new;
		for(auto iter=ghs->pt.begin();iter!=ghs->pt.end();iter++){
			LogisticKernel *pt_new_k=(*iter)->clone();
			pt_new_k->psi[l]=psi_new;
			pt_new.push_back(pt_new_k);
		}
		double l_ratio=1.0;
		double new_prior_ratio;
		switch(hp[pid]->type){
			case Distribution_Exponential:{ //assume an exponential prior for the decaying coefficient
					new_prior_ratio=exp_pdf(((ExponentialParameters *)hp[pid])->lambda,new_p);
					if(new_prior_ratio<EPS) return EPS;
					l_ratio*=new_prior_ratio/exp_pdf(((ExponentialParameters *)hp[pid])->lambda,p[pid]);
					break;
			}
			case Distribution_Gamma: {//assume a gamma prior for the decaying coefficient
					new_prior_ratio=gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, new_p);
					if(new_prior_ratio<EPS) return EPS;
					l_ratio*=new_prior_ratio/gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, p[pid]);
					break;
			}
			default:
				break;
		}
		//
		//	//get the ratio of the likelihoods of the thinning procedure for the events of type k
		for(unsigned int i=0;i!=s->seq.size();i++)
	
			l_ratio*=(GeneralizedHawkesProcess::likelihood(s->seq[i], pt_new)/GeneralizedHawkesProcess::likelihood(s->seq[i], ghs->pt));

		delete psi_new;
		for(auto iter=pt_new.begin();iter!=pt_new.end(); iter++)
			delete (*iter);
		forced_clear(pt_new);

		return l_ratio;
		
	};
	//metropolis hastings update for the multiplicative coefficient
	double sample;
	mh_vars[pid]->mcmcUpdate();
	double sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	p[pid]=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t,sigma), likelihood_ratio, sample)?sample:p[pid];
	mh_vars[pid]->mu_t=p[pid];
	
	pid=1;
	mh_vars[pid]->mcmcUpdate();
	sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	p[pid]=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t, sigma), likelihood_ratio, sample)?sample:p[pid];
	mh_vars[pid]->mu_t=p[pid];
}


ExponentialKernel *ExponentialKernel::clone() const{
	 return new ExponentialKernel(*this);
}

void ExponentialKernel::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ExponentialKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ExponentialKernel::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ExponentialKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ExponentialKernel::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ExponentialKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ExponentialKernel::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<ExponentialKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void ExponentialKernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void ExponentialKernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}
/****************************************************************************************************************************************************************************
 *
 * Logistic Kernel Function
 *
******************************************************************************************************************************************************************************/
void LogisticKernel::printMatlabExpression(double t0, std::string & matlab_expr) const{

}

LogisticKernel::LogisticKernel(){}

//copy constructor
LogisticKernel::LogisticKernel(const LogisticKernel &s):Kernel(s){
	//set the number of basis kernels
	d=s.d;
	//deep copy of the basis functions
	//copy of private variables

	if(s.Sigma_tmp){
		Sigma_tmp=gsl_matrix_calloc(nofp, nofp);
		gsl_matrix_memcpy (Sigma_tmp, s.Sigma_tmp);
	}
	if(s.mu_tmp){
		mu_tmp=gsl_vector_calloc(nofp);
		gsl_vector_memcpy (mu_tmp, s.mu_tmp);
	}
}

//constructor which sets the hyperparameters and the basis kernel functions
LogisticKernel::LogisticKernel(const std::vector<const DistributionParameters *>  & hp2): Kernel(Kernel_Logistic,hp2), d(hp2.size()-1){}

//constructor which sets the hyperparameters, the point parameters and the basis kernel functions
LogisticKernel::LogisticKernel(const std::vector<const DistributionParameters *>   & hp2, const std::vector<double> & p2): Kernel(Kernel_Logistic, hp2, p2), d(hp2.size()-1){}


//constructor which sets the point parameters and the basis functions of the kernel
LogisticKernel::LogisticKernel(const std::vector<double> & p2):Kernel(Kernel_Logistic, p2), d(p2.size()-1){}


//constructor which sets the hyperparameters and the basis kernel functions
LogisticKernel::LogisticKernel(const std::vector<const DistributionParameters *>  & hp2,  std::vector<Kernel *> const & psi2): Kernel(Kernel_Logistic,hp2), d(psi2.size()), psi(psi2){}

//constructor which sets the hyperparameters, the point parameters and the basis kernel functions
LogisticKernel::LogisticKernel(const std::vector<const DistributionParameters *>   & hp2, const std::vector<double> & p2, const std::vector<Kernel *> & psi2): Kernel(Kernel_Logistic, hp2, p2), d(psi2.size()), psi(psi2){}


//constructor which sets the point parameters and the basis functions of the kernel
LogisticKernel::LogisticKernel(const std::vector<double> & p2, const std::vector< Kernel *>  & psi2):Kernel(Kernel_Logistic, p2), d(psi2.size()), psi(psi2){}


void LogisticKernel::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<LogisticKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
    ar & d;
    ar & psi;

}

void LogisticKernel::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<LogisticKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
    ar & d;
    ar & psi;
}

void LogisticKernel::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<LogisticKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
    ar & d;
    ar & psi;
}

void LogisticKernel::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<LogisticKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
    ar & d;
    ar & psi;
}

void LogisticKernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void LogisticKernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}


void LogisticKernel::print(std::ostream & file) const{
	file<<"dimension of kernel functions "<<d<<std::endl;
	file<<"normal priors \n";
	for(auto iter=hp.begin();iter!=hp.end();iter++){
		(*iter)->print(file);
	}
	file<<"weight values\n";
	for(auto iter=p.begin();iter!=p.end();iter++)
		file<<*iter<<std::endl;
	file<<"basis kernel functions\n";
	for(auto iter=psi.begin();iter!=psi.end();iter++)
		(*iter)->print(file);

}

void LogisticKernel::generate(){
	//sample the weights
	Kernel::generate();

}

void LogisticKernel::reset(){
	//reset the weights
	Kernel::reset();
}


LogisticKernel *LogisticKernel::clone() const{
	LogisticKernel * cpy=new LogisticKernel(*this);
	 for(long unsigned int d=0;d!=psi.size();d++){
		 if(psi[d])
			 cpy->psi.push_back(psi[d]->clone());
	 }
	 return cpy;
}

LogisticKernel::~LogisticKernel(){
	//release basis kernel functions
//	for(auto iter=psi.begin();iter!=psi.end();iter++)
//		delete (*iter);
//	forced_clear(psi);
	//the destructor of the basis class was declared virtual: it is automatically called

}

double LogisticKernel::compute(double t, double t0) const{
	if(p.empty())
		std::cerr<<"unitialized kernel weights\n";
	if(p.size()!=d+1||psi.size()!=d)
		std::cerr<<"invalid weight or kernel function dimension\n";
	
	double v=p[0];
	unsigned int nofks=psi.size();//number of kernel functions 

	for(unsigned int d=0;d!=nofks;d++){
		if(!psi[d])
			std::cerr<<"uninitialized basis kernel \n";
		v+=p[d+1]*psi[d]->compute(t,t0);
	}

	v=sigmoid(v);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double LogisticKernel::compute(double t) const{
	
	if(p.size()!=d+1){
		std::cerr<<"wrong number of weights\n";
		std::cerr<<d<<std::endl;
		std::cerr<<p.size()<<std::endl;
	}
	
	if(psi.size()!=d){
		std::cerr<<"wrong number of basis functions\n";
		std::cerr<<psi.size()<<std::endl;
	}

	double v=p[0];
	for(unsigned int d=0;d!=psi.size();d++){
		v+=p[d+1]*psi[d]->compute(t);
	}

	v=sigmoid(v);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double LogisticKernel::compute(double t, std::vector<std::vector<double>> t0) const{
	if(p.size()!=d+1){
		std::cerr<<"wrong number of weights\n";
		std::cerr<<d<<std::endl;
		std::cerr<<p.size()<<std::endl;
	}

	if(psi.size()!=d){
		std::cerr<<"wrong number of basis functions\n";
		std::cerr<<psi.size()<<std::endl;
	}

	if(t0.size()!=d)
		std::cerr<<"wrong number of time points\n";
	
	double v=p[0];
	unsigned int nofks=psi.size();//number of kernel functions 

	for(unsigned int d=0;d!=nofks;d++){
		double h=0;
		if(!psi[d])
			std::cerr<<"invalid kernel function\n";
		for( auto iter=t0[d].begin();iter!=t0[d].end();iter++){
			h+=psi[d]->compute(t,*iter); 
		}
		v+=p[d+1]*h;
	}
	v=sigmoid(v);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}


double LogisticKernel::compute(const double *h) const{

	if(!h)
		std::cerr<<"unitialized kernel history\n";
	
	if(p.size()!=d+1){
		std::cerr<<"wrong number of weights\n";
		std::cerr<<d<<std::endl;
		std::cerr<<p.size()<<std::endl;
	}
	
	if(psi.size()!=d){
		std::cerr<<"wrong number of basis functions\n";
		std::cerr<<d<<std::endl;
		std::cerr<<psi.size()<<std::endl;
	}
	double v=p[0];//add the bias term
	for(unsigned int d=0;d!=psi.size();d++){
		v+=p[d+1]*h[d];//compute the weighted history  TODO: account for multiple types/ kernel functions
	}
	v=sigmoid(v);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double LogisticKernel::computeArg(const double *h) const{
	if(!h)
		std::cerr<<"unitialized kernel history\n";
	
	if(p.size()!=d+1){
		std::cerr<<"wrong number of weights\n";
		std::cerr<<d<<std::endl;
		std::cerr<<p.size()<<std::endl;
	}
	
	if(psi.size()!=d)
		std::cerr<<"wrong number of basis functions\n";
	
	unsigned int nofks=psi.size();//number of kernel functions 
	double v=p[0];
	for(unsigned int d=0;d!=nofks;d++){
		v+=p[d+1]*h[d];
	}
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

//todo: do it numerically
double LogisticKernel::computeMaxValue(double s, double e, double t0) const{
return 0;
}


//todo: do it numerically
double LogisticKernel::computeIntegral(double ts, double te, double t0) const{
return 0;
}


void LogisticKernel::mcmcWeightsUpdate(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads){

	//form the posterior covariance matrix for the multivariate Gaussian of the weights
	gsl_matrix *Sigma;
	mcmcCovUpdate2(s,k2,&Sigma,0,0,nof_threads);

	//form the posterior mean for the multivariate Gaussian of the weights
	gsl_vector *mu;
	mcmcMeanUpdate2(s,k2,Sigma,&mu,nof_threads);

	//sample the  weights from the multivariate Gaussian
	gsl_linalg_cholesky_decomp1(Sigma);//get the cholesky decomposition of the covariance matrix
	multivariate_normal(mu, Sigma, p);//draw a sample from the multivariate normal
		
	gsl_matrix_free(Sigma);
	gsl_vector_free(mu);
}
	 
void LogisticKernel::mcmcCovUpdate1(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads){
	
	//form the matrix HP=Sigma^-1+H^T*Omega*H
	if(Sigma_tmp)
		gsl_matrix_free(Sigma_tmp);
	Sigma_tmp=gsl_matrix_calloc(nofp,nofp); //the symmetric matrix which contains the polyagamma and kernel history of events
	//TODO:: keep history of events of type k2 only here!!!

	
	unsigned int L=s.polyagammas.size();//nof training event sequences
	for(unsigned int i=0;i<nofp;i++){
		for(unsigned int j=0;j<=i;j++){
			double Sigma_ij=0;
			for(unsigned int l=0;l!=L;l++){
			
				unsigned int N=s.polyagammas[l][k2].size();//number of realized events of type k2
				unsigned int Nt=s.thinned_polyagammas[l][k2].size();//number of thinned events of type k2
				pthread_t mcmc_threads[nof_threads];
				unsigned int nof_events_thread[nof_threads];
				double *Sigmaij_ps=(double*) calloc (nof_threads,sizeof(double));
				//distribute the events across the threads
				unsigned int nof_events_thread_q=(N+Nt)/nof_threads;
				unsigned int nof_events_thread_m=(N+Nt)%nof_threads;
				for(unsigned int t=0;t<nof_threads;t++){
					nof_events_thread[t]=nof_events_thread_q;
					if(t<nof_events_thread_m)
						nof_events_thread[t]++;
				}
				
				//create threads
				unsigned int event_id_offset=0;
		
				for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
					if(!nof_events_thread[thread_id])
						break;
				
					FormCovThreadParams* p;
					//use the history of realized and thinned events which correspond to type k2
					p= new FormCovThreadParams(this, thread_id, event_id_offset,nof_events_thread[thread_id], s.polyagammas[l][k2], s.thinned_polyagammas[l][k2],const_cast<const double **> (s.h[l][k2]), const_cast<const double **> (s.thinned_h[l][k2]), i,j, Sigmaij_ps[thread_id]);
					int rc = pthread_create(&mcmc_threads[thread_id], NULL, mcmcCovUpdate_, (void *)p);
					if (rc){
						 std::cerr<< "Error:unable to create thread," << rc << std::endl;
						 exit(-1);
					}
					event_id_offset+=nof_events_thread[thread_id];
				
				}
				//wait for all the partial sums to be computed
				for (unsigned int t = 0; t <nof_threads; t++){
					if(nof_events_thread[t])
						pthread_join (mcmc_threads [t], NULL);
				}
				
			
				for(unsigned int t = 0; t <nof_threads; t++){
					if(nof_events_thread[t])
						Sigma_ij+=Sigmaij_ps[t];//TODO, TODO, TODO: uncomment this
				}
			}
			gsl_matrix_set(Sigma_tmp,i,j,Sigma_ij);
			gsl_matrix_set(Sigma_tmp,j,i,Sigma_ij);//the matrix is symmetric
		}
	}
}

void LogisticKernel:: mcmcCovUpdate2(GeneralizedHawkesProcess::State & s, int k2, gsl_matrix **Sigma, gsl_matrix **Tau, double *Tau_det, unsigned int nof_threads){
	//copy the intermediate result H^TOmegaH

	*Sigma=gsl_matrix_calloc(nofp,nofp); 
	gsl_matrix_memcpy (*Sigma, Sigma_tmp);

			
	if(hp.size()!=nofp)
		std::cerr<<"wrong number of hyperparamters\n";

	//add the contribution of the prior
	for(unsigned int i=0;i<nofp;i++){
		double Sigma_ii=gsl_matrix_get(Sigma_tmp, i,i);
		Sigma_ii+=((NormalParameters *)hp[i])->tau;
		gsl_matrix_set(*Sigma, i,i, Sigma_ii);
	}

	//invert the precision matrix to get the covariance matrix
	if(Tau){
		*Tau=gsl_matrix_calloc(nofp,nofp); 
		gsl_matrix_memcpy (*Tau, *Sigma);
	}
	gsl_linalg_cholesky_decomp1(*Sigma);
	if(Tau_det)
		*Tau_det=gsl_cholesky_det(*Sigma);
	gsl_linalg_cholesky_invert(*Sigma);
}

void * LogisticKernel::mcmcCovUpdate_(void *p){
	//unwrap thread parameters
	std::unique_ptr<FormCovThreadParams> params(static_cast< FormCovThreadParams * >(p));
	unsigned int event_id_offset=params->thread_id;
	unsigned int nof_events=params->nof_events;
	const std::vector<double> & polyagammas=params->polyagammas;
	const std::vector<double> & thinned_polyagammas=params->thinned_polyagammas;
	const double ** thinned_h=params->thinned_h;
	const double  **h=params->h;
	unsigned int i=params->i;
	unsigned int j=params->j;
	double &Sigmaij=params->Sigmaij;
	
	unsigned int Nt=thinned_polyagammas.size();//number of thinned events
	unsigned int N=polyagammas.size();//number of observed events
	
	Sigmaij=0;
	for(unsigned int event_id=event_id_offset;event_id<event_id_offset+nof_events;event_id++){
		if(event_id>=Nt+N){
			std::cerr<<"invalid event id!\n";
		}
		
		if(event_id>=N){//the event is a thinned event
			unsigned int thinned_event_id=event_id-N;
			double hi=(i==0)?1:thinned_h[thinned_event_id][i-1];
			double hj=(j==0)?1:thinned_h[thinned_event_id][j-1];
			Sigmaij+=hi*hj*thinned_polyagammas[thinned_event_id];	
		}
		else{//the event is observed
			double hi=(i==0)?1:h[event_id][i-1];
			double hj=(j==0)?1:h[event_id][j-1];
			Sigmaij+=hi*hj*polyagammas[event_id];
		}	
	}
	return 0;
}
	//it updates the mean of of the mutlivariate Gaussian distrbution for the weights in the logistic kernel
void LogisticKernel::mcmcMeanUpdate1(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads){
	
	unsigned int L=s.polyagammas.size();
	//gsl_vector *mu_tmp=gsl_vector_calloc(nofp);
	
	if(mu_tmp)
		gsl_vector_free(mu_tmp);
	mu_tmp=gsl_vector_calloc(nofp);

	//form (Sigma^-1*mu+H^T*z) vector
	for(unsigned int i=0;i<nofp;i++){
		double mu_i=0;
		
		for(unsigned int l=0;l!=L;l++){

			unsigned int N=s.polyagammas[l][k2].size();//number of realized events of type k2
			unsigned int Nt=s.thinned_polyagammas[l][k2].size();//number of thinned events of type k2
			pthread_t mcmc_threads[nof_threads];
			unsigned int nof_events_thread[nof_threads];
	
			double *mui_ps=(double*) calloc (nof_threads,sizeof(double));
			//distribute the events across the threads
			unsigned int nof_events_thread_q=(N+Nt)/nof_threads;
			unsigned int nof_events_thread_m=(N+Nt)%nof_threads;
			
			for(unsigned int t=0;t<nof_threads;t++){
				nof_events_thread[t]=nof_events_thread_q;
				if(t<nof_events_thread_m)
					nof_events_thread[t]++;
			}
			
			//create threads
			unsigned int event_id_offset=0;
			//split the types across the threads, each thread updates the kernel parameters for a batch of types
			for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
				if(!nof_events_thread[thread_id])
					break;
			
				FormMeanThreadParams* p;
				//use the history of realized and thinned events which correspond to type k2
				p= new FormMeanThreadParams(this, thread_id, event_id_offset,nof_events_thread[thread_id], s.polyagammas[l][k2], s.thinned_polyagammas[l][k2], const_cast<const double **>(s.h[l][k2]), const_cast<const double **>(s.thinned_h[l][k2]), i, mui_ps[thread_id]);
				int rc = pthread_create(&mcmc_threads[thread_id], NULL, mcmcMeanUpdate_, (void *)p);
				if (rc){
					 std::cout << "Error:unable to create thread," << rc << std::endl;
					 exit(-1);
				}
				event_id_offset+=nof_events_thread[thread_id];
			
			}
			//wait for all the partial sums to be computed
			for (unsigned int t = 0; t <nof_threads; t++){
				if(nof_events_thread[t])
					pthread_join (mcmc_threads [t], NULL);
			}
	
			for(unsigned int t = 0; t <nof_threads; t++){
				if(nof_events_thread[t])
					mu_i+=mui_ps[t]; //TODO: uncomment this
			}
		}
		gsl_vector_set(mu_tmp,i,mu_i);
	}
}

void LogisticKernel::mcmcMeanUpdate2(GeneralizedHawkesProcess::State & s, int k2, const gsl_matrix *Sigma, gsl_vector **mu, unsigned int nof_threads){
	*mu=gsl_vector_calloc(nofp); 
	gsl_vector *mu_tmp2=gsl_vector_calloc(nofp); 
	
	for(unsigned int i=0;i<nofp;i++){
		double mu_i=gsl_vector_get(mu_tmp,i);
		mu_i+=(((NormalParameters *)hp[i])->tau)*(((NormalParameters *)hp[i])->mu);	
		gsl_vector_set(mu_tmp2,i, mu_i);
	}
	gsl_blas_dgemv(CblasNoTrans,1.0, Sigma, mu_tmp2,0.0,*mu);
}


void * LogisticKernel::mcmcMeanUpdate_(void *p){

	//unwrap thread parameters
	std::unique_ptr<FormMeanThreadParams> params(static_cast< FormMeanThreadParams * >(p));

	unsigned int event_id_offset=params->thread_id;
	unsigned int nof_events=params->nof_events;
	const double ** thinned_h=params->thinned_h;
	const double ** h=params->h;
	unsigned int i=params->i;
	double &mui=params->mui;
	
	unsigned int Nt=params->thinned_polyagammas.size();//number of thinned events
	unsigned int N=params->polyagammas.size();//number of observed events
	
	mui=0;
	for(unsigned int event_id=event_id_offset;event_id<event_id_offset+nof_events;event_id++){
		if(event_id>=Nt+N){
			std::cerr<<"invalid event id!\n";
		}
		
		if(event_id>=N){//the event is a thinned event
			unsigned int thinned_event_id=event_id-N;
			double hi=(i==0)?1:thinned_h[thinned_event_id][i-1];//i-1 to account for the bias term
			mui+=hi*(-0.5);
		}
		else{//the event is observed
			double hi=(i==0)?1:h[event_id][i-1];
			mui+=hi*(0.5);	
		}	
	}
	return 0;
}




void LogisticKernel::mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads) {

	//todo: check that the kernel is hierarchical from hp->hp

	mcmcPriorUpdate(*ghs, *s, k2, nof_threads);//it needs the hawkes process state because it contains the samples of the excitatory coefficients needed for the prior of the covariance matrix
	
	mcmcWeightsUpdate(*ghs,k2,nof_threads);

}

void LogisticKernel::mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads ) {

}

//thinning kernels for events of type k
void LogisticKernel::mcmcPriorUpdate(GeneralizedHawkesProcess::State & ghs, HawkesProcess::State & hs, int k, unsigned int nof_threads){

	//check whether the kernel is hierarchical (normal-gamma prior is assumed for the weights)
	bool hierarchical_model=false;
	for(unsigned int d2=0;d2!=d;d2++){
		if(hp[d2+1]->hp.empty())
			continue;
		hierarchical_model=true;
		break;
	}
	


	mcmcCovUpdate1(ghs, k, nof_threads);
	mcmcMeanUpdate1(ghs, k, nof_threads);

	if(!hierarchical_model)
		return;


	unsigned int d2;//the dimension whose precision will be updated
	auto collapsedlikelihood_ratio=[&](std::vector<double> new_p)->double{
		//create a kernel with the proposed value new_p as prior precision and mean for the dimension d2
		LogisticKernel *pt_new=clone();
		Kernel *phi=0;
	
		

		
		//set the precision and mean for the weight of the logistic kernel to the proposed value
		NormalParameters * hp_new=(NormalParameters *)pt_new->hp[d2+1];
		if(new_p[1]<0) return EPS;
		hp_new->tau=new_p[1];
		hp_new->sigma=sqrt(1/hp_new->tau);
		hp_new->mu=new_p[2];
		

		auto collapsedlikelihood=[&](LogisticKernel *f, unsigned int d, std::vector<double> & t)->void{
			NormalParameters * hp=(NormalParameters *)f->hp[d+1]; //normal parameters for dimension d, we will computed the collapsed likelihood for these parameters
						
			t[0]=sqrt(hp->tau);
			
			gsl_matrix *Sigma;
			gsl_matrix *Tau;
			double Tau_det;
			f->mcmcCovUpdate2(ghs,k,&Sigma,&Tau, &Tau_det, nof_threads);
		
			//form the posterior mean for the multivariate Gaussian of the weights
			gsl_vector *mu;
			f->mcmcMeanUpdate2(ghs,k,Sigma,&mu,nof_threads);
			gsl_vector *y=gsl_vector_calloc(nofp);
			gsl_blas_dsymv (CblasLower, 1.0, Tau, mu, 0.0,  y);
			//double res;
			double c;
			gsl_blas_ddot(mu, y, &c);
			t[1]=0.5*c-hp->mu*hp->mu*hp->tau*0.5;
			t[2]=sqrt(Tau_det);
			gsl_matrix_free(Sigma);
			gsl_matrix_free(Tau);
			gsl_vector_free(y);
			
			//likelihood of events of type k trigerred by events of type d
			t[3]=1.0;
			for(unsigned int l=0;l!=hs.seq.size();l++){
				t[3]*=HawkesProcess::likelihood(hs.seq[l], *phi,d,k);
			}

		};


		//set the multiplicative coefficient of the excitatory kernel to the proposed value
		phi=hs.phi[d2][k]->clone();
		phi->p[0]=new_p[0];
		
		std::vector<double> t_new(4);
		collapsedlikelihood(pt_new, d2, t_new);
		
		
		std::vector<double> t_old(4);
		phi=hs.phi[d2][k]->clone();
		collapsedlikelihood(this, d2, t_old);
		double ratio;

		ratio=((t_new[0]*t_new[2]*t_new[3])/(t_old[0]*t_old[2]*t_old[3]))*exp(t_new[1]-t_old[1]);
	
		return ratio;
	};
	//todo: metropolis hastings with sparsenormalgamma as proposal. add the hawkeslikelihood ratios in the collapsed ratio (to account for the excitatory coefficient), remove the rest factors and in case of accept, update and the phis

	for(d2=0;d2!=d;d2++){
		std::vector<double> ng_params;
		//bool accept=MetropolisHastings(((NormalGammaParameters *)hp[d2+1]->hp[0]), collapsedlikelihood_ratio, ng_params);//?(ng_params):((NormalParameters *)hp[d2+1])->tau;
		bool accept=MetropolisHastings(((SparseNormalGammaParameters *)ghs.pt_hp[d2][k]), collapsedlikelihood_ratio, ng_params);//?(ng_params):((NormalParameters *)hp[d2+1])->tau;
		if(accept){
			//todo: update the multiplicative coefficient of the kernels
			
			//update the normal parameters
			hs.phi[d2][k]->p[0]=ng_params[0]; //update the multiplicative coefficient of the kernel
			((NormalParameters *)hp[d2+1])->tau=ng_params[1]; //update the precision of the corresponding weight, +1 due to the bis term
			((NormalParameters *)hp[d2+1])->sigma=1/sqrt(((NormalParameters *)hp[d2+1])->tau);  //update the variance of the corresponding weight
			((NormalParameters *)hp[d2+1])->mu=ng_params[2]; //update the mean of the corresponding weight
			
			//update the normal gamma parameters-->it is not needed somewhere
	
		}
	}
}

//TODO: correct adaptive metropolis hastings for rayleigh and power low kernels!
/****************************************************************************************************************************************************************************
 *
 * Power Law Kernel: incomplete implementation
 *
******************************************************************************************************************************************************************************/
void PowerLawKernel::printMatlabExpression(double t0, std::string & matlab_expr) const{

}

PowerLawKernel::PowerLawKernel(){}

//constructor which sets the hyperparameters of the kernel
PowerLawKernel::PowerLawKernel( const std::vector<const DistributionParameters *>  & hp):Kernel(Kernel_PowerLaw, hp){
	if(hp.size()!=3){
		std::cerr<<"wrong number of hyperparameters in the constructor for the constant kernel\n";
	} 
}

//constructor which sets the point parameters of the kernel
PowerLawKernel::PowerLawKernel(const std::vector<double>   & p):Kernel(Kernel_PowerLaw, p){
	if(p.size()!=3)
		std::cerr<<"wrong number of hyperparameters in the constructor for the constant kernel\n";
}

//constructor which sets both the hyperparameters and the parameters of the kernel
PowerLawKernel::PowerLawKernel(const std::vector<const DistributionParameters *>   & hp,  const std::vector<double>  & p):Kernel(Kernel_PowerLaw, hp,p){
	if(hp.size()!=3|| p.size()!=3)
		std::cerr<<"wrong number of hyperparameters or parameters in the constructor for the constant kernel\n";
}


PowerLawKernel::PowerLawKernel(const PowerLawKernel &s):Kernel(s){}

PowerLawKernel *PowerLawKernel::clone() const{
	 return new PowerLawKernel(*this);
}

void PowerLawKernel::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<PowerLawKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void PowerLawKernel::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<PowerLawKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void PowerLawKernel::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<PowerLawKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void PowerLawKernel::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<PowerLawKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void PowerLawKernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void PowerLawKernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void PowerLawKernel::print(std::ostream & file) const{

	if(hp.size()!=3 || p.size()!=3){
		std::cerr<<"constant kernel: wrong number of parameters or hyperparameters\n";
		return ;
	}
	file<<"c*(t+gamma)^{-(1+beta)}\n";
	file<<"c: "<<p[0]<<std::endl;
	file<<"gamma: "<<p[1]<<std::endl;
	file<<"beta: "<<p[2]<<std::endl;
	if(!hp[0]||!hp[1]||!hp[2]){
		std::cerr<<"uninitialized prior for the kernel parameters\n";
		return ;
	}
	
	file<<"prior of c\n";
	hp[0]->print(file);
	file<<"prior of gamma\n";
	hp[1]->print(file);
	file<<"prior of beta\n";
	hp[2]->print(file);

}

double PowerLawKernel::compute(double t, double t0) const{
	return p[0]*(std::pow(t-t0+p[1],-(1+p[2])));
}

double PowerLawKernel::compute(double t) const{
	return  p[0]*(std::pow(t+p[1],-(1+p[2])));
}


double PowerLawKernel::computeMaxValue(double s, double e, double t0) const{
	return p[0]*(std::pow(s-t0+p[1],-(1+p[2])));
}

double PowerLawKernel::computeIntegral(double ts, double te, double t0) const{ 
	if (ts>te)
		throw std::exception();
	return p[0]/(2+p[2])*(std::pow(ts-t0+p[1],-(2+p[2]))-std::pow(te-t0+p[1],-(2+p[2])));
}

//todo: create similar overloaded functions for mcmc states of new models....
void PowerLawKernel::mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads){
	
	if(!ghs){
		switch (hp[0]->type){
			case Distribution_Gamma:{//gamma prior is assumed for the constant kernel
				std::random_device rd;
				std::mt19937 gen(rd());

				unsigned int N_k_k2=0;
				double e=0.0;
				double b_inv=1/(2+p[2]);

				for(unsigned int l=0;l!=s->seq.size();l++){
					N_k_k2+=s->seq[l].countTriggeredEvents(k,k2);//number of events of type k2 trigerred by event of type k

					for(auto e_iter=s->seq[l].type[k].begin();e_iter!=s->seq[l].type[k].end();e_iter++)
						e+=b_inv*(std::pow(p[1],-(2+p[2]))-std::pow(s->end_t-e_iter->second->time+p[1],-(2+p[2])));
				}

				std::gamma_distribution<double> gamma_distribution(((GammaParameters *)hp[0])->alpha+N_k_k2,1/(((GammaParameters *)hp[0])->beta+e));
				p[0]=gamma_distribution(gen);

				break;
			}
			default:
				break;
		}
	}
}

void PowerLawKernel::mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads ) {
	mcmcExcitatoryCoeffUpdate(k,k2, s);//
}

void PowerLawKernel::mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads ) {
}



/****************************************************************************************************************************************************************************
 *
 * Rayleigh Kernel: incomplete implementation
 *
******************************************************************************************************************************************************************************/

RayleighKernel:: RayleighKernel(){}

RayleighKernel:: RayleighKernel(const std::vector<const DistributionParameters *>  & hp2): Kernel( Kernel_Rayleigh, hp2){
	if(hp2.size()!=2)
		std::cerr<<"wrong number of hyperparameters\n";

};

RayleighKernel:: RayleighKernel( const std::vector<const DistributionParameters *>   & hp2,  const std::vector<double>  & p2): Kernel(Kernel_Rayleigh, hp2,p2){
	if(hp2.size()!=2 || p2.size()!=2){
		std::cerr<<"exp kernel: wrong number of parameters or hyperparameters\n";
		return;
	}

};


RayleighKernel:: RayleighKernel( const std::vector<double>  & p2): Kernel( Kernel_Rayleigh, p2){
	if(p2.size()!=2)
		std::cerr<<"wrong number of parameters \n";
};


RayleighKernel:: RayleighKernel(const RayleighKernel &s):Kernel(s){
}

void  RayleighKernel::print(std::ostream & file) const{
	file<<"rayleigh kernel\n";
	file<<"prior for the multiplicative coefficient\n";
	hp[0]->print(file);
	if(p.size()>=1)
		file<<"value for the multiplicative coefficient "<<p[0]<<std::endl;
	file<<"prior for the decaying coefficient\n";
	hp[1]->print(file);
	if(p.size()>=2)
		file<<"value for the decaying coefficient "<<p[1]<<std::endl;
	if(!mh_vars.empty()){
		if(mh_vars[0])
			mh_vars[0]->print(file);

		if(mh_vars[1])
			mh_vars[1]->print(file);
	}

};

void  RayleighKernel::printMatlabExpression(double t0, std::string & matlab_expr) const{
	if(p.size()!=2)
		std::cerr<<"wrong number of kernel parameters\n";
	std::cout<<p[0]<<std::endl;

	std::cout<<p[1]<<std::endl;
	matlab_expr.clear();
	matlab_expr=std::to_string(p[0])+".*(t-"+std::to_string(t0)+")"+".*exp(-"+std::to_string(p[1])+".*(heaviside(t-"+std::to_string(t0)+").*(t-"+std::to_string(t0)+").^2)).*heaviside(t-"+std::to_string(t0)+")";
}

double  RayleighKernel::compute(double t, double t0) const{
	if(p.size()<2){
		std::cerr<<"wrong number of parameters for the kernel "<<p.size();
	}


	double v=p[0]*(t-t0)*exp(-p[1]*(t-t0)*(t-t0));


	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}

double RayleighKernel::compute(double t) const{
	double v=p[0]*t*exp(-p[1]*t*t);
	if(v<-INF)
		return -INF;
	else if(v>INF)
		return INF;
	else
		return v;
}
//TODO: check it again
double RayleighKernel::computeMaxValue(double s, double e, double t0) const{
	double tmax=std::sqrt(1/(2*p[1]))+t0;
	if (s>e)
		throw std::exception();
	double v;
	 if(tmax>e)
		 v=p[0]*(e-t0)*exp(-p[1]*(e-t0)*(e-t0));
	 if(tmax<s)
		 v=p[0]*(s-t0)*exp(-p[1]*(s-t0)*(s-t0));
	 if(tmax>s && tmax<e)
		 v=p[0]*(tmax-t0)*exp(-p[1]*(tmax-t0)*(tmax-t0));
	 return v;
}

double RayleighKernel::computeIntegral(double ts, double te, double t0) const{
	if (ts>te)
		throw std::exception();
	double v=(p[0]/(2*p[1]))*(exp(-p[1]*(ts-t0)*(ts-t0))-exp(-p[1]*(te-t0)*(te-t0)));
	return v>KERNEL_EPS?v:KERNEL_EPS;
}


void RayleighKernel::mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads){
//	unsigned int N_k_k2=0;
//	double e=0.0;
//	double d_inv=p[0]/(2*p[1]);
//
//	for(unsigned int l=0;l!=s->seq.size();l++){
//		N_k_k2+=s->seq[l].countTriggeredEvents(k,k2);//number of events of type k2 trigerred by event of type k
//
//		for(auto e_iter=s->seq[l].type[k].begin();e_iter!=s->seq[l].type[k].end();e_iter++)
//			e+=d_inv*(1-exp(-p[1]*(s->end_t-e_iter->second->time)*(s->end_t-e_iter->second->time)));
//	}
//
//	double alpha=((GammaParameters *)hp[0])->alpha+N_k_k2;
//	double beta=((GammaParameters *)hp[0])->beta+e;
//
//
//
//	if(!ghs){//there is no coupling between the inhibitory and excitatory part
//
//		std::random_device rd;
//		std::mt19937 gen(rd());
//		std::gamma_distribution<double> gamma_distribution(alpha,1/beta);
//		p[0]=gamma_distribution(gen);
//	}
//	else{//hierarchical point process
//
//		if(!ghs->mcmc_iter){
//			mh_vars.resize(p.size());
//			mh_vars[0]=new AdaptiveMetropolisHastingsState(ghs->mcmc_params->c_t_am_sd, ghs->mcmc_params->c_t_am_t0, ghs->mcmc_params->c_t_am_c0, AM_EPSILON);
//			mh_vars[0]->mu_t=alpha/beta;
//		}
//
//		//there is coupling between the excitatory and inhibitory part of the process TODO: finish it, finish it!!! also allocate the mh vars!
//		unsigned int pid=0;
//		auto likelihood_ratio_0=[&](double new_p)->double{
//			//create a kernel with the proposed value new_p as excitatory coefficient
//			Kernel *phi_new=clone();
//			phi_new->p[pid]=new_p;
//
//			//get the sparse normal-gamma parameters
//			double nu_tau=ghs->pt_hp[k][k2]->nu_tau;
//			//std::function<double(double, double, double)> op=ghs->pt_hp[k][k2]->phi->op;
//			double kappa=ghs->pt_hp[k][k2]->lambda;
//			//double kappa=1.0;
//			double alpha_tau=ghs->pt_hp[k][k2]->alpha_tau;
//			double beta_tau=ghs->pt_hp[k][k2]->beta_tau;
//			//get the precision of the logistic kernel
//			double tau=((NormalParameters *)ghs->pt[k2]->hp[k+1])->tau;
//
//
//			double nu_mu=ghs->pt_hp[k][k2]->nu_mu;
//			//std::function<double(double, double, double)> op=ghs->pt_hp[k][k2]->phi->op;
//			//double kappa=ghs->pt_hp[k][k2]->kappa;
//			//double kappa=1.0;
//			double alpha_mu=ghs->pt_hp[k][k2]->alpha_mu;
//
//			//ratio of the likelihood for the new and old sample
//			double l_ratio=1.0;
//			double new_prior_ratio;
//
//			//assume a gamma prior for the multiplicative coefficient
//
//			//part of the prior for the multiplicative cofficient
//			new_prior_ratio=gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, phi_new->p[pid]);
//
//			//part of the prior for the precision
//			//new_prior_ratio*=gamma_pdf(ghs->pt_hp[k][k2]->phi_tau->op(nu, p[pid], alpha_tau),beta, tau);
//			new_prior_ratio*=gamma_pdf(ghs->pt_hp[k][k2]->phi_tau->op(nu_tau, phi_new->p[pid], alpha_tau),beta_tau, tau);
//
//			//part of the prior for the mean
//			new_prior_ratio*=normal_pdf(((NormalParameters *)ghs->pt[k2]->hp[k+1])->mu, -1/(ghs->pt_hp[k][k2]->phi_mu->op(nu_mu, phi_new->p[pid], alpha_mu)), 1/std::sqrt(kappa*tau));
//
//
//			if(new_prior_ratio<=EPS) return EPS;
//			l_ratio*=new_prior_ratio/gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, p[pid]);
//			l_ratio*=1/(gamma_pdf(ghs->pt_hp[k][k2]->phi_tau->op(nu_tau, p[pid], alpha_tau),beta_tau, tau));
//
//			//part of the prior for the mean
//			l_ratio*=1/normal_pdf(((NormalParameters *)ghs->pt[k2]->hp[k+1])->mu, -1/(ghs->pt_hp[k][k2]->phi_mu->op(nu_mu, p[pid], alpha_mu)), 1/std::sqrt(kappa*tau));
//
//
//			//ratio of the hawkes process
//			for(unsigned int l=0;l!=s->seq.size();l++){
//				l_ratio*=(HawkesProcess::likelihood(s->seq[l], *phi_new,k,k2)/HawkesProcess::likelihood(s->seq[l], *this,k,k2));
//			}
//
//			delete phi_new;
//			return l_ratio;
//		};
//
//		//double sample;
//		//p[pid]=MetropolisHastings(new GammaParameters(alpha, beta), likelihood_ratio_0, sample)?sample:p[pid];
//		mh_vars[0]->mcmcUpdate();
//		double sample;
//		double sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
//		double mu=(mh_vars[pid]->samples_n<=mh_vars[pid]->t0)? alpha/beta: mh_vars[pid]->mu_t;
//		p[pid]=MetropolisHastings(new NormalParameters(mu, sigma), likelihood_ratio_0, sample)?sample:p[pid];
//
//	}

}

//mcmc update for the exponential kernel when it is used in the excitatory part of the process
void RayleighKernel::mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads){

	//update the multiplicative coefficient of the exponential kernel
	mcmcExcitatoryCoeffUpdate(k, k2, s, nof_threads);

	//update the decaying coeficient of the exponential kernel
	//if the it is the first iteration, allocate the auxiliary variables for the metropolis hastings
	if(!s->mcmc_iter){
		if(mh_vars.size()!=p.size())
			mh_vars.resize(p.size());
		mh_vars[1]=new AdaptiveMetropolisHastingsState(s->mcmc_params->d_t_am_sd, s->mcmc_params->d_t_am_t0,  s->mcmc_params->d_t_am_c0, AM_EPSILON);
		mh_vars[1]->mu_t=p[1];
	}
	//lambda expression for computing the likelihood ratio, with the new and old decaying coefficient for the excitatory kernels
	unsigned int pid=1;
	auto likelihood_ratio_1=[&](double new_p)->double{
		//create a kernel with the proposed value new_p as decaying coefficient
		Kernel *phi_new=clone();
		phi_new->p[pid]=new_p;
		//ratio of the likelihood for the new and old sample
		double l_ratio=1.0;
		double new_prior_ratio;


		switch(hp[pid]->type){
			case Distribution_Exponential: {//assume an exponential prior for the decaying coefficient
				new_prior_ratio=exp_pdf(((ExponentialParameters *)hp[pid])->lambda,phi_new->p[pid]);
				if(new_prior_ratio<EPS) return EPS;
				l_ratio*=new_prior_ratio/exp_pdf(((ExponentialParameters *)hp[pid])->lambda,p[pid]);
				break;
			}
			case Distribution_Gamma:{ //assume a gamma prior for the decaying coefficient
				new_prior_ratio=gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, phi_new->p[pid]);
				if(new_prior_ratio<EPS) return EPS;
				l_ratio*=new_prior_ratio/gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, p[pid]);
				break;
			}
			default:
				break;
		}

		//ratio of the hawkes process
		for(unsigned int l=0;l!=s->seq.size();l++)
			l_ratio*=(HawkesProcess::likelihood(s->seq[l], *phi_new,k,k2)/HawkesProcess::likelihood(s->seq[l], *this,k,k2));

		delete phi_new;
		return l_ratio;
	};

	mh_vars[pid]->mcmcUpdate();
	double sample;
	double sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	p[pid]=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t, sigma), likelihood_ratio_1, sample)?sample:p[pid];

}

//mcmc update for the exponential kernel when it is used in the inhibition part of the process as history kernel
void RayleighKernel::mcmcInhibitoryUpdate(int l, int k, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads) {

	//if it is the first mcmc step, allocate the metropolis hastings variables for the multiplicative and decaying coefficient
	if(!ghs->mcmc_iter){
		mh_vars.resize(p.size());
		mh_vars[0]=new AdaptiveMetropolisHastingsState(ghs->mcmc_params->c_h_am_sd, ghs->mcmc_params->c_h_am_t0, ghs->mcmc_params->c_h_am_c0, AM_EPSILON);
		mh_vars[0]->mu_t=p[0];
		mh_vars[1]=new AdaptiveMetropolisHastingsState(ghs->mcmc_params->d_h_am_sd, ghs->mcmc_params->d_h_am_t0, ghs->mcmc_params->d_h_am_c0, AM_EPSILON);
		mh_vars[1]->mu_t=p[1];
	}

	unsigned int pid=0;
	auto likelihood_ratio=[&](double new_p)->double{
		//allocate a sigmoid kernel, with a history kernel function for dimension l with value new_p for the parameter pid
		Kernel *psi_new=this->clone();
		psi_new->p[pid]=new_p;
		

		//asign the new history kernel function to each sigmoid kernel of the process
		std::vector<LogisticKernel *> pt_new;
		for(auto iter=ghs->pt.begin();iter!=ghs->pt.end();iter++){
			LogisticKernel *pt_new_k=(*iter)->clone();
			pt_new_k->psi[l]=psi_new;
			pt_new.push_back(pt_new_k);
		}
		double l_ratio=1.0;

		double new_prior_ratio;
		switch(hp[pid]->type){
			case Distribution_Exponential:{ //assume an exponential prior for the decaying coefficient
					new_prior_ratio=exp_pdf(((ExponentialParameters *)hp[pid])->lambda,new_p);
					if(new_prior_ratio<EPS) return EPS;
					l_ratio*=new_prior_ratio/exp_pdf(((ExponentialParameters *)hp[pid])->lambda,p[pid]);
					break;
			}
			case Distribution_Gamma: {//assume a gamma prior for the decaying coefficient
					new_prior_ratio=gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, new_p);
					if(new_prior_ratio<EPS) return EPS;
					l_ratio*=new_prior_ratio/gamma_pdf(((GammaParameters *)hp[pid])->alpha,((GammaParameters *)hp[pid])->beta, p[pid]);
					break;
			}
			default:
				break;
		}
		//
		//	//get the ratio of the likelihoods of the thinning procedure for the events of type k
		for(unsigned int i=0;i!=s->seq.size();i++)
			//l_ratio*=(GeneralizedHawkesProcess::likelihood(s->seq[i],k, pt_new)/GeneralizedHawkesProcess::likelihood(s->seq[i], k, ghs->pt[k]));
			l_ratio*=(GeneralizedHawkesProcess::likelihood(s->seq[i], pt_new)/GeneralizedHawkesProcess::likelihood(s->seq[i], ghs->pt));

		delete psi_new;
		for(auto iter=pt_new.begin();iter!=pt_new.end(); iter++)
			delete (*iter);
		forced_clear(pt_new);
	
		return l_ratio;

	};
	//metropolis hastings update for the multiplicative coefficient
	double sample;
	mh_vars[pid]->mcmcUpdate();
	double sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	p[pid]=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t,sigma), likelihood_ratio, sample)?sample:p[pid];

	pid=1;
	mh_vars[pid]->mcmcUpdate();
	sigma=(mh_vars[pid]->samples_n>=mh_vars[pid]->t0)? sqrt(mh_vars[pid]->sample_cov): sqrt(mh_vars[pid]->c0);
	p[pid]=MetropolisHastings(new NormalParameters(mh_vars[pid]->mu_t, sigma), likelihood_ratio, sample)?sample:p[pid];
}


RayleighKernel *RayleighKernel::clone() const{
	 return new RayleighKernel(*this);
}

void RayleighKernel::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<RayleighKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void RayleighKernel::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<RayleighKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void RayleighKernel::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<RayleighKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void RayleighKernel::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
    boost::serialization::void_cast_register<RayleighKernel,Kernel>();
    ar & boost::serialization::base_object<Kernel>(*this);
}

void RayleighKernel::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void RayleighKernel::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}
