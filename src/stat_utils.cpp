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
#include <boost/math/special_functions/gamma.hpp>
#include <iostream>
#include <math.h>

#include "Kernels.hpp"
#include "stat_utils.hpp"
#include "struct_utils.hpp"
BOOST_SERIALIZATION_ASSUME_ABSTRACT(DistributionParameters)
BOOST_CLASS_EXPORT(NormalParameters)
BOOST_CLASS_EXPORT(ExponentialParameters)
BOOST_CLASS_EXPORT(GammaParameters)
BOOST_CLASS_EXPORT(NormalGammaParameters)
BOOST_CLASS_EXPORT(AdaptiveMetropolisHastingsState)

/****************************************************************************************************************************************************************************
 * Variables for Adaptive Metropolis Hastings
******************************************************************************************************************************************************************************/
AdaptiveMetropolisHastingsState::AdaptiveMetropolisHastingsState(): sd{0},t0{0},c0{0},eps{0}{};

AdaptiveMetropolisHastingsState::AdaptiveMetropolisHastingsState(double s, double t, double c, double e):sd{s}, t0{t}, c0{c}, eps{e}{};

AdaptiveMetropolisHastingsState::AdaptiveMetropolisHastingsState(double s, double t, double c, double e, double m, double sc):sample_mean{m}, sample_cov{sc}, sd{s}, t0{t}, c0{c}, eps{e} {}

AdaptiveMetropolisHastingsState::AdaptiveMetropolisHastingsState(const AdaptiveMetropolisHastingsState &v):
		sample_mean{v.sample_mean},
		sample_cov{v.sample_cov},
		mu_t{v.mu_t},
		samples_n{v.samples_n},
		sd{v.sd},
		t0{v.t0},
		c0{v.c0},
		eps{v.eps}
		{}


//get the parameters for the parameters of the kernel
void AdaptiveMetropolisHastingsState::getVariables(std::vector<double> & vars) const{
	//vars.clear();
	forced_clear(vars);
	vars[0]=sample_mean;
	vars[1]=sample_cov;

}


//set the parameters of the kernel
void AdaptiveMetropolisHastingsState::setVariables(const std::vector<double> & vars){
	sample_mean=vars[0];
	sample_cov=vars[1];
}

void AdaptiveMetropolisHastingsState::save(std::ofstream & file) const{

	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void AdaptiveMetropolisHastingsState::load(std::ifstream & file){

	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void AdaptiveMetropolisHastingsState::print(std::ostream &file){
	file<<"sample mean "<<sample_mean<<std::endl;
	file<<"sample covariance "<<sample_cov<<std::endl;
}


AdaptiveMetropolisHastingsState*  AdaptiveMetropolisHastingsState::clone() const{
	return new  AdaptiveMetropolisHastingsState(*this);//copy constructor implicitly declared
}


void AdaptiveMetropolisHastingsState::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	ar & sample_mean;
	ar & sample_cov;
}


void AdaptiveMetropolisHastingsState::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	ar & sample_mean;
	ar & sample_cov;
}


void AdaptiveMetropolisHastingsState::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	ar & sample_mean;
	ar & sample_cov;
}


void AdaptiveMetropolisHastingsState::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	ar & sample_mean;
	ar & sample_cov;
}

void AdaptiveMetropolisHastingsState::mcmcUpdate(){
	//update recursively the sample mean of the parameter
	if(samples_n){
		double sample_mean_old=sample_mean;
		//update recursively the sample mean
		sample_mean=((samples_n*sample_mean)+mu_t)/(samples_n+1);
		sample_cov=((samples_n-1)/(double)samples_n)*sample_cov+(sd/samples_n)*(samples_n*sample_mean_old*sample_mean_old-(samples_n+1)*sample_mean*sample_mean+mu_t*mu_t+eps);
	}	
	
	//increase the number of samples
	samples_n++;
}

bool MetropolisHastings(DistributionParameters * proposal, std::function<double(double)> likelihoodRatio, double &p){
	std::random_device rd;
	std::mt19937 gen(rd());

	switch(proposal->type){
		case Distribution_Gamma:{
			// todo: check if the kernel is hierarchical
			//sample from a gamma distribution
			std::gamma_distribution<double> gamma_distribution(((GammaParameters *)(proposal))->alpha, 1/((GammaParameters *)(proposal))->beta);
			p=gamma_distribution(gen);
			break;
		}
		case Distribution_Exponential:{
			// todo: check if the kernel is hierarchical
			//sample from a exponential distribution
			std::exponential_distribution<double> exp_distribution(((ExponentialParameters *)(proposal))->lambda);
			p=exp_distribution(gen);
			break;
		}
		case Distribution_Normal:{
			//sample from a normal distribution
			std::normal_distribution<double> norm_distribution(((NormalParameters *)proposal)->mu,((NormalParameters *)proposal)->sigma);
			p=norm_distribution(gen);
			break;
		}
		default:
			break;
	}
	
	double ratio=likelihoodRatio(p);
	//compute acceptance probability
	double accept_prob=std::min(1.0,ratio);
	//accept/reject the new value
	std::bernoulli_distribution bern_distribution(accept_prob);
	return (bern_distribution(gen));

}


bool MetropolisHastings(DistributionParameters * proposal, std::function<double(std::vector<double>)> likelihoodRatio, std::vector<double> &p){
	std::random_device rd;
	std::mt19937 gen(rd());

	p.clear();// p contains the proposed values, and in case of acceptance the new samples after the joint update of the random variables
	switch(proposal->type){
		case Distribution_NormalGamma:{
		
			std::gamma_distribution<double> gamma_distribution(((NormalGammaParameters *)proposal)->alpha, 1/((NormalGammaParameters *)proposal)->beta);
			double tau=gamma_distribution(gen);

			p.push_back(tau);
			std::normal_distribution<double> normal_distribution(((NormalGammaParameters *)proposal)->mu, std::sqrt(1/((NormalGammaParameters *)proposal)->kappa*tau));
			
			double mu=normal_distribution(gen);
			p.push_back(mu);
			break;
		}
		case Distribution_SparseNormalGamma:{

			if(!proposal->hp.empty()){ //sample alpha from its prior, in case it is defined; otherwise its current value is used for
				switch(proposal->hp[0]->type){
					case Distribution_Gamma:{
						// todo: check if the kernel is hierarchical
						//sample from a gamma distribution
						std::gamma_distribution<double> gamma_distribution(((GammaParameters *)proposal->hp[0])->alpha, 1/((GammaParameters *)proposal->hp[0])->beta);
						((SparseNormalGammaParameters *)proposal)->alpha=gamma_distribution(gen);
						break;
					}
					case Distribution_Exponential:{
						// todo: check if the kernel is hierarchical
						//sample from a exponential distribution
						std::exponential_distribution<double> exp_distribution(((ExponentialParameters *)proposal->hp[0])->lambda);
						((SparseNormalGammaParameters *)proposal)->alpha=exp_distribution(gen);
						break;
					}
					default:
						break;
				}
			}
			p.push_back(((SparseNormalGammaParameters *)proposal)->alpha);
			
			
			NormalGammaParameters  *ngp=new NormalGammaParameters{0,0,0,0};
			ngp->alpha=((SparseNormalGammaParameters *)proposal)->phi_tau->op(((SparseNormalGammaParameters *)proposal)->nu_tau, ((SparseNormalGammaParameters *)proposal)->alpha, ((SparseNormalGammaParameters *)proposal)->alpha_tau);
			ngp->beta=((SparseNormalGammaParameters *)proposal)->beta_tau;
			ngp->mu=-1/(((SparseNormalGammaParameters *)proposal)->phi_mu->op(((SparseNormalGammaParameters *)proposal)->nu_mu, ((SparseNormalGammaParameters *)proposal)->alpha, ((SparseNormalGammaParameters *)proposal)->alpha_mu));
			ngp->kappa=((SparseNormalGammaParameters *)proposal)->lambda;

			//sample the precision
			std::gamma_distribution<double> gamma_distribution(ngp->alpha, 1/ngp->beta);
			double tau=gamma_distribution(gen);
			p.push_back(tau);

			//sample the mean
			std::normal_distribution<double> normal_distribution(ngp->mu, std::sqrt(1/(ngp->kappa*tau)));
			double mu=normal_distribution(gen);
			p.push_back(mu);
		}
		default:
			break;
		// todo: add case for sparse normal gamma, add cases for the prior of the alpha parameter
		//todo: add cases for multivariate normal, and independent multivariate normal
	}
	
	double ratio=likelihoodRatio(p);
	//compute acceptance probability
	double accept_prob=std::min(1.0,ratio);
	//accept/reject the new value
	std::bernoulli_distribution bern_distribution(accept_prob);
	return (bern_distribution(gen));
}

/****************************************************************************************************************************************************************************
 * Definition of distribution families
******************************************************************************************************************************************************************************/

//Distribution  Family parameters


//constructor for flat model
DistributionParameters::DistributionParameters(DistributionFamily t, unsigned int n): type{t},nofp{n} {};


//constructor for hierarchical model
DistributionParameters::DistributionParameters(DistributionFamily t, unsigned int n, const std::vector<DistributionParameters *> & hp2):type{t},nofp{n}{
	hp.resize(hp2.size());
	for(unsigned int i=0;i!=hp2.size();i++)
		hp[i]=hp2[i]->clone();
}

void DistributionParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	ar & type;
	ar & nofp;
	ar & hp;
}

void DistributionParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	ar & type;
	ar & nofp;
	ar & hp;
}

void DistributionParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	ar & type;
	ar & nofp;
	ar & hp;
}

void DistributionParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	ar & type;
	ar & nofp;
	ar & hp;
}

DistributionParameters::~DistributionParameters(){
	for(auto iter=hp.begin();iter!=hp.end();iter++)
		delete *iter;
	forced_clear(hp);
}

//Univariate Normal distribution parameters

NormalParameters::NormalParameters(): DistributionParameters{Distribution_Normal, 2}, mu(0), sigma(0) {};

NormalParameters::NormalParameters(double mu2, double sigma2): DistributionParameters{Distribution_Normal, 2}, mu(mu2), sigma(sigma2), tau (1/(sigma2*sigma2)){};

NormalParameters::NormalParameters(double mu2, double sigma2, std::vector<DistributionParameters *> & hp): DistributionParameters{Distribution_Normal, 2, hp},  mu(mu2), sigma(sigma2), tau (1/(sigma2*sigma2)){};

NormalParameters::NormalParameters(const NormalParameters &h): DistributionParameters{Distribution_Normal, h.nofp, h.hp},mu{h.mu}, sigma{h.sigma}, tau{h.tau}{
};

void NormalParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalParameters,DistributionParameters>();
	ar & mu;
	ar & sigma;
	ar & tau;
}


void NormalParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalParameters,DistributionParameters>();
	ar & mu;
	ar & sigma;
	ar & tau;
}

void NormalParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalParameters,DistributionParameters>();
	ar & mu;
	ar & sigma;
	ar & tau;
}


void NormalParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalParameters,DistributionParameters>();
	ar & mu;
	ar & sigma;
	ar & tau;
}


void NormalParameters::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void NormalParameters::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void NormalParameters::print(std::ostream &file){
	file<<"mean :" <<mu<<std::endl;
	file<<"standard deviation "<<sigma<<std::endl;
	file<<"precision "<<1/(sigma*sigma)<<std::endl;
	
	file<<"hyperparameters for normal prior\n";
	for(auto iter=hp.begin();iter!=hp.end();iter++){
		if(*iter)
			(*iter)->print(file);
	}
}

NormalParameters* NormalParameters::clone() const{
	return new NormalParameters(*this);
}


void NormalParameters::getParams(std::vector<double> & params) const {
	
	//params.clear();
	forced_clear(params);
	
	params.push_back(mu);
	params.push_back(sigma);
}

//Exponential distribution parameters

ExponentialParameters::ExponentialParameters(): DistributionParameters{Distribution_Exponential, 1}, lambda(0) {};

ExponentialParameters::ExponentialParameters(double l0): DistributionParameters{Distribution_Exponential, 1}, lambda(l0){};//flat

ExponentialParameters::ExponentialParameters(double l0, std::vector<DistributionParameters *> & hp): DistributionParameters{Distribution_Exponential, 1, hp}, lambda(l0){};//hierarchical;

ExponentialParameters::ExponentialParameters(const ExponentialParameters &h): DistributionParameters{Distribution_Exponential, 1, h.hp},lambda{h.lambda}{};


void ExponentialParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<ExponentialParameters,DistributionParameters>();
	ar & lambda;
}

void ExponentialParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<ExponentialParameters,DistributionParameters>();
	ar & lambda;
}

void ExponentialParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<ExponentialParameters,DistributionParameters>();
	ar & lambda;
}

void ExponentialParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<ExponentialParameters,DistributionParameters>();
	ar & lambda;
}


void ExponentialParameters::load(std::ifstream & file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void ExponentialParameters::save(std::ofstream & file) const{
	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}



void ExponentialParameters::print(std::ostream &file){
	file<<"exponential distribution\n";
	file<<"lambda "<<lambda<<std::endl;
}

ExponentialParameters* ExponentialParameters::clone() const{
	return new ExponentialParameters(*this);
}

void ExponentialParameters::getParams(std::vector<double> & params) const{
	forced_clear(params);
	params.push_back(lambda);
}

//Gamma distribution parameters

GammaParameters::GammaParameters(): DistributionParameters{Distribution_Gamma,2},alpha(0), beta(0) {};

GammaParameters::GammaParameters(double a0, double b0): DistributionParameters{Distribution_Gamma, 2}, alpha(a0), beta(b0){};//flat model

GammaParameters::GammaParameters(double a0, double b0, std::vector<DistributionParameters *> & hp): DistributionParameters{Distribution_Gamma, 2, hp}, alpha(a0), beta(b0){};//hierarchical model

GammaParameters::GammaParameters(const GammaParameters &h): DistributionParameters{Distribution_Gamma, 2, h.hp},alpha{h.alpha},beta{h.beta}{};

void GammaParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<GammaParameters,DistributionParameters>();
	ar & alpha;
	ar & beta;
}

void GammaParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<GammaParameters,DistributionParameters>();
	ar & alpha;
	ar & beta;
}

void GammaParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<GammaParameters,DistributionParameters>();
	ar & alpha;
	ar & beta;
}

void GammaParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<GammaParameters,DistributionParameters>();
	ar & alpha;
	ar & beta;
}


void GammaParameters::load(std::ifstream & file){

	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void GammaParameters::save(std::ofstream & file) const{

	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}


void GammaParameters::print(std::ostream &file){
	file<<"gamma distribution\n";
	file<<"alpha "<<alpha<<std::endl;
	file<<"beta "<<beta<<std::endl;
}

void GammaParameters::getParams(std::vector<double> & params) const {
	forced_clear(params);
	params.push_back(alpha);
	params.push_back(beta);
}

GammaParameters* GammaParameters::clone() const{

	return new GammaParameters(*this);
}

//Normal Gamma distribution parameters

NormalGammaParameters::NormalGammaParameters():DistributionParameters{Distribution_NormalGamma, 4}, mu(0),kappa(0),alpha(0),beta(0) {}

NormalGammaParameters::NormalGammaParameters(double m0, double k0, double a0, double b0): DistributionParameters{Distribution_NormalGamma, 4}, mu(m0), kappa(k0), alpha(a0), beta(b0){};//flat model

NormalGammaParameters::NormalGammaParameters(double m0, double k0, double a0, double b0, std::vector<DistributionParameters *> & hp): DistributionParameters{Distribution_NormalGamma, 4, hp}, mu(m0), kappa(k0), alpha(a0), beta(b0){};//hierarchical

NormalGammaParameters::NormalGammaParameters(const NormalGammaParameters &h): DistributionParameters{Distribution_NormalGamma, 4, h.hp}, mu{h.mu},kappa{h.kappa},alpha{h.alpha},beta{h.beta}{};


void NormalGammaParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalGammaParameters,DistributionParameters>();
	ar & mu;
	ar & kappa;
	ar & alpha;
	ar & beta;
}

void NormalGammaParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalGammaParameters,DistributionParameters>();
	ar & mu;
	ar & kappa;
	ar & alpha;
	ar & beta;
}

void NormalGammaParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalGammaParameters,DistributionParameters>();
	ar & mu;
	ar & kappa;
	ar & alpha;
	ar & beta;
}

void NormalGammaParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<NormalGammaParameters,DistributionParameters>();
	ar & mu;
	ar & kappa;
	ar & alpha;
	ar & beta;
}


void NormalGammaParameters::load(std::ifstream & file){

	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void NormalGammaParameters::save(std::ofstream & file) const{

	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}

void NormalGammaParameters::print(std::ostream &file){
	file<<"normal gamma distribution\n";
	file<<"mu "<<mu<<std::endl;
	file<<"kappa "<<kappa<<std::endl;
	file<<"alpha "<<alpha<<std::endl;
	file<<"beta "<<beta<<std::endl;
}

NormalGammaParameters* NormalGammaParameters::clone() const{
	return new NormalGammaParameters(*this);
}

void NormalGammaParameters::getParams(std::vector<double> & params) const{
	forced_clear(params);
	params.push_back(mu);
	params.push_back(kappa);
	params.push_back(alpha);
	params.push_back(beta);
}


//Sparse Normal Gamma distribution parameters

SparseNormalGammaParameters::SparseNormalGammaParameters():DistributionParameters{Distribution_SparseNormalGamma, 8}, lambda(0), alpha(0), alpha_tau(0), beta_tau(0), nu_tau(0), alpha_mu(0), nu_mu(0) {};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double at, double bt, double nt, double am, double nm): DistributionParameters{Distribution_SparseNormalGamma, 8}, lambda(l0), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm){};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm): DistributionParameters{Distribution_SparseNormalGamma, 8}, lambda(l0), alpha(a), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm){};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, std::vector<DistributionParameters *> & hp): DistributionParameters{Distribution_SparseNormalGamma, 8, hp}, lambda(l0), alpha(a), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm){};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double at, double bt, double nt, double am, double nm, ActivationFunction *pt, ActivationFunction *pm): DistributionParameters{Distribution_SparseNormalGamma, 8}, lambda(l0), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm), phi_tau(pt), phi_mu{pm}{};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, ActivationFunction *pt, ActivationFunction *pm): DistributionParameters{Distribution_SparseNormalGamma, 8}, lambda(l0), alpha(a), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm), phi_tau(pt), phi_mu{pm}{};

SparseNormalGammaParameters::SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, std::vector<DistributionParameters *> & hp, ActivationFunction *pt, ActivationFunction *pm): DistributionParameters{Distribution_SparseNormalGamma, 8, hp}, lambda(l0), alpha(a), alpha_tau(at), beta_tau(bt), nu_tau(nt), alpha_mu(am), nu_mu(nm), phi_tau(pt), phi_mu{pm}{};

SparseNormalGammaParameters::SparseNormalGammaParameters(const SparseNormalGammaParameters &h): DistributionParameters{Distribution_SparseNormalGamma, 8, h.hp}, lambda(h.lambda), alpha(h.alpha), alpha_tau(h.alpha_tau), beta_tau(h.beta_tau), nu_tau(h.nu_tau), alpha_mu(h.alpha_mu), nu_mu(h.nu_mu), phi_tau(h.phi_tau), phi_mu{h.phi_mu}{};


void SparseNormalGammaParameters::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<SparseNormalGammaParameters,DistributionParameters>();	
	ar & lambda;
	ar & alpha;
	ar & alpha_tau;
	ar & beta_tau;
	ar & nu_tau;
	ar & alpha_mu;
	ar & nu_mu;
	ar & phi_tau;
	ar & phi_mu;
}

void SparseNormalGammaParameters::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<SparseNormalGammaParameters,DistributionParameters>();
	ar & lambda;
	ar & alpha;
	ar & alpha_tau;
	ar & beta_tau;
	ar & nu_tau;
	ar & alpha_mu;
	ar & nu_mu;
	ar & phi_tau;
	ar & phi_mu;
}

void SparseNormalGammaParameters::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<SparseNormalGammaParameters,DistributionParameters>();
	ar & lambda;
	ar & alpha;
	ar & alpha_tau;
	ar & beta_tau;
	ar & nu_tau;
	ar & alpha_mu;
	ar & nu_mu;
	ar & phi_tau;
	ar & phi_mu;
}

void SparseNormalGammaParameters::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
	boost::serialization::void_cast_register<SparseNormalGammaParameters,DistributionParameters>();
	ar & lambda;
	ar & alpha;
	ar & alpha_tau;
	ar & beta_tau;
	ar & nu_tau;
	ar & alpha_mu;
	ar & nu_mu;
	ar & phi_tau;
	ar & phi_mu;
}


void SparseNormalGammaParameters::load(std::ifstream & file){

	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}

void SparseNormalGammaParameters::save(std::ofstream & file) const{

	boost::archive::binary_oarchive oa{file};
	oa<<*this;
}



void SparseNormalGammaParameters::print(std::ostream &file){
	file<<"sparse normal gamma distribution\n";
	file<<"lambda "<<lambda<<std::endl;
	file<<"alpha "<<alpha<<std::endl;
	file<<"alpha_tau "<<alpha_tau<<std::endl;
	file<<"beta_tau "<<beta_tau<<std::endl;
	file<<"nu_tau "<<nu_tau<<std::endl;
	file<<"alpha_mu "<<alpha_mu<<std::endl;
	file<<"nu_mu "<<nu_mu<<std::endl;
	file<<"activation function for the mean\n";
	phi_mu->print(file);
	file<<"activation function for the precision\n";
	phi_tau->print(file);
}

SparseNormalGammaParameters* SparseNormalGammaParameters::clone() const{
	return new SparseNormalGammaParameters(*this);
}

void SparseNormalGammaParameters::getParams(std::vector<double> & params) const {
	forced_clear(params);
}



/****************************************************************************************************************************************************************************
 * Simulation of Poisson Processes
******************************************************************************************************************************************************************************/

//Generates event arrivals coming from a homogeneous Poisson Process in the interval [s,e].

std::vector<double> simulateHPoisson(double lambda, double s, double e,  int N){
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::vector <double> arrival_times;
	std::uniform_real_distribution<> uniform_distribution(0,1.0);
	double t=s;
	unsigned int nof_events=0;
	double dt;
	while(true){
		double u=uniform_distribution(gen);
		dt=-log(1-u)/lambda;
		t+=dt;
		if(t>e)
			break;
		arrival_times.push_back(t);
		nof_events++;
		if(N>0 && nof_events==(unsigned int)N)
			break;
	}
	return arrival_times;
}
//Generates event arrivals coming from a non-homogeneous Poisson Process with bounded intensity in the interval [ts, te].
//see: (Lewis and Shedler, Simulation of nonhomogenous Poisson processes by thinning)
std::vector<double> simulateNHPoisson(const Kernel & phi, double t0, double ts, double te, int N){
	std::vector <double> arrival_times;
	double t=ts;
	double lambda=phi.computeMaxValue(ts, te, t0);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> uniform_distribution(0,1.0);
	unsigned int nof_events=0;
	while(true){
		//compute next arrival time
		t=t-log(uniform_distribution(gen))/lambda;
		if(t>te) break;
		if(uniform_distribution(gen)<=phi.compute(t,t0)/lambda){
			arrival_times.push_back(t);
			nof_events++;
		}
		if(N>0 && nof_events==(unsigned int)N)
			break;
	}
	return arrival_times;
}

/****************************************************************************************************************************************************************************
 * Computation of pdf (probability density functions)
******************************************************************************************************************************************************************************/

// it returns the pdf value of the gamma distribution (assuming an inverse-scale parametrization at x).
double gamma_pdf(double alpha, double beta, double x) {
	if (alpha<0){
		return 0.0;
	}
	if (beta<0){
		std::cerr<<"Invalid beta parameter.\n";
	}
	if (x<0){
		return 0.0;
	}
    
    return boost::math::gamma_p_derivative(alpha, x *beta) *beta;
}

//it returns the pdf value of the exponential distribution (assuming an inverse-scale parametrization at x).
double exp_pdf(double lambda, double x){
	if(lambda<0){
		std::cerr<<"Invalid alpha parameter.\n";
		throw std::exception();
	}
	if (x<0){
		return 0.0;
	}
	return (lambda*exp(-lambda*x));
}

double normal_pdf(float x, float mu, float sigma){
	if(sigma<0){
		std::cerr<<"Invalid sigma parameter.\n";
		throw std::exception();
	}
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5f * a * a);
}

/****************************************************************************************************************************************************************************
 * Computation of sample statistics
******************************************************************************************************************************************************************************/


void meanplot(const std::vector<double> & theta, std::vector<double> & mean_v){
	accumulator_set<double, stats<tag::mean>> theta_a;

	for(auto iter=theta.begin();iter!=theta.end();iter++){
		theta_a(*iter);
		mean_v.push_back(mean(theta_a));
	}
}

void meanplot(const boost::numeric::ublas::vector<double> & theta, std::vector<double> & mean_v){
	accumulator_set<double, stats<tag::mean>> theta_a;

	for(auto iter=theta.begin();iter!=theta.end();iter++){
		theta_a(*iter);
		mean_v.push_back(mean(theta_a));
	}
}

//https://stackoverflow.com/questions/7616511/calculate-mean-and-standard-deviation-from-a-vector-of-samples-in-c-using-boos
void acf(const std::vector<double> & theta, std::vector<double> & acf_v){
	accumulator_set<double, stats<tag::mean,tag::variance>> theta_a;
	for(auto iter=theta.begin();iter!=theta.end();iter++)
		theta_a(*iter);
	
	double mean_theta=mean(theta_a);
	double var_theta=variance(theta_a);
	
	long unsigned int N=theta.size();
	if(acf_v.size()<N)
		acf_v.resize(N);
	
	for(long unsigned int lag=0;lag<N;lag++){
		double acf_sum=0;
		for(long unsigned int i=0;i<N-lag;i++){
			acf_sum+=(theta[i]-mean_theta)*(theta[i+lag]-mean_theta);
		}
		if(abs(var_theta)>EPS)
			acf_v[lag]=(acf_sum/N)/var_theta;
		else
			acf_v[lag]=1;
	}  
}

void acf(const boost::numeric::ublas::vector<double> & theta, std::vector<double> & acf_v){
	accumulator_set<double, stats<tag::mean,tag::variance>> theta_a;
	for(auto iter=theta.begin();iter!=theta.end();iter++)
		theta_a(*iter);

	double mean_theta=mean(theta_a);
	double var_theta=variance(theta_a);

	unsigned int N=theta.size();
	if(acf_v.size()<N)
		acf_v.resize(N);

	for(unsigned int lag=0;lag<N;lag++){
		boost::numeric::ublas::scalar_vector<double> mean_theta_v(N-lag,mean_theta);
		//TODO:this is not memory efficient
		boost::numeric::ublas::vector<double> theta_lag(N-lag);
		std::move(theta.begin(),theta.begin()+N-lag,theta_lag.begin());
		theta_lag=theta_lag-mean_theta_v;
		boost::numeric::ublas::vector<double> theta_shifted_lag(N-lag);
		std::move(theta.begin()+lag,theta.end(),theta_shifted_lag.begin());
		theta_shifted_lag=theta_shifted_lag-mean_theta_v;
		double acf_sum=inner_prod(theta_lag,theta_shifted_lag);
		if(abs(var_theta)>EPS)
			acf_v[lag]=(acf_sum/N)/var_theta;
		else
			acf_v[lag]=1;
	}
}

double sample_mode(const boost::numeric::ublas::vector<double> &samples){
	boost::numeric::ublas::vector<double> x;
	boost::numeric::ublas::vector<double> p_x;
	double h=gkr_h(samples);
	if(h==EPS){//the kernel bandwidth is close to zero==samples have low variance, we close to point mass distribution
		accumulator_set<double, stats<tag::mean,tag::variance,tag::min,tag::max>> acc;
		for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));
		return mean(acc);
	}

	gkr(samples,x,p_x,NOF_POINTS);
	//find the posterior mode
    auto mode_p_x_iter=boost::max_element(p_x);


    double m=x(std::distance(p_x.begin(), mode_p_x_iter));
    return m;
}

double sample_mean(const boost::numeric::ublas::vector<double> &samples){
	accumulator_set<double, stats<tag::mean>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));
	return mean(acc);
}

/****************************************************************************************************************************************************************************
 * Density estimation functions
******************************************************************************************************************************************************************************/

double gkr_h(const boost::numeric::ublas::vector<double> &samples){
	unsigned int nofss=samples.size();
	typedef accumulator_set<double, stats<tag::mean,tag::variance,tag::min,tag::max,boost::accumulators::tag::tail_quantile< boost::accumulators::right> > > accumulator_t_right;
	accumulator_t_right acc(tag::tail< boost::accumulators::right>::cache_size = nofss);
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));

	//find the optimal kernel bandwidth by Silverman's rule of thumb

	double sigma_samples=sqrt(variance(acc));
	double h=1.06*sigma_samples*pow(nofss,-0.2);
	if(h<EPS) //for the case the samples have zero or very small variance
		h=EPS;
	return h;
}


void gkr(const boost::numeric::ublas::vector<double> &samples, boost::numeric::ublas::vector<double> & x_v, boost::numeric::ublas::vector<double> & p_x, unsigned int nofxs, double h){
	unsigned int nofss=samples.size();

	//https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Boost/Accumulator/Accumulator.cpp
	accumulator_set<double, stats<tag::mean,tag::variance,tag::min,tag::max>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));
	
	if(h==-1){//kernel bandwidth is not provided by caller
	 h=gkr_h(samples);
	}

	//find the range for which the posterior will be plotted;[min(min_of_samples,mean-2*std), max(max_of_samples,mean+2*std)]
	double sigma_samples=sqrt(variance(acc));
	double max_sample=extract_result<tag::max>(acc);
	double min_sample=extract_result<tag::min>(acc);

	double start_x=min_sample-sigma_samples;
	double end_x=max_sample+sigma_samples;
	double step=(end_x-start_x)/(nofxs-1);

	boost::numeric::ublas::scalar_vector<double> start_x_v(nofxs,start_x);
	boost::numeric::ublas::vector<double> step_x_v(nofxs);
	std::iota(step_x_v.begin(),step_x_v.end(), 0);
	if(x_v.size()<nofxs)
		x_v.resize(nofxs);
	x_v=start_x_v+step_x_v*step;

	//compute the kernel values for each pair of test point and sample
	boost::numeric::ublas::scalar_vector<double> ones_nofss(nofss,1.0);
	//it repeats the test points in the columns
	boost::numeric::ublas::matrix<double> X=outer_prod(x_v,ones_nofss);
	//it repeats the posterior samples in the rows
	boost::numeric::ublas::scalar_vector<double> ones_nofxs(nofxs,1.0);
	boost::numeric::ublas::matrix<double> S=outer_prod(ones_nofxs,samples);
	//apply kernel function. each row of K corresponds to a test point
	boost::numeric::ublas::matrix<double> K=(X-S)*1/h;

	std::transform(K.data().begin(),K.data().end(),K.data().begin(),[&](double x){
		return normal_pdf(x,0,1.0);
	});
	
	//add the columns of K to sum the kernel functions
	if(p_x.size()!=nofxs)
		p_x.resize(nofxs);

	p_x=prod(K, ones_nofss);
	p_x/=(h*nofss);
}

void gkr(const boost::numeric::ublas::vector<double> &samples, const boost::numeric::ublas::vector<double> &x_v, boost::numeric::ublas::vector<double> & p_x, double h){
	unsigned int nofss=samples.size();

	//https://github.com/daviddoria/Examples/blob/master/c%2B%2B/Boost/Accumulator/Accumulator.cpp
	accumulator_set<double, stats<tag::mean,tag::variance,tag::min,tag::max>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));

	if(h==-1){//kernel bandwidth is not provided by caller
	 h=gkr_h(samples);
	 
	}
	
	//find the range for which the postrior will be plotted;[min(min_of_samples,mean-2*std), max(max_of_samples,mean+2*std)]
	unsigned int nofxs=x_v.size();

	//compute the kernel values for each pair of test point and sample
	boost::numeric::ublas::scalar_vector<double> ones_nofss(nofss,1.0);
	//it repeats the test points in the columns
	boost::numeric::ublas::matrix<double> X=outer_prod(x_v,ones_nofss);
	//it repeats the posterior samples in the rows
	boost::numeric::ublas::scalar_vector<double> ones_nofxs(nofxs,1.0);
	boost::numeric::ublas::matrix<double> S=outer_prod(ones_nofxs,samples);
	//apply kernel function. each row of K corresponds to a test point
	boost::numeric::ublas::matrix<double> K=(X-S)*1/h;

	std::transform(K.data().begin(),K.data().end(),K.data().begin(),[&](double x){
		return normal_pdf(x,0,1.0);
	}
	);

	//add the columns of K to sum the kernel functions
	if(p_x.size()!=nofxs)
		p_x.resize(nofxs);
	p_x=prod(K, ones_nofss);
	p_x/=(h*nofss);

}




/****************************************************************************************************************************************************************************
 * sampling from some distributions functions
******************************************************************************************************************************************************************************/
//it draws samples from a multivariate gaussian
void multivariate_normal(const gsl_vector *mu, const gsl_matrix *L, std::vector<double> & sample){

	  const size_t M = L->size1;
	  const size_t N = L->size2;
	  gsl_vector * result;

	  if (M != N)
	    {
	      std::cerr<<"sampling from multivariate normal requires square covariance matrix\n";
	    }
	  else if (mu->size != M)
	    {
	      std::cerr<<"incompatible dimension of mean vector with variance-covariance matrix\n";
	    }
	  else
	    {
		  //allocate the gsl vector that will hold the sample
		  result=gsl_vector_calloc(M);

		  //initialize the random number generator
			gsl_rng *r=gsl_rng_alloc (gsl_rng_taus2);
			long seed = time (NULL) * getpid();
			gsl_rng_set (r, seed);



	      size_t i;

	      for (i = 0; i < M; ++i){
	          gsl_vector_set(result, i, gsl_ran_ugaussian(r));
	      }

	      gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, L, result);
	      gsl_vector_add(result, mu);


	    }

	  //move the sample to a std::vector TODO: better way to do that
	  sample.resize(M);
	  for(unsigned int i=0;i<M;i++)
		  sample[i]=gsl_vector_get(result,i);

}


void multivariate_normal(const gsl_vector *mu, const gsl_matrix *L, gsl_vector ** result){

	  const size_t M = L->size1;
	  const size_t N = L->size2;

	  if (M != N)
	    {
	      std::cerr<<"sampling from multivariate normal requires square covariance matrix\n";
	    }
	  else if (mu->size != M)
	    {
	      std::cerr<<"incompatible dimension of mean vector with variance-covariance matrix\n";
	    }
	  else
	    {
		  //allocate the gsl vector that will hold the sample
		  *result=gsl_vector_calloc(M);

		  //initialize the random number generator
			gsl_rng *r=gsl_rng_alloc (gsl_rng_taus2);
			long seed = time (NULL) * getpid();
			gsl_rng_set (r, seed);


	      size_t i;

	      for (i = 0; i < M; ++i)
	        gsl_vector_set(*result, i, gsl_ran_ugaussian(r));

	      gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, L, *result);
	      gsl_vector_add(*result, mu);


	    }
}



void normal_gamma(const NormalGammaParameters &h, double &mu, double & lambda){

	//initialize the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	//draw lambda

	std::gamma_distribution<double> gamma_distribution(h.alpha, 1/h.beta);

	lambda=gamma_distribution(gen);


	//draw mu


	std::normal_distribution<double> normal_distribution(h.mu,1/(sqrt(h.kappa*lambda)));
	mu=normal_distribution(gen);
}




//it returns a sample for the mean and precision of the normal model, given its observations w and the hyperparameters h
void sample_ng_normal_model(const NormalGammaParameters &h, const std::vector<double> & w, double &mu, double & lambda){

	//compute sample variance and mean of the data w
	accumulator_set<double, stats<tag::mean,tag::variance>> w_a;
	for(auto iter=w.begin();iter!=w.end();iter++)
		w_a(*iter);

	double mean_w=mean(w_a);

	double var_w=variance(w_a);

	unsigned int n=w.size();

	//compute posterior parameters
	double mu_n=(h.kappa*h.mu+n*mean_w)/(h.kappa+n);
	double kappa_n=h.kappa+n;
	double alpha_n=h.alpha+n/2;
	double beta_n=h.beta+n*var_w/2+(h.kappa*n*(mean_w-h.mu)*(mean_w-h.mu))/(2*(h.kappa+n));

	NormalGammaParameters hp{mu_n,kappa_n,alpha_n,beta_n};//posterior hyperparameters

	//sample new mean and precision for the univariate normal model
	normal_gamma(hp,mu,lambda);


}

