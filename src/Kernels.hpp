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

#ifndef KERNELS_HPP
#define KERNELS_HPP
#include <boost/serialization/export.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <math.h>
#include <fstream>
#include <random>
#include "PointProcess.hpp"
#include "HawkesProcess.hpp"
#include "GeneralizedHawkesProcess.hpp"
#include "stat_utils.hpp"

class PointProcess;
class HawkesProcess;
class GeneralizedHawkesProcess;

enum KernelType {Kernel_Constant, Kernel_Exponential, Kernel_Logistic, Kernel_PowerLaw, Kernel_Rayleigh};
class Kernel{

	public:
		KernelType type;
		unsigned int nofp; //number of parameters for the kernel
		unsigned int nofhp; //number of hyperparameters for the kernel

		std::vector<DistributionParameters *> hp; //hyperparameters for the point parameters of the kernel
		std::vector<double > p; //point parameters of the kernel //todo: make it std::vector<std::vector<double>> p if you have multidimensional distributions
		//auxiliary variables for the adaptive-metropolis-hastings update of the kernel parameter, null if conjugate updates can be used
		//the vector is empty if no adaptive-metropolis-hastings updat for any of the kernel parameters, the order is the same to that in p/hp vectors
		std::vector<AdaptiveMetropolisHastingsState *> mh_vars;
		//metropolis-hastings auxiliary variables for update of the hyperparameters, the first dimension (of size nofp) corresponds to the kernel parameter
		//the second to the hyperparamaters of the hyperprior
		std::vector<std::vector<AdaptiveMetropolisHastingsState *>> mh_hvars;

		//empty constructor
		Kernel();
		//constructor which sets the hyperparameters of the kernel
		Kernel(KernelType t, const std::vector<const DistributionParameters *>  & hp);
		//constructor which sets the point parameters of the kernel
		Kernel(KernelType t, const std::vector<double>  & p);
		//constructor which sets both the hyperparameters and the parameters of the kernel
		Kernel(KernelType t, const std::vector< const DistributionParameters *>  & hp,  const std::vector<double>  & p);
		//copy constructor
		Kernel(const Kernel &s);

		//get the hyperparameters for the parameters of the kernel
		void getHyperParams(std::vector<DistributionParameters *> & hparams) const;
		void getHyperParams(std::vector<std::vector<double>> & hparams) const;
		//set the hyperparameters for the parameters of the kernel
		void setHyperParams(const std::vector<DistributionParameters *> & hparams);
		//get the parameters for the parameters of the kernel
		void getParams(std::vector<double> & params) const;
		//set the parameters for the parameters of the kernel
		void setParams(const std::vector<double> & params);
		//save model
		void save(std::ofstream & os) const;
		//load model
		void load(std::ifstream & is);
		//generate the point parameters from the priors
		void generate();
		//reset the point parameters
		void reset();
		virtual Kernel* clone() const=0;
	    virtual ~Kernel();


		virtual double compute(double t, double t0) const =0;//{return 0.0;};
		virtual double compute(double t) const  =0;//{return 0.0;};
		virtual double computeMaxValue(double ts, double te, double t0) const  =0;//{std::cout<<"im called :(";return 0.0;};
		virtual double computeIntegral(double ts, double te, double t0) const  =0;//{return 0.0;};

		virtual void print(std::ostream & os) const =0;//{};
		virtual void printMatlabExpression(double t0, std::string & matlab_expr) const=0;
		//mcmc update for kernel used for effect from type k on type k2. if k<0 then the kernel describes effect from all the types on type k2
		//for the case that the kernel has excitatory effect
		//todo: remove the GeneralizedHawkesProcess::State * ghs argument
		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1 )=0;
		//for the case that the kernel has inhibitory effect
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads=1 )=0;
		

	private:
        friend class boost::serialization::access;
        void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);        
};

/****************************************************************************************************************************************************************************
 *
 * Constant Kernel
 *
******************************************************************************************************************************************************************************/

class ConstantKernel:public Kernel{
	
	public:
		//constructor which sets the hyperparameters of the kernel
		ConstantKernel();
		ConstantKernel(const  std::vector<const DistributionParameters *>  & hp);
		//constructor which sets the point parameters of the kernel
		ConstantKernel(const std::vector<double>   & p);
		//constructor which sets both the hyperparameters and the parameters of the kernel
		ConstantKernel(const std::vector<const DistributionParameters *>  & hp, const std::vector<double>   & p);
		ConstantKernel(const ConstantKernel &s);

		double compute(double t, double t0) const override;
		double compute(double t) const override;
		double computeMaxValue(double ts, double te, double t0) const override;
		double computeIntegral(double ts, double te, double t0) const override;
		void printMatlabExpression(double t0, std::string & matlab_expr) const override;
		void print(std::ostream & os) const override;
		ConstantKernel* clone() const override;
		void save(std::ofstream & os) const;
		void load(std::ifstream & is);

		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1 ) override;
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads=1 ) override;
		
	private:
        friend class boost::serialization::access;
        void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);

};

/****************************************************************************************************************************************************************************
 *
 * Exponential Kernel
 *
******************************************************************************************************************************************************************************/

//phi(t)=c*exp(-d*t)
class ExponentialKernel:public Kernel{
	
	public:
		ExponentialKernel();
		//constructor which sets the hyperparameters
		ExponentialKernel(const std::vector<const DistributionParameters *>   & hp2);
		//constructor which sets the hyperparameters and the point parameters
		ExponentialKernel(const std::vector<const DistributionParameters *>   & hp2,  const std::vector<double>  & p2);
		//constructor which sets the point parameters of the kernel
		ExponentialKernel(const std::vector<double>  & p2);
		ExponentialKernel(const ExponentialKernel &s);
		
		double compute(double t, double t0) const override;
		double compute(double t) const override;
		double computeMaxValue(double ts, double te, double t0) const override;
		double computeIntegral(double ts, double te, double t0) const override;

		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1 ) override;
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads=1 ) override;
		
		void mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1);
		void mcmcExcitatoryExpUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1);

		void printMatlabExpression(double t0, std::string & matlab_expr) const override;
		void print(std::ostream & file) const override;
		ExponentialKernel* clone() const override;
		void save(std::ofstream & os) const;
		void load(std::ifstream & is);

		
	private:
		friend class boost::serialization::access;
		friend class HawkesProcess;
		friend class GeneralizedHawkesProcess;
		


     	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
     	

};

/****************************************************************************************************************************************************************************
 *
 * Logistic Kernel
 *
******************************************************************************************************************************************************************************/


class LogisticKernel:public Kernel{


	public:
		unsigned int d=0; //number of covariates
		std::vector<Kernel *> psi;
		
		LogisticKernel();
		//copy constructor
		LogisticKernel(const LogisticKernel &s);
		//constructor which sets the hyperparameters 
		LogisticKernel( const std::vector<const DistributionParameters *>   & hp2);
		//constructor which sets the hyperparameters and  the point parameters
		LogisticKernel(const std::vector<const DistributionParameters *>   & hp2, const std::vector<double> & p2);
		//constructor which sets the point parameters of the kernel
		LogisticKernel( const std::vector<double>   & p2);
		//constructor which sets the hyperparameters and the basis kernel functions
		LogisticKernel( const std::vector<const DistributionParameters *>   & hp2, const std::vector< Kernel *>  & psi2);
		//constructor which sets the hyperparameters, the point parameters and the basis kernel functions
		LogisticKernel(const std::vector<const DistributionParameters *>   & hp2, const std::vector<double> & p2, const std::vector<  Kernel *>   & psi2);
		//constructor which sets the point parameters of the kernel and the basis kernel functions
		LogisticKernel( const std::vector<double>   & p2, const std::vector< Kernel *>   & psi2);
		//destructor
		~LogisticKernel();
		
        void generate();
        void reset();
		void save(std::ofstream & os) const;
		void load(std::ifstream & is);
		LogisticKernel* clone() const override;
		
		double compute(double t, double t0) const override;
		double compute(double t) const override;
		//t is D-dimensional, for each kernel function it keeps the times at which it will be computed and finally added 
		//in order to get the value of the covariate for multiplication with the corresponding weight
		double compute(double t, std::vector<std::vector<double>> t0) const; 
		double compute(const double *h) const; //h is d- dimensional covariate vector applied on the weights
		double computeArg(const double *h) const; //compute the argument that will be passed through sigmoid
		double computeMaxValue(double ts, double te, double t0) const override;
		double computeIntegral(double ts, double te, double t0) const override;
		void print(std::ostream & os) const override;
		void printMatlabExpression(double t0, std::string & matlab_expr) const override;
		
		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1) override;
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs=0,  unsigned int nof_threads=1) override;
		

	private:

		gsl_matrix *Sigma_tmp=0;//it holds the H^TOmegaH part  for the computation of the posterior covariance
		gsl_vector *mu_tmp=0;//it holds the H^TOmegaz part for the computation of the posterior mean
		
		struct FormCovThreadParams{
			LogisticKernel *lk; //calling object
			unsigned int thread_id; //the id of the thread which executes the subcomputation
			unsigned int event_id_offset; //id of the first event that will be added for the computation of the history-polyagamma kernel matrix
			unsigned int nof_events; //number of events that will be added
			const std::vector<double> & polyagammas; //the polyagammas of observed that will be used for the covariance matrix
			const std::vector<double> & thinned_polyagammas;//the polyagammas of realized events that will be used for the covariance matrix
			double const ** h;
			double const ** thinned_h;
			unsigned int i; //first basis kernel function that will be used for the formation of the (i,j) entry of the cov matrix
			unsigned int j; //second basis kernel function that will be used for the formation of the (i,j) entry of the cov matrix
			double &Sigmaij; //variable where the partial sum of the thread will be stored
			FormCovThreadParams(LogisticKernel *k, unsigned int i, unsigned int o, unsigned int n, const std::vector<double> & p, const std::vector<double> & tp, double const ** h1, double const ** th, unsigned int i1, unsigned int j1, double &s ): lk(k), thread_id(i), event_id_offset(o), nof_events(n), polyagammas(p), thinned_polyagammas(tp), h(h1), thinned_h(th), i(i1), j(j1), Sigmaij(s){};
		
		};
		
		struct FormMeanThreadParams{
			LogisticKernel *lk; //calling object
			unsigned int thread_id; //the id of the thread which executes the subcomputation
			unsigned int event_id_offset; //id of the first event that will be added for the computation of the history-polyagamma kernel matrix
			unsigned int nof_events; //number of events that will be added
			const std::vector<double> & polyagammas; //the polyagammas of observed that will be used for the covariance matrix
			const std::vector<double> & thinned_polyagammas;//the polyagammas of realized events that will be used for the covariance matrix
			double const ** h;
			double const ** thinned_h;
			unsigned int i; //first basis kernel function that will be used for the formation of the (i,j) entry of the cov matrix
			double &mui; //variable where the partial sum of the thread will be stored
			FormMeanThreadParams(LogisticKernel *k, unsigned int i, unsigned int o, unsigned int n, const std::vector<double> & p, const std::vector<double> & tp, double const ** h1, double const ** th, unsigned int i1, double &s): lk(k),thread_id(i), event_id_offset(o),nof_events(n), polyagammas(p), thinned_polyagammas(tp), h(h1), thinned_h(th), i(i1), mui(s){};
		};
		
		friend class boost::serialization::access;
     	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
     	//it updates the covariance matrix of the mutlivariate Gaussian distrbution for the weights in the logistic kernel
     	void mcmcCovUpdate1(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads=1);//it computes the H^TOmegaH part 
     	void mcmcCovUpdate2(GeneralizedHawkesProcess::State & s, int k2, gsl_matrix **Sigma, gsl_matrix **Tau=0, double *Tau_det=0, unsigned int nof_threads=1);//it adds the prior contripution in the posterior cov matrix and it inverts
     	static void * mcmcCovUpdate_(void *p);
     	//it updates the mean of of the mutlivariate Gaussian distrbution for the weights in the logistic kernel
     	void mcmcMeanUpdate1(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads=1);//it computes the H^TOmegaz part of the mean
     	void mcmcMeanUpdate2(GeneralizedHawkesProcess::State & s, int k2, const gsl_matrix *Sigma, gsl_vector **mu, unsigned int nof_threads=1);//it adds the prior contripution in the posterior mean and it multiplies by the cov matrix
     	static void * mcmcMeanUpdate_(void *p);
     	//it updates the interaction weights of the logistic kernel
     	void mcmcWeightsUpdate(GeneralizedHawkesProcess::State & s, int k2, unsigned int nof_threads=1);
     	//it updates the prior (precision) of the weights of the logistic kernel
     	void mcmcPriorUpdate(GeneralizedHawkesProcess::State & ghs, HawkesProcess::State & hs, int k, unsigned int nof_threads=1);
     	//void mcmcPrecisionCollapsedPosterior(GeneralizedHawkesProcess::State & ghs, HawkesProcess::State & hs, int l, int k, unsigned int nof_threads=1);

};

/****************************************************************************************************************************************************************************
 *
 * Power Law Kernel: incomplete implementation
 *
******************************************************************************************************************************************************************************/

//phi(t)=c*(t+gamma)^(-(1+beta))


class PowerLawKernel:public Kernel{
	
	public:
		//constructor which sets the hyperparameters of the kernel
		PowerLawKernel();
		PowerLawKernel(/*enum Effect f, */ const  std::vector<const DistributionParameters *>  & hp);
		//constructor which sets the point parameters of the kernel
		PowerLawKernel(/*enum Effect f, */const std::vector<double>   & p);
		//constructor which sets both the hyperparameters and the parameters of the kernel
		PowerLawKernel(/*enum Effect f, */ const std::vector<const DistributionParameters *>  & hp, const std::vector<double>   & p);
		PowerLawKernel(const PowerLawKernel &s);

		double compute(double t, double t0) const override;
		double compute(double t) const override;
		double computeMaxValue(double ts, double te, double t0) const override;
		double computeIntegral(double ts, double te, double t0) const override;
		void printMatlabExpression(double t0, std::string & matlab_expr) const override;
		void print(std::ostream & os) const override;
		PowerLawKernel* clone() const override;
		void save(std::ofstream & os) const;
		void load(std::ifstream & is);

		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1 ) override;
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs=0,  unsigned int nof_threads=1 ) override;

		
	private:
        friend class boost::serialization::access;
        void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
        void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
        
        void mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs=0,  unsigned int nof_threads=1 );

};



/****************************************************************************************************************************************************************************
 *
 * Rayleigh Kernel: incomplete implementation
 *
******************************************************************************************************************************************************************************/

//phi(t)=c*exp(-d*t)
class RayleighKernel:public Kernel{

	public:
		RayleighKernel();
		//constructor which sets the hyperparameters
		RayleighKernel(/*enum Effect f, */const std::vector<const DistributionParameters *>   & hp2);
		//constructor which sets the hyperparameters and the point parameters
		RayleighKernel(/*enum Effect f, */const std::vector<const DistributionParameters *>   & hp2,  const std::vector<double>  & p2);
		//constructor which sets the point parameters of the kernel
		RayleighKernel(/*enum Effect f, */ const std::vector<double>  & p2);
		RayleighKernel(const RayleighKernel &s);

		double compute(double t, double t0) const override;
		double compute(double t) const override;
		double computeMaxValue(double ts, double te, double t0) const override;
		double computeIntegral(double ts, double te, double t0) const override;

		virtual void mcmcExcitatoryUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1 ) override;
		virtual void mcmcInhibitoryUpdate(int k, int k2, HawkesProcess::State * s, GeneralizedHawkesProcess::State * ghs,  unsigned int nof_threads=1 ) override;

		void printMatlabExpression(double t0, std::string & matlab_expr) const override;
		void print(std::ostream & file) const override;
		RayleighKernel* clone() const override;
		void save(std::ofstream & os) const;
		void load(std::ifstream & is);


	private:
		friend class boost::serialization::access;
		friend class HawkesProcess;
		friend class GeneralizedHawkesProcess;

		void mcmcExcitatoryCoeffUpdate(int k, int k2, HawkesProcess::State * s,  unsigned int nof_threads=1);

     	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
     	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);


};


#endif /* KERNELS_HPP_ */
