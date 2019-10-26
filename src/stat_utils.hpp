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

#ifndef STAT_UTILS_HPP_
#define STAT_UTILS_HPP_


#include <boost/serialization/export.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <iostream>
#include <math.h>
#include "math_utils.hpp"

#define EPS 1e-10
#define INF 1e10
#define NOF_POINTS 100
using namespace boost::accumulators;
using namespace std;

class Kernel;



/****************************************************************************************************************************************************************************
 * Definition of distribution families
******************************************************************************************************************************************************************************/

enum DistributionFamily {Distribution_Gamma, Distribution_Exponential, Distribution_Normal, Distribution_NormalGamma, Distribution_SparseNormalGamma, Distribution_NormalWishart};

struct DistributionParameters{
	DistributionFamily type; //the type of the distribution //TODO: make these params const but for some reason the boost::serialization asserts error!
	unsigned int nofp; //number of parameters for the family of distributions
	
	//for hierarchical models, the prior parameters for the parameters, assume a flat model by default
	std::vector<DistributionParameters *> hp;
	//number of parameters
	//unsigned int nofhp=0;

	//constructor with the type of the family and the number of parameters, flat model
	DistributionParameters(DistributionFamily t, unsigned int n);
	//constructor with the type of the family and the number of parameters, hierarchical model
	DistributionParameters(DistributionFamily t, unsigned int n, const std::vector<DistributionParameters *> & hp);


	virtual void getParams(std::vector<double> & params) const=0;
	virtual void save(std::ofstream & os) const=0;
	virtual void load(std::ifstream & is)=0;
	virtual void print(std::ostream &file)=0;
	virtual DistributionParameters *clone() const =0;
	virtual ~DistributionParameters ();
	
	private:
    	friend class boost::serialization::access;
    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);

};

struct NormalParameters: DistributionParameters{
	double mu; //mean of normal distribution, it may have a normal prior
	//in case of independent normal variables
	double sigma; //std matrix (if it's diagonal),
	double tau;   //precision matrix (inverse of variance)//
	
	NormalParameters();
	NormalParameters(double mu2, double sigma2);//flat model
	NormalParameters(double mu2, double sigma2, std::vector<DistributionParameters *> & hp);//hierarchical model
	NormalParameters(const NormalParameters &h);

	void getParams(std::vector<double> & params) const override;
	void print(std::ostream &file) override;
	void save(std::ofstream & os) const override;
	void load(std::ifstream & is) override;
	NormalParameters *clone() const override;
	
	private:
		friend class boost::serialization::access;
		friend class GeneralizedHawkesProcess;
		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
};


//Exponential parameters
struct ExponentialParameters: DistributionParameters{

	double lambda;

	ExponentialParameters();
	ExponentialParameters(double l0);
	ExponentialParameters(double l0, std::vector<DistributionParameters *> & hp);
	ExponentialParameters(const ExponentialParameters &h);

	void getParams(std::vector<double> & params) const override;
	void print(std::ostream &file) override;
	void save(std::ofstream & os) const override;
	void load(std::ifstream & is) override;
	ExponentialParameters *clone() const override;

	private:

    	friend class boost::serialization::access;
    	friend class GeneralizedHawkesProcess;

    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);

};



//Gamma parameters
struct GammaParameters: DistributionParameters{

	double alpha;
	double beta;
	
	GammaParameters();
	GammaParameters(double a0, double b0);
	GammaParameters(double a0, double b0, std::vector<DistributionParameters *> & hp);
	GammaParameters(const GammaParameters &h);

	void getParams(std::vector<double> & params) const override;
	void print(std::ostream &file) override;
	void save(std::ofstream & os) const override;
	void load(std::ifstream & is) override;

	GammaParameters *clone() const override;

	private:

    	friend class boost::serialization::access;
    	friend class GeneralizedHawkesProcess;

    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);

};

//Normal Gamma parameters. an inverse-scale parametrization is assumed for the gamma distrbution
struct NormalGammaParameters: DistributionParameters{

	//normal gamma hyperparameters
	double mu;
	double kappa;
	double alpha;
	double beta;

	NormalGammaParameters();
	NormalGammaParameters(double m0, double k0, double a0, double b0);
	NormalGammaParameters(double m0, double k0, double a0, double b0, std::vector<DistributionParameters *> & hp);
	NormalGammaParameters(const NormalGammaParameters &h);


	void getParams(std::vector<double> & params) const override;
	void print(std::ostream &file) override;
	void save(std::ofstream & os) const override;
	void load(std::ifstream & is) override;

	NormalGammaParameters *clone() const override;

	private:

    	friend class boost::serialization::access;
    	friend class GeneralizedHawkesProcess;

    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);


};

//Normal Gamma parameters to model sparsity for the gaussian samples, where the sparsity depends on alpha and bounded by alpha_0
//an inverse-scale parametrization is assumed for the gamma distrbution. The model is:
// tau \sim Gamma(nu*alpha^2+alpha_0, betta)
// w \sim N(0, kappa*tau^-1)
struct SparseNormalGammaParameters: DistributionParameters{

	//normal gamma hyperparameters
	double lambda;
	double alpha;// it is drawn from a prior
	double alpha_tau;
	double beta_tau;
	double nu_tau;
	double alpha_mu;
	double nu_mu;
	
	ActivationFunction *phi_tau; 
	ActivationFunction *phi_mu;

	SparseNormalGammaParameters();

	SparseNormalGammaParameters(double l0, double at, double bt, double nt, double am, double nm);

	SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm);

	SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, std::vector<DistributionParameters *> & hp);

	SparseNormalGammaParameters(double l0, double at, double bt, double nt, double am, double nm, ActivationFunction *pt, ActivationFunction *pm);

	SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, ActivationFunction *pt, ActivationFunction *pm);

	SparseNormalGammaParameters(double l0, double a, double at, double bt, double nt, double am, double nm, std::vector<DistributionParameters *> & hp, ActivationFunction *pt, ActivationFunction *pm);

	SparseNormalGammaParameters(const SparseNormalGammaParameters &h);

	void getParams(std::vector<double> & params) const override;
	void print(std::ostream &file) override;
	void save(std::ofstream & os) const override;
	void load(std::ifstream & is) override;

	SparseNormalGammaParameters *clone() const override;

	private:

    	friend class boost::serialization::access;
    	friend class GeneralizedHawkesProcess;

    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);


};

//Normal wishart parameters.
//struct NormalWishartParameters: DistributionParameters{
//	//normal gamma hyperparameters
//	const gsl_vector * mu; //mean
//	const gsl_matrix  *Tau; //precision
//	std::vector<double> mu_ser; //for serializing the mean
//	std::vector<double> Tau_ser; //for serializing the precision
//
//	const double kappa;
//	const double nu;
//	const unsigned int p; //dimension of the distribution
//
//	NormalWishartParameters();
//	NormalWishartParameters( gsl_vector * m0,  gsl_matrix  * T0, double k, double n, unsigned int p2) ;
//	NormalWishartParameters(const NormalWishartParameters &h) ;
//
//	void print(std::ostream &file) override;
////	void getParams(std::vector<double> & params) const override;
////	void setParams(const std::vector<double> & params) override;
//	void save(std::ofstream & os) const override;
//	void load(std::ifstream & is) override;
//
//	NormalWishartParameters *clone() const override;
//
//	private:
//
//    	friend class boost::serialization::access;
//    	friend class GeneralizedHawkesProcess;
//
//    	void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
//    	void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
//    	void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
//    	void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
//};


/****************************************************************************************************************************************************************************
 * Variables for Adaptive Metropolis Hastings
******************************************************************************************************************************************************************************/
struct AdaptiveMetropolisHastingsState{

	//metropolis-hastings auxiliary variables
    double sample_mean=0; //the sample mean of the previous step
    double sample_cov=0;  //the sample covariance of the previous step
    double mu_t; //mean of the gaussian proposal
    double samples_n=0; //nof metropolis-hastings steps

    //metropolis-hastings parameters
	const double sd; //scaling paramater
    const double t0; //initial period of the covariance matrix
    const double c0; //initial variance for the proposal
    const double eps;//scaling paramater for improving the condition of the matrix

    AdaptiveMetropolisHastingsState();
    AdaptiveMetropolisHastingsState(double s, double t, double c, double e);
    AdaptiveMetropolisHastingsState(double s, double t, double c, double e, double m, double sc);
    AdaptiveMetropolisHastingsState(const AdaptiveMetropolisHastingsState &v);
	
    void getVariables(std::vector<double> & vars) const;
	void setVariables(const std::vector<double> & vars) ;
    void save(std::ofstream & os) const;
	void load(std::ifstream & is);
	void print(std::ostream &file);
	AdaptiveMetropolisHastingsState *clone() const;
	void mcmcUpdate();

	private:
		friend class boost::serialization::access;
		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
};

//it performs a metropolis hastings update: it samples from the proposal and rejects or accepts the new value, according to the ratio of lieklihoods computed by likelihoodRatio
bool MetropolisHastings(DistributionParameters * proposal, std::function<double(double)> likelihoodRatio, double &p);

//it performs a metropolis hastings update: it samples from the multivariate proposal and rejects or accepts the new value, according to the ratio of lieklihoods computed by likelihoodRatio
bool MetropolisHastings(DistributionParameters * proposal, std::function<double(std::vector<double>)> likelihoodRatio, std::vector<double> &p);

/****************************************************************************************************************************************************************************
 * Simulation of Poisson Processes
******************************************************************************************************************************************************************************/
//Generates event arrivals coming from a homogeneous Poisson Process. If N>0, it returns only the next N events that lie in the [s,e], otherwise it returns the full realization of the process.
std::vector<double> simulateHPoisson(double lambda, double s, double e, int N=-1);

//Generates event arrivals coming from a non-homogeneous Poisson Process. If N>0, it returns only the next N events that lie in the [s,e], otherwise it returns the full realization of the process.
//see: (Lewis and Shedler, Simulation of nonhomogenous Poisson processes by thinning)
std::vector<double> simulateNHPoisson(const Kernel & phi, double t0, double ts, double te, int N=-1);

/****************************************************************************************************************************************************************************
 * Computation of pdf (probability density functions)
******************************************************************************************************************************************************************************/

// it returns the pdf value of the gamma distribution (assuming an inverse-scale parametrization at x).
//TODO: check if ublas provides this
double gamma_pdf(double alpha, double beta, double x);

//it returns the pdf value of the exponential distribution (assuming an inverse-scale parametrization at x).
//TODO: check if ublas provides this  https://www.boost.org/doc/libs/1_62_0/libs/math/doc/html/special.html
double exp_pdf(double lambda, double x);

double normal_pdf(float x, float mu=0, float sigma=1.0);



/****************************************************************************************************************************************************************************
 * Computation of sample statistics
******************************************************************************************************************************************************************************/

//it computes the autocorrelation function of samples of theta whose distance is lag.
void acf( const std::vector<double> & theta,std::vector<double> & acf_v);

void acf( const boost::numeric::ublas::vector<double> & theta,std::vector<double> & acf_v);

//it computes the posterior mean of the parameter, with samples from up to an iteration.
void meanplot(const std::vector<double> & theta, std::vector<double> & mean_v);

void meanplot(const boost::numeric::ublas::vector<double> & theta, std::vector<double> & mean_v);


//it returns the mode of the distribution estimated by gaussian kernel regression from samples
double sample_mode(const boost::numeric::ublas::vector<double> &samples);

//it returns the mean of the distribution estimated by gaussian kernel regression from samples
double sample_mean(const boost::numeric::ublas::vector<double> &samples);

/****************************************************************************************************************************************************************************
 * Density estimation functions
******************************************************************************************************************************************************************************/

//it computes the kernel bandwidth by silverman's rule
double gkr_h(const boost::numeric::ublas::vector<double> &samples);

//gaussian kernel estimation from distribution samples, for a number of points in the range
void gkr(const boost::numeric::ublas::vector<double> &samples, boost::numeric::ublas::vector<double> &x, boost::numeric::ublas::vector<double> & p_x, unsigned int nofx, double h=-1.0);

//gaussian kernel estimation from distribution samples at specific points
void gkr(const boost::numeric::ublas::vector<double> &samples, const boost::numeric::ublas::vector<double> &x, boost::numeric::ublas::vector<double> & p_x, double h=-1.0);


/****************************************************************************************************************************************************************************
 * sampling from some distributions functions
******************************************************************************************************************************************************************************/

//it draws a sample from the normal gamma distribution
void normal_gamma(const NormalGammaParameters &h, double &mu, double & lambda);

//it draws a sample from the normal wishart distribution
// void normal_wishart(const NormalWishartParameters &h, gsl_vector **mu, gsl_matrix ** Tau);

//it returns a sample for the mean and precision of the normal model assuming a normal gamma prior, given its observations w and the hyperparameters h
void sample_ng_normal_model(const NormalGammaParameters &h, const std::vector<double> & w, double &mu, double & lambda);

//it returns a sample for the mean and precision of the normal model assuming a normal wishart prior, given its observations w and the hyperparameters h
//void sample_nw_normal_model(const NormalWishartParameters &h, const std::vector<gsl_vector *> & w, gsl_vector **mu, gsl_matrix ** Tau);

//it draws one sample from a multivariate gaussian.
//using the Cholesky decomposition L of the variance-covariance matrix, following "Computational Statistics" from Gentle (2009), section 7.4.
void multivariate_normal(const gsl_vector *mu, const gsl_matrix *L, std::vector<double> &sample);

void multivariate_normal(const gsl_vector *mu, const gsl_matrix *L, gsl_vector ** result);




#endif /* STAT_UTILS_HPP_ */
