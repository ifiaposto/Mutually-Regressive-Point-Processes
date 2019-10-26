/*
==============================================================================
 * Copyright 2019
 * Author: Ifigeneia Apostolopoulou iapostol@andrew.cmu.edu, ifiaposto@gmail.com.
 * All Rights Reserved.
 * 
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
#ifndef LEARNING_PARAMS_HPP_
#define LEARNING_PARAMS_HPP_

#include <fstream>
#include <boost/serialization/export.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


// mcmc related parameters
const unsigned int RUNS=1; //number of runs/ mcmc chains
const unsigned int NOF_BURNIN_ITERS=5000; //number of burnin iterations for the markov chain
const unsigned int MAX_NUM_ITER=5000; //number of mcmc iterations (excluding the burnin)


 //batch of training samples steps, after which the loglikelihood will be reported
const unsigned int SAMPLES_STEP=1000;
const unsigned int MCMC_SAVE_PERIOD=12000; //number of seconds after which the current posterior samples will be saved for backup

// 
// multi-threading parameters
const unsigned int MCMC_NOF_FIT_THREADS=1; //number of threads across which the runs/initial conditions of the mcmc will be shared
const unsigned int MCMC_ITER_NOF_THREADS=1; //number of threads across which the param of different types/ latent variables will be updated
const unsigned int PLOT_NOF_THREADS=10; //number of threads for the generation of the learning plots
const unsigned int NOF_PLOT_POINTS=100000; //number of points, for which functions related to learning (intensities, posteriors etc) will be computed and ploted
const unsigned int MODEL_SIMULATION_THREADS=1;

//adaptive Metropolis Hastings parameters
const double C_T_AM_SD=10;  //for the multiplicative coefficient in the mutually trigerring kernel functions
//const double C_T_AM_T0=500; //initial period of the covariance matrix for the multiplicative coefficient in the mutually trigerring kernel functions
const double C_T_AM_T0=NOF_BURNIN_ITERS+MAX_NUM_ITER;
//const double C_T_AM_T0=100;
const double C_T_AM_C0=0.01; //initial variance for the proposal of the multiplicative coefficient in the mutually trigerring kernel functions

const double D_T_AM_SD=10;  //for the decaying coefficient in the mutually trigerring kernel functions
const double D_T_AM_T0=3000; //initial period of the covariance matrix for the decaying coefficient in the mutually trigerring kernel functions
const double D_T_AM_C0=0.1; //initial variance for the proposal of the decaying coefficient in the mutually trigerring kernel functions

const double C_H_AM_SD=10;   //for the multiplicative coefficient in the history kernel function of the logistic kernel
const double C_H_AM_T0=1000;  //initial period of the covariance matrix for the multiplicative coefficient in the history kernel function of the logistic kernel
const double C_H_AM_C0=0.1; //initial variance for the proposal of the multiplicative coefficient in the history kernel function of the logistic kernel

const double D_H_AM_SD=10;   //for the decaying coefficient in the history kernel function of the logistic kernel
const double D_H_AM_T0=1000; //initial period of the covariance matrix for the decaying coefficient in the history kernel function of the logistic kernel
const double D_H_AM_C0=0.01;    //initial variance for the proposal of the decaying coefficient in the history kernel function of the logistic kernel

const double T_W_AM_SD=10;    //for the precision of the interaction weights in the logistic kernel
const double T_W_AM_T0=3000;  //initial period of the precision of the interaction weights in the logistic kernel
const double T_W_AM_C0=0.1;  //initial variance for the precision of the interaction weights in the logistic kernel

//other constants

const unsigned int MC_NOF_SAMPLES=1000; //nof samples for monte carlo estimation

const unsigned int MODEL_SIMULATION_SAMPLES=1000; //nof model simulation for the prediction tasks

const double AM_EPSILON=1e-4;
const double KERNEL_EPS=1e-20;
const double KERNEL_INF=1e20;


struct MCMCParams{
	const unsigned int runs=RUNS; //number of runs/ mcmc chains
	const unsigned int nof_burnin_iters=NOF_BURNIN_ITERS; //number of burnin iterations for the markov chain
	const unsigned int max_num_iter=MAX_NUM_ITER; //number of mcmc iterations (excluding the burnin)
	const unsigned int samples_step=SAMPLES_STEP;  //batch of training samples steps, after which the loglikelihood will be reported
	const unsigned int mcmc_save_period=MCMC_SAVE_PERIOD; //number of seconds after which the current posterior samples will be saved for backup
	
	unsigned int nof_sequences=1;//nof sequences per run for the training of the model
	bool profiling=false; //produce log files to report learning time or not
	
	std::string dir_path_str; //directory for the learning results
	
	// multi-threading parameters
	const unsigned int mcmc_fit_nof_threads=MCMC_NOF_FIT_THREADS; //number of threads across which the runs/initial conditions of the mcmc will be shared
	const unsigned int mcmc_iter_nof_threads=MCMC_ITER_NOF_THREADS; //number of threads across which the param of different types/ latent variables will be updated
	const unsigned int plot_nof_threads=PLOT_NOF_THREADS; //number of threads for the generation of the learning plots
	const unsigned int nof_plot_points=NOF_PLOT_POINTS; //number of points, for which functions related to learning (intensities, posteriors etc) will be computed and ploted
	
	MCMCParams();
	MCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string  d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp);
	MCMCParams(unsigned int n, bool p, const std::string & d);

	MCMCParams(const MCMCParams &p);
	
	
	void print(std::ostream & file);
	
	private:
    friend class boost::serialization::access;
    void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
    void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
    void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
    void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
};

struct HawkesProcessMCMCParams:MCMCParams{
	const double d_t_am_sd=D_T_AM_SD;  //for the decaying coefficient in the mutually trigerring kernel functions
	const double d_t_am_t0=D_T_AM_T0; //initial period of the covariance matrix for the decaying coefficient in the mutually trigerring kernel functions
	const double d_t_am_c0=D_T_AM_C0; //initial variance for the proposal of the decaying coefficient in the mutually trigerring kernel functions
	
	HawkesProcessMCMCParams();
	HawkesProcessMCMCParams(const HawkesProcessMCMCParams &p);
	HawkesProcessMCMCParams(unsigned int n, bool p, const std::string & d);
	HawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp);
	HawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp,double sd, double t0, double c0);

	
	void print(std::ostream & file);
	private:
	friend class boost::serialization::access;
	
	void serialize(boost::archive::text_oarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::text_iarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::binary_iarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::binary_oarchive &ar,  unsigned int version);
};


struct GeneralizedHawkesProcessMCMCParams:HawkesProcessMCMCParams{
	
	const double c_t_am_sd=C_T_AM_SD;  //for the multiplicative coefficient in the mutually trigerring kernel functions
	const double c_t_am_t0=C_T_AM_T0; //initial period of the covariance matrix for the multiplicative coefficient in the mutually trigerring kernel functions
	const double c_t_am_c0=C_T_AM_C0; //initial variance for the proposal of the multiplicative coefficient in the mutually trigerring kernel functions
	
	const double c_h_am_sd=C_H_AM_SD;   //for the multiplicative coefficient in the history kernel function of the logistic kernel
	const double c_h_am_t0=C_H_AM_T0;  //initial period of the covariance matrix for the multiplicative coefficient in the history kernel function of the logistic kernel
	const double c_h_am_c0=C_H_AM_C0; //initial variance for the proposal of the multiplicative coefficient in the history kernel function of the logistic kernel

	const double d_h_am_sd=D_H_AM_SD;   //for the decaying coefficient in the history kernel function of the logistic kernel
	const double d_h_am_t0=D_H_AM_T0; //initial period of the covariance matrix for the decaying coefficient in the history kernel function of the logistic kernel
	const double d_h_am_c0=D_H_AM_C0;    //initial variance for the proposal of the decaying coefficient in the history kernel function of the logistic kernel

	const double t_w_am_sd=T_W_AM_SD;    //for the precision of the interaction weights in the logistic kernel
	const double t_w_am_t0=T_W_AM_T0;  //initial period of the precision of the interaction weights in the logistic kernel
	const double t_w_am_c0=T_W_AM_C0;  //initial variance for the precision of the interaction weights in the logistic kernel
	
	GeneralizedHawkesProcessMCMCParams();
	
	GeneralizedHawkesProcessMCMCParams(const GeneralizedHawkesProcessMCMCParams &p);
	
	GeneralizedHawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp);
		
	GeneralizedHawkesProcessMCMCParams(unsigned int n, bool p, const std::string & d);
	
	GeneralizedHawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp, double sd, double t0, double c0,
									   double tcsd, double tct0, double tcc0,
									   double hcsd, double hct0, double hcc0,
									   double hdsd, double hdt0, double hdc0,
									   double twsd, double twt0, double twc0);
	void print(std::ostream & file);
	private:
	friend class boost::serialization::access;
	
	void serialize(boost::archive::text_oarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::text_iarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::binary_iarchive &ar,  unsigned int version);
	
	void serialize(boost::archive::binary_oarchive &ar,  unsigned int version);
	
};



#endif /* LEARNING_PARAMS_HPP_ */


