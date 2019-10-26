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

#ifndef HAWKESPROCESS_HPP_
#define HAWKESPROCESS_HPP_

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iterator>
#include <algorithm>
#include <vector>
#include <random>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/spirit/include/karma.hpp>
#include "PointProcess.hpp"

//#define _POSIX_C_SOURCE 199309L

class Kernel;
class ConstantKernel;

using namespace boost::accumulators;
using namespace boost::numeric::operators;

class HawkesProcess:public virtual PointProcess{
	public:
			
		std::vector<ConstantKernel *> mu;
		std::vector<std::vector<Kernel *>> phi;//phi[k][k2]==effect from type k to type k2
		
		//-------------------------------------------------------------------------------------- the posterior point estimates   ---------------------------------------------------------------------------------------------------------------------------
		//the posterior mean point estimates for the kernel parameters
		std::vector<ConstantKernel*> post_mean_mu;
		
		std::vector<std::vector<Kernel *>> post_mean_phi;
		
		//the posterior mode point estimates for the kernel parameters
		std::vector<ConstantKernel*> post_mode_mu;

		std::vector<std::vector<Kernel *>> post_mode_phi;

	public:           

		/****************************************************************************************************************************************************************************
		 * Model Construction and Destruction Methods
		******************************************************************************************************************************************************************************/
		
		/***************************************************   anononymous point processes    *********************************************/
		HawkesProcess(); //empty process
		
		HawkesProcess(unsigned int k, double ts, double te); //empty multivariate process
		
		HawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m); //multivariate mutually independent poisson process
		
		HawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p); //hawkes process
		
		/***************************************************   named point processes    *********************************************/
		HawkesProcess(std::string name);  //empty process

		HawkesProcess(std::string name, unsigned int k, double ts, double te);  //empty multivariate process
		
		HawkesProcess(std::string name,unsigned int k, double ts, double te, std::vector<ConstantKernel *> m); //multivariate mutually independent poisson process

		HawkesProcess(std::string name,unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p); //hawkes process
		
		HawkesProcess(std::ifstream &file); //import model from file

		~HawkesProcess();
		
		/****************************************************************************************************************************************************************************
		 * Model Utility Methods
		******************************************************************************************************************************************************************************/

		std::string createModelDirectory() const override;
		
		std::string createModelInferenceDirectory() const override;
		
		void print(std::ostream &file)const override;

		/****************************************************************************************************************************************************************************
		 * Model Likelihood  Methods
		******************************************************************************************************************************************************************************/
		//these are joint likelihood methods for the observed and latent variables of the model
		double likelihood(const EventSequence & seq, bool normalized=true, double t0=-1) const override;

		//likelihood (of only mutual excitation part)
	    double likelihood(const EventSequence & seq, int k, int k2, double t0=-1) const override;

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		double loglikelihood(const EventSequence & seq, bool normalized=true, double t0=-1) const override;

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		void posteriorModeLoglikelihood(EventSequence & seq, std::vector<double> &logl, bool normalized=true, double t0=-1) override;

		//log-likelihood (of only the mutual excitation part)
		double loglikelihood(const EventSequence & seq, int k, int k2, double t0=-1) const override;
		
		/****************************************************************************************************************************************************************************
		 * Synthetic Model Generation Methods
		******************************************************************************************************************************************************************************/
		
		void generate() override;
		
		/****************************************************************************************************************************************************************************
		 *  Model Simulation Methods
		******************************************************************************************************************************************************************************/
		//non static methods
		EventSequence simulate(unsigned int id=0) override;
		
		void simulate(EventSequence & seq, double dt) override;//augment event sequence seq, with other dt simulation
		
		void simulate(Event *e, int k, double ts, double te, EventSequence &s) override;
		
		void simulateNxt(const EventSequence & seq, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N) override;
		
		//void simulateNxt(const EventSequence & seq, double th, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) override;
		
		//static methods: TODO: in the signature use also const qualifier???
		static EventSequence simulate(std::vector< ConstantKernel *> & mu, std::vector<std::vector<Kernel *>> & phi, double start_t, double end_t, std::string name="", unsigned int id=0);
		
		static void simulate(std::vector< ConstantKernel *> & mu, std::vector<std::vector<Kernel *>> & phi, EventSequence &s, double dt); //augment event sequence seq, with other dt simulation

		static void simulate(std::vector<std::vector<Kernel *>> & phi, Event *e, int k, double ts, double te, EventSequence &s);
		
		static void simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N=1); //augment event sequence seq with the next N events which occur at most after dt time units generated by the intensities mu and phi

		//simulation of the cluster point process to get only the next event in the interval [start_t, end_t] when the events are observed up to time th and there are no events in [th, start_t]
		static void simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s, double th, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N=1); //augment event sequence seq with the next N events which occur at most after dt time units generated by the intensities mu and phi

		//static void simulateNxt(std::vector<std::vector<Kernel *>> & phi, Event *e, int k, double ts, double te, EventSequence &s, unsigned int N=1); //augment event sequence seq with the next N events which occurs at most after dt time units generated by the intensities mu and phi

		/****************************************************************************************************************************************************************************
		 * Model Memory Methods
		******************************************************************************************************************************************************************************/
	    void save(std::ofstream &file) const override; 
	    
	    void load(std::ifstream &file) override; 
	    
	    void saveParameters(std::ofstream &file)  const override;
	    
	    void loadData(std::ifstream &file, std::vector<EventSequence> & data, unsigned int file_type=0, const std::string & name="", double t0=-1,double t1=-1) override;

	    void loadData(std::vector<std::ifstream> &file, std::vector<EventSequence> & data, unsigned int file_type=0, double t0=-1,double t1=-1) override;
		
		/****************************************************************************************************************************************************************************
		 * Model Inference Methods
		******************************************************************************************************************************************************************************/
		void fit(MCMCParams const * const p) override;

		void initProfiling(unsigned int runs,  unsigned int max_num_iter) override;
		
		void writeProfiling(const std::string &dir_path_str)const override;

		//it sets the posterior mean and mode parameters to the kernels
		void setPosteriorParams() override;
		

		//it sets the parents of the events in the data of the model to their posterior modes (set from the posterior samples of the model)
		void setPosteriorParents() override;

		//it sets the parents of the events of the sequence seq to their posterior modes (set from the posterior samples of the model)
		void setPosteriorParents(EventSequence & seq) override;

		//it discards the first samples of the mcmc inference, the burnin samples may have to be preserved for plotting purposes
		void flushBurninSamples(unsigned int nof_burnin=NOF_BURNIN_ITERS) override;
		
		/****************************************************************************************************************************************************************************
		 * Goodness-of-fit Methods
		******************************************************************************************************************************************************************************/
		static void goodnessOfFitMatlabScript(const std::string & dir_path_str, const std::string & seq_path_str, std::string model_name, std::vector<EventSequence> & data, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi) ;
		
		/****************************************************************************************************************************************************************************
		 * Model Testing Methods
		******************************************************************************************************************************************************************************/
		
		//void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples=0, const std::string & true_logl_summary=0, bool normalized=true)  override;
		void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, double t0, double t1, unsigned int burnin_samples=0, const std::string & true_logl_summary=0, bool normalized=true)  override;
				
		
		void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples=0, const std::string & true_logl_summary=0, bool normalized=true)  override;
		
		//TODO: prediction tasks incomplete
		void testPredictions(EventSequence & seq, const std::string &test_dir_path_str, double & mode_rmse, double & mean_rmse, double & mode_errorrate, double & mean_errorrate, double t0=-1) override;
		
		
		/****************************************************************************************************************************************************************************
		 * Model Prediction Methods: //TODO: prediction tasks incomplete
		******************************************************************************************************************************************************************************/
		

		struct predictNextArrivalTimeThreadParams{
			unsigned int thread_id;
			unsigned int nof_samples;
			double & mean_arrival_time;
			const std::vector<ConstantKernel *>  & mu;
			const std::vector<std::vector<Kernel*>> & phi;
			const EventSequence & seq;
			double t_n;
			
			predictNextArrivalTimeThreadParams(unsigned int i, unsigned int n, double &pm, const std::vector<ConstantKernel *>  & m, const std::vector<std::vector<Kernel*>> &p,  const EventSequence &s, double t): 
				thread_id(i), 
				nof_samples(n), 
				mean_arrival_time(pm), 
				mu(m), 
				phi(p), 
				seq(s),
				t_n(t){};
		};
		
		struct predictNextTypeThreadParams{
			unsigned int thread_id;
			unsigned int nof_samples;
			//double & mean_arrival_time;
			std::map<int, unsigned int> & type_counts;
			const std::vector<ConstantKernel *>  & mu;
			const std::vector<std::vector<Kernel*>> & phi;
			const EventSequence & seq;
			double t_n;
			
			predictNextTypeThreadParams(unsigned int i, unsigned int n, std::map<int, unsigned int> & tc, const std::vector<ConstantKernel *>  & m, const std::vector<std::vector<Kernel*>> &p,  const EventSequence &s, double t): 
				thread_id(i), 
				nof_samples(n), 
				type_counts(tc), 
				mu(m), 
				phi(p), 
				seq(s),
				t_n(t){};
		};

		
		//it computes the probability that the next arrival after tn will happen at t
		static double computeNextArrivalTimeProbability(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, double tn, double t);
		
		static double predictNextArrivalTime(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES, unsigned int nof_threads=MODEL_SIMULATION_THREADS);
		
		static void * predictNextArrivalTime_(void *p);
		
		static int predictNextType(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, std::map<int, double> & type_prob, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES, unsigned int nof_threads=MODEL_SIMULATION_THREADS);
		
		static void * predictNextType_(void *p);
		
		void predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, double t0=-1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) override;
		
		static void predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str , const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, double * rmse, double * errorrate, double t0=-1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) ;
				
		/****************************************************************************************************************************************************************************
		 * Model Plot Methods
		******************************************************************************************************************************************************************************/
		void generateMCMCplots(unsigned int samples_step=SAMPLES_STEP, bool true_values=false, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool write_png_file=false, bool write_csv_file=false) override;
		
		void generateMCMCplots(const std::string & dir_path_str, unsigned int samples_step=SAMPLES_STEP, bool true_values=false, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool write_png_file=false, bool write_csv_file=false) override;
		
		void generateMCMCTracePlots(const std::string & dir_path_str, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCTracePlots(bool write_png_file=false, bool write_csv_file=false)  const override;
		
		void generateMCMCMeanPlots(const std::string & dir_path_str, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCMeanPlots(bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCAutocorrelationPlots(const std::string & dir_path_str, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCAutocorrelationPlots(bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCPosteriorPlots(const std::string & dir_path_str, bool true_values, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCPosteriorPlots(bool true_values,bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCIntensityPlots(const std::string & dir_path_str, bool true_values, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCIntensityPlots(bool true_values, bool write_png_file=false, bool write_csv_file=false) const override;
		
		void generateMCMCTrainLikelihoodPlots(unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true, const std::string & plot_dir_path_str="") const override;
		
		void generateMCMCTestLikelihoodPlots(const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true,  double t0=-1, double t1=-1, const std::string & plot_dir_path_str="") const override;

		
		void plotPosteriorIntensity(const std::string & dir_path_str, unsigned int k, const EventSequence &s,  bool true_intensity, unsigned int sequence_id=0, unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=false, bool write_csv_file=false) const;
		/****************************************************************************************************************************************************************************
		 * Model Intensity Methods
		******************************************************************************************************************************************************************************/
		//print intensity functions of the event sequence s as matlab functions
		static void printMatlabFunctionIntensities(const std::string & dir_path_str, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi, const EventSequence &s);//

		//print intensity function of type k of the event sequence s as matlab functions
		static void printMatlabFunctionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, std::string & matlab_lambda);

		static void printMatlabFunctionIntensity(const ConstantKernel * mu,const EventSequence &s, std::string & matlab_lambda);
		
		static void printMatlabExpressionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, std::string & matlab_lambda);

		static void printMatlabExpressionIntensity(const ConstantKernel * mu,const EventSequence &s, std::string & matlab_lambda);


		//compute and return value of intensity of type k at time t
		double computeIntensity(unsigned int k, const EventSequence &s, double t) const override;

		//compute intensity at equally spaced "nofps" timepoints within the observation window
		void computeIntensity(unsigned int k, const EventSequence &s, boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v, unsigned int nofps=NOF_PLOT_POINTS) const override;
		
		//compute intensity at determined timepoints t
		void computeIntensity(unsigned int k, const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v) const override;

		static double computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, double t);
		
		static double computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, double th, double t);

		static void computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v, unsigned int nofps=NOF_PLOT_POINTS);
		
		static void computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v);

		void plotIntensity(const std::string &filename, unsigned int k, const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=false, bool write_csv_file=false) const override;

		void plotIntensities(const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=false, bool write_csv_file=false) const override ;

		void plotIntensities(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=false, bool write_csv_file=false) const override;
	
	protected:
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//the mcmc state, with current sample of model parameters and laten variables
		struct State{
			unsigned int K;
			double start_t;
			double end_t;
			
			std::vector<ConstantKernel *> mu; //sample for the backround intensity parameters
			std::vector<std::vector<Kernel*>> phi; //sample for the exciting kernel parameters
			
					
			std::vector<EventSequence> seq; //sample for the observed sequences (the branching structure of the events)
			std::vector<std::vector<Event *>> observed_seq_v;//vector structure of the realized events, needed for the parent sampling
			std::vector<std::vector<Event *>> seq_v; //vector structure of both the realized and thinned events
			
			unsigned int mcmc_iter=0;//number of previous mcmc samples
			
			HawkesProcessMCMCParams const * mcmc_params;
			
			private:
			
		        friend class boost::serialization::access;
		        friend class HawkesProcess;
		        
		        void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		        void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		        void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		        void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
		        void print (std::ostream &file) const;
		        
		 };
		
		std::vector<State> mcmc_state; //the current state of the mcmc runs
		
	private:
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//	synchronization variables
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		std::vector<pthread_mutex_t *> update_event_mutex;//1 for each thread so that it's children threads don't change simulataneously the parent structure of the sequence
		unsigned int nof_fit_threads_done; //number of threads which have finished the inference
		pthread_mutex_t fit_mtx;
		pthread_cond_t fit_con;
		std::vector<pthread_mutex_t *> save_samples_mtx;
		
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//auxiliary structs for the thread's parameters 
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		//struct which holds the parameters for the thread which is responsible for running a mcmc run
		struct InferenceThreadParams{
			unsigned int thread_id; //the id of the thread which executes the mcmc run
			HawkesProcess *hp;//calling object
			unsigned int run_id_offset; //id of the first mcmc run of the thread
			unsigned int runs; //number of runs that the thread will run
			unsigned int burnin_iter;//number of burnin samples in each run
			unsigned int max_num_iter;//number of iterations/ samples to get from each run
			unsigned int iter_nof_threads;//number of threads for each iteration of the mcmc
			bool profiling;
			
			unsigned int fit_nof_threads;
			unsigned int *  nof_fit_threads_done;
			pthread_mutex_t *fit_mtx;
			pthread_cond_t *fit_con;
			
			InferenceThreadParams(unsigned int i, HawkesProcess *h , unsigned int o, unsigned int r, unsigned int b, unsigned int m, unsigned int t, bool p, unsigned int nt, unsigned int *d, pthread_mutex_t * fm, pthread_cond_t *fc): 
				thread_id(i),
				hp(h), 
				run_id_offset(o), 
				runs(r), 
				burnin_iter(b), 
				max_num_iter(m), 
				iter_nof_threads(t), 
				profiling(p),
				fit_nof_threads(nt),
				nof_fit_threads_done(d),
				fit_mtx(fm),
				fit_con(fc)
				{};
		};
		
		struct UpdateEventThreadParams{
			unsigned int thread_id; //the id of the thread which executes the mcmc run
			HawkesProcess *hp;//calling object
			unsigned int run_id; //the id of the thread which executes the mcmc run
			unsigned int seq_id; //the id of the training event sequence
			//unsigned int parent_thread_id;
			unsigned int event_id_offset; //id of the first event of the thread
			unsigned int nof_events; //number of events that the thread will update
			pthread_mutex_t *update_event_mutex;
			UpdateEventThreadParams(unsigned int i, HawkesProcess *h, unsigned int r, unsigned int s, unsigned int o, unsigned int n, pthread_mutex_t *mtx):
				thread_id(i),
				hp(h),
				run_id(r),
				seq_id(s),
				event_id_offset(o),
				nof_events(n),
				update_event_mutex(mtx){};
		};
		
		struct UpdateKernelThreadParams{
			unsigned int thread_id; //the id of the thread which executes the mcmc run
			PointProcess *hp;//calling object
			//unsigned int run_id; //the id of the thread which executes the mcmc run
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			unsigned int run_id; //id for the run of the mcmc
			unsigned int step_id;//id for the sample of the current mcmc run
			bool save;//save or not the posterior sample
			pthread_mutex_t *update_event_mutex;
			
			UpdateKernelThreadParams(unsigned int i, PointProcess *h, unsigned int o, unsigned int n, unsigned int r, unsigned int s, bool sv/*, GeneralizedHawkesProcess::State *g*/): thread_id(i), hp(h), type_id_offset(o), nof_types(n), run_id(r), step_id(s), save(sv)/*, ghs(g)*/{};
		};
		
		struct InitKernelThreadParams{
			unsigned int thread_id; //the id of the thread which executes the mcmc run
			HawkesProcess *hp;//calling object
			unsigned int run_id; //the mcmc run currently executed by the thread
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update	
			InitKernelThreadParams(unsigned int i, HawkesProcess *h, unsigned int r, unsigned int o, unsigned int n): thread_id(i), hp(h), run_id(r), type_id_offset(o), nof_types(n){};
		};
		
		//-----------------------------------------------------------------------------------  the posterior samples for the model parameters  ------------------------------------------------------------------------------------------------------------
		//each dimension means: type k-type k2-kernel parameter-mcmc run-samples
		std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> phi_posterior_samples;//TODO: if you put prior on the hyperparameters define a similar vector
		
		//each dimension means: type k-kernel parameter-mcmc run-samples
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> mu_posterior_samples;

		
		//-----------------------------------------------------------------------------------  variables for keeping the inference time  ------------------------------------------------------------------------------------------------------------
		
		std::vector<std::vector<double>>  profiling_mcmc_step;//inference time for the full mcmc steps up to a iteration, first dimension refers to the run of the thread, second to the mcmc step
		
		std::vector<std::vector<double>>  profiling_parent_sampling;//inference time for updating the parent of the events
		
		std::vector<std::vector<double>>  profiling_intensity_sampling;//inference time for updating the kernel parameters for the background intensities
		
				

		/****************************************************************************************************************************************************************************
		 * Model Serialization Methods
		******************************************************************************************************************************************************************************/
		friend class boost::serialization::access;

		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
		
		friend class Kernel;
		
		friend class ExponentialKernel;
		
		friend class LogisticKernel;

		friend class ConstantKernel;
		
		friend class PowerLawKernel;
		
		friend class RayleighKernel;

		friend class GeneralizedHawkesProcess;
		
	protected:
		/****************************************************************************************************************************************************************************
		 * Model protected Inference Methods
		******************************************************************************************************************************************************************************/
		
		void fit_(HawkesProcessMCMCParams const * const mcmc_params);
		
		static  void* fit_( void *p); //it executes a batch of mcmc runs, needed for multithreaded programming
				
		void mcmcInit(MCMCParams const * const mcmc_params) override;
		
		void mcmcSynchronize(unsigned int fit_nof_threads);
		
		void mcmcPolling(const std::string &dir_path_str, unsigned int fit_nof_threads);
		
		void mcmcStartRun(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads) override;

		void mcmcStep(unsigned int thread_id, unsigned int run, unsigned int step, bool save, unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *) , bool profiling=false) override;//save=true if the posterior samples should be saved, iteration is not in the burnin
		
		void mcmcUpdateExcitationKernelParams(unsigned int thread_id, unsigned int run,unsigned int step, bool save , unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *) , bool profiling=false);
		
		void mcmcUpdateParents(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads, bool profiling=false,unsigned int step_id=0);
		
		static void* mcmcUpdateParents_(void *p);//it updates the parent for a batch of events, needed for multithreaded programming
		
		static void* mcmcUpdateExcitationKernelParams_(void *p);//it updates the kernel parameters for a batch of event types, needed for multithreaded programming
		
		static void* mcmcStartRun_(void *p);//it initializes the kernel parameters from the prior for a batch of event types, needed for multithreaded programming
		
		static void computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, std::vector<std::vector<std::vector<double>>> & mean_phi_param, 
										   std::vector<std::vector<double>> & mode_mu_param, std::vector<std::vector<std::vector<double>>> & mode_phi_param,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples, 
										   const std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> & phi_posterior_samples);
		
		void computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, std::vector<std::vector<std::vector<double>>> & mean_phi_param, 
										   std::vector<std::vector<double>> & mode_mu_param, std::vector<std::vector<std::vector<double>>> & mode_phi_param);
		
		static void computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, 
										   std::vector<std::vector<double>> & mode_mu_param,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples);
		
		void computePosteriorParams(std::vector<std::vector<double>> & mean_mu_param, std::vector<std::vector<double>> & mode_mu_param);

		static void setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
				                       std::vector<std::vector<Kernel *>> & post_mean_phi, 
									   std::vector<ConstantKernel*> & post_mode_mu, 
									   std::vector<std::vector<Kernel *>> & post_mode_phi,
									   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples,  
									   const std::vector<std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>>> & phi_posterior_samples
		                               ); 
		
		static void setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
									   std::vector<ConstantKernel*> & post_mode_mu, 
									   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples
		                               ); 


		void static setPosteriorParents(EventSequence & seq, const std::vector<ConstantKernel *> & post_mode_mu,   const std::vector<std::vector<Kernel *>> & post_mode_phi);

		/****************************************************************************************************************************************************************************
		 * Model private Likelihood Methods
		******************************************************************************************************************************************************************************/
		// ********* full likelihood methods (with parent structure known) ********* //

		//full likelihood (of both the background intensity and the mutual excitation part)
		static double fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);
		
		//likelihood (of only the background intensity part)
		static double fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized=true, double t0=-1);

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		static double fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);

		//log-likelihood (of only the background excitation part)
		static double fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, bool normalized=true, double t0=-1);

		// ********* partial likelihood methods (with parent structure unknown) ********* //

		static double partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);

		//likelihood (of only the background intensity part)
		static double partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized=true, double t0=-1);

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		static double partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);
		
		//log-likelihood (of only the background excitation part)
		static double partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, bool normalized=true, double t0=-1);
		

		// ********* likelihood methods (it choses the full or patial likelihood according to the flag in seq) ********* //
		static double likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);

		//likelihood (of only the background intensity part)
		static double likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, bool normalized=true, double t0=-1);

		//log-likelihood (of only the mutual excitation part, it assumes known parent structure)
		static double likelihood(const EventSequence & seq, const Kernel & phi, int k, int k2, double t0=-1);

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		static double loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, bool normalized=true, double t0=-1);

		//log-likelihood (of only the mutual excitation part, it assumes known parent structure)
		static double loglikelihood(const EventSequence & seq, const Kernel & phi, int k, int k2, double t0=-1);

		//log-likelihood (of only the background excitation part)
		static double loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, bool normalized=true, double t0=-1);

		
		/****************************************************************************************************************************************************************************
		 * Model private Intensity Methods
		******************************************************************************************************************************************************************************/
		void plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=true, bool write_csv_file=false) const;
		
		/****************************************************************************************************************************************************************************
		 * Model private mcmc plot methods
		******************************************************************************************************************************************************************************/
		void generateMCMCplots_(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool write_png_file=true, bool write_csv_file=false);
		
		void generateMCMCTracePlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
		
		void generateMCMCMeanPlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
		
		void generateMCMCAutocorrelationPlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
		
		void generateMCMCPosteriorPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const;
		
		void generateMCMCIntensityPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const;

		void generateMCMCLikelihoodPlot(const std::string & dir_path_str, unsigned int samples_step, bool true_values,  const EventSequence & seq, bool write_png_file=true, bool write_csv_file=false, bool normalized=true) const;
		
		struct generateMCMCLikelihoodPlotThreadParams{
			const HawkesProcess *hp;//calling object
			const EventSequence & seq;
			//unsigned int seq_id;
			unsigned int *nof_samples;
			bool *endof_samples;
			unsigned int samples_step;
			std::vector<double> & post_mean_loglikelihood;
			std::vector<double> & post_mode_loglikelihood;
			pthread_mutex_t *mtx;
			bool normalized;
			generateMCMCLikelihoodPlotThreadParams(const HawkesProcess *p, const EventSequence & s, unsigned int *n, bool *e, unsigned int ss, std::vector<double> &mel, std::vector<double> &mol, pthread_mutex_t *m, bool nr):
				hp(p), 
				seq(s),
				nof_samples(n),
				endof_samples(e),
				samples_step(ss),
				post_mean_loglikelihood(mel),
				post_mode_loglikelihood(mol),
				mtx(m),
				normalized(nr)
			{};
		};
		
		static void *generateMCMCLikelihoodPlots__(void *p);
			
};  
#endif /* HAWKESPROCESS_HPP_ */
