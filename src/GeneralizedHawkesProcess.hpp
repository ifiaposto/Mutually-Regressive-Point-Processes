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

#ifndef GENERALIZEDHAWKESPROCESS_HPP_
#define GENERALIZEDHAWKESPROCESS_HPP_

#include "HawkesProcess.hpp"
#include "polyagammaSampler/PolyaGamma.h"


class Kernel;
class ConstantKernel;
class LogisticKernel;
struct SparseNormalGammaParameters;

using namespace boost::accumulators;
using namespace boost::numeric::operators;

class GeneralizedHawkesProcess:public virtual HawkesProcess{
	public:
			
	    //kernels used for the thinning probability per type of events
		//TODO: make it more general: SigmoidKernel and LogisticKernel (linear wrt the weights) subclass of it
		std::vector<LogisticKernel *> pt;//
		
		std::vector<Kernel *> psi; //history kernel functions
		
		std::vector<std::vector<SparseNormalGammaParameters *>> pt_hp;//hyperparameters for the logistic kernel, the first dimension refers to effects from, the second to effects on a certain type
		
		//-------------------------------------------------------------------------------------- the posterior point estimates   ---------------------------------------------------------------------------------------------------------------------------
		//the posterior mean point estimates for the  logistic kernel parameters		
		std::vector<LogisticKernel *> post_mean_pt;
		
		//the posterior mode point estimates for the kernel parameters
		std::vector<LogisticKernel *> post_mode_pt;
		
		//-------------------------------------------------------------------------------------- the posterior point estimates   ---------------------------------------------------------------------------------------------------------------------------
		//the posterior mean point estimates for the  history kernel functions		
		std::vector<Kernel *> post_mean_psi;
		
		//the posterior mode point estimates for the history kernel functions
		std::vector<Kernel *> post_mode_psi;
				

	public:           

		/****************************************************************************************************************************************************************************
		 * Model Construction and Destruction Methods
		******************************************************************************************************************************************************************************/
		
		/***************************************************   anononymous point processes    *********************************************/
		GeneralizedHawkesProcess(); //empty process
		
		GeneralizedHawkesProcess(unsigned int k, double ts, double te); //empty multivariate process
		
		GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m); //multivariate mutually independent poisson process
		
		GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p);//hawkes process
		
		GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps);//multivariate multually dependent poisson process
		
		GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps);//generalized hawkes process
		
		//hierarchical generalized hawkes process
		GeneralizedHawkesProcess(unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps, std::vector<std::vector<SparseNormalGammaParameters *>> hp);
		
		/***************************************************   named point processes    *********************************************/
		GeneralizedHawkesProcess(std::string name); //empty process
		
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te); //empty multivariate process
		
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m); //multivariate mutually independent poisson process
		
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p);//hawkes process
		
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps);//multivariate multually dependent poisson process
		
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps);//generalized hawkes process

		//hierarchical generalized hawkes process
		GeneralizedHawkesProcess(std::string name, unsigned int k, double ts, double te, std::vector<ConstantKernel *> m, std::vector<std::vector<Kernel *>> p, std::vector<LogisticKernel *> h, std::vector<Kernel *> ps, std::vector<std::vector<SparseNormalGammaParameters *>> hp);
				
		GeneralizedHawkesProcess(std::ifstream &file);  //import model from file

		~GeneralizedHawkesProcess();
		
		/****************************************************************************************************************************************************************************
		 * Model Utility Methods
		******************************************************************************************************************************************************************************/

		std::string createModelDirectory() const override;
		
		std::string createModelInferenceDirectory() const override;
		
		void print(std::ostream &file) const override;

		/****************************************************************************************************************************************************************************
		 * Model Likelihood  Methods
		******************************************************************************************************************************************************************************/
		//these are joint likelihood methods for the observed and latent variables of the model
		double likelihood(const EventSequence & seq, bool normalized=true, double t0=-1) const override;

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		double loglikelihood(const EventSequence & seq, bool normalized=true, double t0=-1) const override;
		
	
		void posteriorModeLoglikelihood(EventSequence & seq, std::vector<double> &logl, bool normalized=true, double t0=-1) override;//todo: why not const seq????
		/****************************************************************************************************************************************************************************
		 *Generative Model Methods
		******************************************************************************************************************************************************************************/
		
		void generate() override;
		
		//update the normal prior of the inhibition weights from  the excitation coefficients
		//void updateNormalGammas();
		void setSparseNormalGammas();


		//update the normal prior of the inhibitiob weights from  the excitation coefficients
		//static void updateNormalGamma(const std::vector<Kernel *> & phi, const std::vector<double> & kappa, LogisticKernel *pt);
		static void setSparseNormalGamma(const std::vector<Kernel *> & phi, const std::vector<SparseNormalGammaParameters *> & hp, LogisticKernel *pt);
		
		/****************************************************************************************************************************************************************************
		 *  Model Simulation Methods
		******************************************************************************************************************************************************************************/
		using HawkesProcess::simulate;
		
		EventSequence simulate(unsigned int id=0) override;
	
		void simulate(EventSequence & seq, double dt) override;
		
		void simulateNxt(const EventSequence & seq, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N) override;
		
		//TODO: add const qualifiers here
		static EventSequence simulate(std::vector<ConstantKernel *>  & mu, std::vector < std::vector<Kernel *>> & phi, std::vector<LogisticKernel *> & pt, double start_t, double end_t, std::string name="", unsigned int id=0);
		
		static void simulate(std::vector<ConstantKernel *>  & mu, std::vector < std::vector<Kernel *>> & phi, std::vector<LogisticKernel *> & pt, EventSequence & seq, double dt);
		
		static void simulateNxt(const std::vector< ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, const EventSequence &s, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N=1); //augment event sequence seq with the next N events which occur at most after dt time units generated by the intensities mu and phi


		/****************************************************************************************************************************************************************************
		 * Model Memory Methods
		******************************************************************************************************************************************************************************/
	    void save(std::ofstream &file) const override; 
	    
	    void load(std::ifstream &file) override; 
	    
	    void saveParameters(std::ofstream &file)  const override;
	

		/****************************************************************************************************************************************************************************
		 * Model Intensity Methods
		******************************************************************************************************************************************************************************/
		


		static void printMatlabFunctionIntensities(const std::string & dir_path_str, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt,  const std::vector<Kernel *> & psi, const EventSequence &s) ;

		//print intensity function of type k of the event sequence s as matlab functions
	
		static void printMatlabFunctionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda);

		static void printMatlabFunctionIntensity(const ConstantKernel * mu,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda);
		
		static void printMatlabExpressionIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda);

		static void printMatlabExpressionIntensity(const ConstantKernel * mu,const LogisticKernel *pt, const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda);

		static void printMatlabExpressionThining(const LogisticKernel *pt,  const std::vector<Kernel *> & psi, const EventSequence &s, std::string & matlab_lambda);

	    //compute intensity at equally spaced "nofps" timepoints within the observation window
		void computeIntensity(unsigned int k, const EventSequence &s, boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v, unsigned int nofps=NOF_PLOT_POINTS) const override;
		
		//compute intensity at determined timepoints t
		void computeIntensity(unsigned int k, const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v) const override;

		static void computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel *pt, const EventSequence &s, boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v, unsigned int nofps=NOF_PLOT_POINTS);
		
		static void computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel *pt, const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v);

		static double computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, const EventSequence &s, double t);
		
		static double computeIntensity(const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, const EventSequence &s, double th, double t);

		void plotIntensity(const std::string &filename, unsigned int k, const EventSequence &s, unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=true, bool write_csv_file=false) const override;
		
		void plotIntensities(const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS,bool write_png_file=true, bool write_csv_file=false) const override ;

		void plotIntensities(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS,bool write_png_file=true, bool write_csv_file=false) const override;
		
		// static plot methods
		static void plotIntensities(const std::string & dir_path_str, const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, unsigned int nofps=NOF_PLOT_POINTS,bool write_png_file=true, bool write_csv_file=false) ;
		
		static void plotIntensities(const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, unsigned int nofps=NOF_PLOT_POINTS,bool write_png_file=true, bool write_csv_file=false) ;
			
		static void plotIntensity(const std::string &filename, const EventSequence &s,  const ConstantKernel * mu,const std::vector<Kernel *> & phi, const LogisticKernel* pt, unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=true, bool write_csv_file=false);

		/****************************************************************************************************************************************************************************
		 * Model Inference Methods
		******************************************************************************************************************************************************************************/
		void fit(MCMCParams const * const p) override;
		
		//void fit(const std::string &dir_path_str,unsigned int runs=MCMC_RUNS, unsigned int nof_sequences=1, unsigned int burnin_iter=MCMC_BURNIN_NUM_ITER, unsigned int max_num_iter=MCMC_MAX_NUM_ITER, unsigned int fit_nof_threads=MCMC_FIT_NOF_THREADS, unsigned int iter_nof_threads=MCMC_ITER_NOF_THREADS, bool profiling=false) override;

		void initProfiling(unsigned int runs,  unsigned int max_num_iter)  override;
		
		//it prints the inference time in  csv files for all the steps of the inference
		void writeProfiling(const std::string &dir_path_str) const override;

		//it sets the posterior mean and mode parameters to the kernels
		void setPosteriorParams() override;
		
		//it discards the first samples of the mcmc inference, the burnin samples may have to be preserved for plotting purposes
		void flushBurninSamples(unsigned int nof_burnin=NOF_BURNIN_ITERS) override;
		
		/****************************************************************************************************************************************************************************
		 * Model Goodness of fit methods
		******************************************************************************************************************************************************************************/
		static void goodnessOfFitMatlabScript(const std::string & dir_path_str, const std::string & seq_path_str, std::string model_name, std::vector<EventSequence> & data, const std::vector<ConstantKernel *> mu, const std::vector<std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> pt, const std::vector<Kernel *> & psi) ;
		
		/****************************************************************************************************************************************************************************
		 * Model Testing Methods //TODO: prediction tasks incomplete
		******************************************************************************************************************************************************************************/
	
		void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences,  double t0, double t1, unsigned int burnin_samples=0,  const std::string & true_logl_summary=0, bool normalized=true)  override;
		
		void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples=0,  const std::string & true_logl_summary=0, bool normalized=true)  override;
		
		void testPredictions(EventSequence & seq, const std::string &test_dir_path_str, double & mode_rmse, double & mean_rmse, double & mode_errorrate, double & mean_errorrate, double t0=-1) override;
				

		/****************************************************************************************************************************************************************************
		 * Model Prediction Methods //TODO: prediction tasks incomplete
		******************************************************************************************************************************************************************************/
	
		struct predictNextArrivalTimeThreadParams{
			unsigned int thread_id;
			unsigned int nof_samples;
			double & mean_arrival_time;
			const std::vector<ConstantKernel *>  & mu;
			const std::vector<std::vector<Kernel*>> & phi;
			const std::vector<LogisticKernel*> & pt;
			const EventSequence & seq;
			double t_n;
			
			predictNextArrivalTimeThreadParams(unsigned int i, unsigned int n, double &pm, const std::vector<ConstantKernel *>  & m, const std::vector<std::vector<Kernel*>> &p,  const std::vector<LogisticKernel*> &p2, const EventSequence &s, double t): 
				thread_id(i), 
				nof_samples(n), 
				mean_arrival_time(pm), 
				mu(m), 
				phi(p), 
				pt(p2),
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
			const std::vector<LogisticKernel*> & pt;
			const EventSequence & seq;
			double t_n;
			
			predictNextTypeThreadParams(unsigned int i, unsigned int n, std::map<int, unsigned int> & tc, const std::vector<ConstantKernel *>  & m, const std::vector<std::vector<Kernel*>> &p,  const std::vector<LogisticKernel*> &p2, const EventSequence &s, double t): 
				thread_id(i), 
				nof_samples(n), 
				type_counts(tc), 
				mu(m), 
				phi(p), 
				pt(p2),
				seq(s),
				t_n(t){};
		};

		
		//it computes the probability that the next arrival after tn will happen at t
		static double computeNextArrivalTimeProbability(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi,  const std::vector<LogisticKernel*> & pt, double tn, double t);
		
		static double predictNextArrivalTime(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES, unsigned int nof_threads=MODEL_SIMULATION_THREADS);
		
		static void * predictNextArrivalTime_(void *p);
		
		static int predictNextType(const EventSequence & seq, double t_n, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, std::map<int, double> & type_prob, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES, unsigned int nof_threads=MODEL_SIMULATION_THREADS);
		
		static void * predictNextType_(void *p);
		
		void predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, double t0=-1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) override ;
	
		static void predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, double *rmse=0, double *error_rate=0, double t0=-1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES);
		
		/****************************************************************************************************************************************************************************
		 * Model Plot Methods
		******************************************************************************************************************************************************************************/
		void generateMCMCplots(unsigned int samples_step=SAMPLES_STEP, bool true_values=false, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool png_file=true, bool csv_file=false) override;
		
		void generateMCMCplots(const std::string & dir_path_str, unsigned int samples_step=SAMPLES_STEP, bool true_values=false, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool png_file=true, bool csv_file=false) override;
		
		void generateMCMCTracePlots(const std::string & dir_path_str, bool write_png_file=true , bool write_csv_file=false) const override;
		
		void generateMCMCTracePlots(bool write_png_file=true , bool write_csv_file=false)  const override;
		
		void generateMCMCMeanPlots(const std::string & dir_path_str, bool write_png_file=true , bool write_csv_file=false) const override;
		
		void generateMCMCMeanPlots(bool write_png_file=true , bool write_csv_file=false) const override;
		
		void generateMCMCAutocorrelationPlots(const std::string & dir_path_str,bool write_png_file=true , bool write_csv_file=false) const override;
		
		void generateMCMCAutocorrelationPlots(bool write_png_file=true , bool write_csv_file=false) const override;
		
		void generateMCMCPosteriorPlots(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const override;
		
		void generateMCMCPosteriorPlots(bool true_values, bool write_png_file=true, bool write_csv_file=false) const override;
		
		void generateMCMCIntensityPlots(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const override;
		
		void generateMCMCIntensityPlots(bool true_values, bool write_png_file=true, bool write_csv_file=false) const override;
		
		void generateMCMCTrainLikelihoodPlots(unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true, const std::string & plot_dir_path_str="") const override;
		
		void generateMCMCTestLikelihoodPlots(const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true,  double t0=-1, double t1=-1, const std::string & plot_dir_path_str="") const override;
	private:
		
		friend class Kernel;
		
		friend class ExponentialKernel;
		
		friend class LogisticKernel;

		friend class ConstantKernel;
		
		friend class PowerLawKernel;
		
		friend class RayleighKernel;

		/****************************************************************************************************************************************************************************
		 * Model Serialization Methods
		******************************************************************************************************************************************************************************/
		friend class boost::serialization::access;

		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
		
		/****************************************************************************************************************************************************************************
		 * Model Specific Methods
		******************************************************************************************************************************************************************************/
		
		//it computes the probability that an event of type k at t given the events in sequence s will be observed/realized.
		double computeRealizationProbability(const EventSequence &s, unsigned int k, double t);
		
		//it computes the probability that an event of type k at t given the events in sequence s will be observed/realized.
		static double computeRealizationProbability(const EventSequence &s,  const LogisticKernel * pt, double t);
		
		//it computes the probability that an event of type k at t given the events in sequence s will be observed/realized.
		static double computeRealizationProbability(const EventSequence &s,  const LogisticKernel * pt, double th, double t);
		
		//compute the history effect of the event sequence hs on the sequence s (both of them separated by type), store in h: <type><event><history kernel function>
		void computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, std::vector<std::map<double,Event *>> & s, double ***&h, unsigned int nof_threads, int thread_id=-1, int run_id=-1);
		
		//compute the history effect of the event sequence hs on the sequence s of events of type k, store in h: <event><history kernel function>
		void computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, std::map<double,Event *> & s, unsigned int k, double **&h, unsigned int nof_threads,  int thread_id=-1, int run_id=-1);
		
		//compute history effect of the event sequence seq for the time t, each dimension of h refers to the history effect that comes from a  thinning kernel function of the kernel pt
		static void computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, const LogisticKernel *pt, double t, double *&h);
		
		//compute history effect of the event sequence seq for the time t, each dimension of h refers to the history effect that comes from a  thinning kernel function of the kernel pt
		static void computeKernelHistory(const std::vector<std::map<double,Event *>> & hs, const LogisticKernel *pt, double th, double t, double *&h);
		
		//it fills the seq with the thinned events according to mu, phi, pt
		static void sampleThinnedEvents(EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt);
				
		
		/****************************************************************************************************************************************************************************
		 * Model private Intensity Methods
		******************************************************************************************************************************************************************************/
		void plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps=NOF_PLOT_POINTS, bool png_file=true, bool csv_file=false) const;
		
		static void plotIntensities_(const std::string & dir_path_str, const EventSequence &s,  const std::vector<ConstantKernel *>  & mu, const std::vector < std::vector<Kernel *>> & phi, const std::vector<LogisticKernel *> & pt, unsigned int nofps=NOF_PLOT_POINTS, bool png_file=true, bool csv_file=false);
		
		/****************************************************************************************************************************************************************************
		 * Model Likelihood  Methods
		******************************************************************************************************************************************************************************/
		//the likelihoods below do not include the priors of the parameters

		//likelihood of generalized hawkes process
		static double likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//likelihood of mutlivariate mutually regressive poisson process
		static double likelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//likelihood of only the thinning procedure of the process
		static double likelihood(const EventSequence & seq, const std::vector<LogisticKernel*> & pt, double t0=-1);
		//likelihood of only the thinning procedure of the process for events of type k
		static double likelihood(const EventSequence & seq, unsigned int k, LogisticKernel const * const pt, double t0=-1);
		//augmented by the polyagammas likelihood of the thinning procedure of the process for events of type k
		static double likelihood(const EventSequence & seq, unsigned int k, LogisticKernel const * const pt, const std::vector<double> & polyagammas, const std::vector<double> & thinned_polyagammas, double t0=-1);


		//loglikelihood of generalized hawkes process
		//they assume full seq with thinned events included
		static double loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//loglikelihood of mutlivariate mutually regressive poisson process
		static double loglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//loglikelihood of only the thinning procedure of the process
		static double loglikelihood(const EventSequence & seq, const std::vector<LogisticKernel*> & pt, double t0=-1);

		
		// ********* full likelihood methods (with parent structure and thinned events known) ********* //
		static double fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//likelihood (of only the background intensity part)
		static double fullLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//full log-likelihood (of both the background intensity and the mutual excitation part)
		static double fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi,  const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//log-likelihood (of only the background excitation part)
		static double fullLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);


		// ********* partial likelihood methods (with parent structure and thinned events unknown) ********* //
		static double partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu, const std::vector<std::vector<Kernel*>> & phi, const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//likelihood (of only the background intensity part)
		static double partialLikelihood(const EventSequence & seq, const std::vector<ConstantKernel *> & mu,  const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//full log-likelihood (of both the background intensity and the mutual excitation part)
		static double partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu, const std::vector<std::vector<Kernel*>> & phi,  const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		//log-likelihood (of only the background excitation part)
		static double partialLoglikelihood(const EventSequence & seq, const std::vector<ConstantKernel *>  & mu,  const std::vector<LogisticKernel*> & pt, bool normalized=true, double t0=-1);
		

		// ********* collapsed likelihood methods (with the weights in the logistic kernel collapsed), the return value is proportionsl
		// to the likelihood and useful only for computation of likelihood ratios ********* //
		//static double collapsedLikelihood(const EventSequence & seq, unsigned int k, LogisticKernel * pt, const std::vector<double> & polyagammas, const std::vector<double> & thinned_polyagammas, double **h, double **thinned_h);

		/****************************************************************************************************************************************************************************
			 * Structures and variables used for the inference
		******************************************************************************************************************************************************************************/
		
		struct State{
			int K;
			
			std::vector<LogisticKernel *> pt;//the sample for the kernels in the inhibition part
			
			std::vector<Kernel *> psi; //the sample for the history kernels in the inhibition part
			
			std::vector<std::vector<SparseNormalGammaParameters *>>  pt_hp;//sparse normal gamma parameters 
			
			//the latent polya-gamma variables for the observed separated by type
			//the first dimension refers to the id of the event sequence, the second dimension refers to the type of the event, the third to the id of the event within the type 
			std::vector<std::vector<std::vector<double>>> polyagammas; //the latent polya-gamma variables for the observed  (virtual event is excluded from this structure) separated by type
			std::vector<std::vector<std::vector<double>>> thinned_polyagammas; //the latent polya-gamma variables for the thinned  separated by type
			
			//the  first dimension refers to the sequence id, the second dimension refers to the type of events, the third dimension refers to the id of the event, the forth to the the sum of history passed from a kernel function
			//kernel history of observed events
			double ****h;
			//the kernel history of the thinned events separated by type
			double **** thinned_h;
			unsigned int mcmc_iter=0;//number of previous mcmc samples
			
			GeneralizedHawkesProcessMCMCParams const * mcmc_params;
			
			private:
			
		        friend class boost::serialization::access;
		        friend class GeneralizedHawkesProcess;
		        
		        void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		        void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		        void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		        void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);
		        void print (std::ostream &file) const;
		 };
		
		//the current state of the mcmc runs
		std::vector<State> mcmc_state; 

		//the first dimension refers to the event sequence for the training,
		//the second to the type of the event,
		//the third to the id of the event,
		//the forth to the the sum of history passed from a kernel function

		//-----------------------------------------------------------------------------------  the posterior samples for the model parameters  ------------------------------------------------------------------------------------------------------------
		//each dimension means: type k-kernel parameter-mcmc run-samples
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> pt_posterior_samples;

		//each dimension means: 
		// history dimension - kernel parameter - mcmc run - samples
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> psi_posterior_samples;
		// type 1 - type 2 - mcmc run - samples
		std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> tau_posterior_samples;
		
		
		//-----------------------------------------------------------------------------------  variables for keeping the inference time  ------------------------------------------------------------------------------------------------------------
		
		//profiling vectors: inference time for each part of the learning
		std::vector<std::vector<double>>  profiling_mcmc_step;//inference time for the full mcmc steps up to a iteration, first dimension refers to the run of the thread, second to the mcmc step
		std::vector<std::vector<double>>  profiling_compute_history;//inference time for computing the history effect of past observed events on thinned events
		std::vector<std::vector<double>>  profiling_thinned_events_sampling;//inference time for the sampling of the thinned events
		std::vector<std::vector<double>>  profiling_polyagammas_sampling;//inference time for the sampling of the polyagammas events
		std::vector<std::vector<double>>  profiling_logistic_sampling;//inference time for sampling the logistic weights
		std::vector<std::vector<double>>  profiling_hawkes_sampling;//inference time for sampling the exciting part
		std::vector<std::vector<double>>  profiling_normal_model_sampling;//inference time for sampling the normal model for the inhibition weights
		
		/****************************************************************************************************************************************************************************
			 * Structures needed for multithreaded programming
		******************************************************************************************************************************************************************************/
		
		//struct which holds the parameters for the thread which is responsible for running a mcmc run
		struct InferenceThreadParams{
			unsigned int thread_id; //the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int run_id_offset; //id of the first mcmc run of the thread
			unsigned int runs; //number of runs that the thread will run
			unsigned int burnin_iter;//number of burnin samples in each run
			unsigned int max_num_iter;//number of iterations/ samples to get from each run
			unsigned int iter_nof_threads;//number of threads for each iteration of the mcmc
			bool profiling;//print inference time in files

			std::string infer_dir_path_str;
			unsigned int fit_nof_threads;
			unsigned int *  nof_fit_threads_done;
			pthread_mutex_t *fit_mtx;
			pthread_cond_t *fit_con;
			

			InferenceThreadParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int o, unsigned int r, unsigned int b, unsigned int m, unsigned int t, bool p, std::string id, unsigned int nt, unsigned int *d, pthread_mutex_t * fm, pthread_cond_t *fc):
				thread_id(i),
				ghp(pp),
				run_id_offset(o),
				runs(r),
				burnin_iter(b),
				max_num_iter(m),
				iter_nof_threads(t),
				profiling(p),
				infer_dir_path_str(id),
				fit_nof_threads(nt),
				nof_fit_threads_done(d),
				fit_mtx(fm),
				fit_con(fc)
				{};
		};
		
		struct SampleThinnedEventsParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int seq_id;
			unsigned run_id; //the id of the thread which executes the mcmc run
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			unsigned int nof_threads; 
			pthread_mutex_t *sample_thinned_events_mutex;
			SampleThinnedEventsParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int s, unsigned int r, unsigned int o, unsigned int n, unsigned int tn, pthread_mutex_t *mtx): 
				thread_id(i), 
				ghp(pp), 
				seq_id(s),
				run_id(r), 
				type_id_offset(o), 
				nof_types(n), 
				nof_threads(tn), 
				sample_thinned_events_mutex(mtx) {};
		};
		
		struct SampleThinnedEventsParams_{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int seq_id;
			unsigned int run_id;//the id of the thread which executes the mcmc run
			int k;//type of the thinned events that will be generated
			unsigned int nof_events;//nof events/non-homogeneous poisson processes for type k that will be sampled 
			unsigned int event_id_offset;//id of the first observed event.non-homogeneous poisson process
			pthread_mutex_t *sample_thinned_events_mutex;
			SampleThinnedEventsParams_(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int s, unsigned int r, unsigned int t, unsigned int n, unsigned int o, pthread_mutex_t *mtx): 
				thread_id(i), 
				ghp(pp), 
				seq_id(s),
				run_id(r), 
				k(t), 
				nof_events(n), 
				event_id_offset(o), 
				sample_thinned_events_mutex(mtx) {};
		};
		
		
		struct SamplePolyagammasParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int seq_id;
			unsigned int run_id; //the id of the thread which executes the mcmc run
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			unsigned int nof_threads;
			SamplePolyagammasParams(unsigned int i, GeneralizedHawkesProcess *pp/*, const std::vector<std::vector<double>> & h2*/, unsigned int s, unsigned int r, unsigned int o, unsigned int n, unsigned int tn):
				thread_id(i),
				ghp(pp),
				seq_id(s),
				run_id(r),
				type_id_offset(o),
				nof_types(n),
				nof_threads(tn) {};
		};
		
		struct SamplePolyagammasParams_{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int seq_id;
			unsigned int run_id; //the id of the thread which executes the mcmc run
			unsigned int k; //type of events whose polyagamma latent variable is sampled
			unsigned int event_id_offset; //id of the first type params of the thread
			unsigned int nof_events; //number of events of typek k whose polyagamma will be sampled
			unsigned int N;
			unsigned int Nt;
			SamplePolyagammasParams_(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int s, unsigned int r, unsigned int k2,unsigned int o, unsigned int n, unsigned int N2, unsigned int Nt2): 
				thread_id(i), 
				ghp(pp), 
				seq_id(s),
				run_id(r), 
				k(k2), 
				event_id_offset(o), 
				nof_events(n), 
				N(N2), 
				Nt(Nt2) {};
		};
		
		struct UpdateThinKernelThreadParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			unsigned int run_id; //id for the run of the mcmc
			unsigned int step_id;//id for the sample of the current mcmc run
			unsigned int nof_threads;
			bool save;//save or not the posterior sample
			
			UpdateThinKernelThreadParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int o, unsigned int n, unsigned int r, unsigned int s, unsigned int ts, bool sv): thread_id(i), ghp(pp), type_id_offset(o), nof_types(n), run_id(r), step_id(s),nof_threads(ts), save(sv){};
		};
		
		struct InitThinKernelThreadParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int run_id; //the id of the thread which executes the mcmc run
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			InitThinKernelThreadParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int r, unsigned int o, unsigned int n): thread_id(i), ghp(pp), run_id(r), type_id_offset(o), nof_types(n){};
		};
		
		struct SampleInhibitionPartThreadParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int run; //run of the mcmc 
			unsigned int step; //step in the run of the mcmc
			bool save; //whether the samples of the weights for the inhibition part will be saved
			unsigned int iter_nof_threads; //nof threads for the current step/iteration
			bool profiling; //whether it will report inference time in a file
			SampleInhibitionPartThreadParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int r, unsigned int s, bool sv, unsigned int n, bool p): 
				thread_id(i), 
				ghp(pp), 
				run(r), 
				step(s), 
				save(sv), 
				iter_nof_threads(n), 
				profiling(p){};
		};
		
		struct SampleExcitationPartThreadParams{
			unsigned int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			unsigned int run;//run of the mcmc 
			unsigned int step;//step in the run of the mcmc
			bool save;//whether the samples of the weights for the inhibition part will be saved
			unsigned int iter_nof_threads;//nof threads for the current step/iteration
			bool profiling;//whether it will report inference time in a file
			SampleExcitationPartThreadParams(unsigned int i, GeneralizedHawkesProcess *pp, unsigned int r, unsigned int s, bool sv, unsigned int n, bool p): 
				thread_id(i), 
				ghp(pp), 
				run(r), 
				step(s), 
				save(sv), 
				iter_nof_threads(n), 
				profiling(p){};
		};
		
		struct computeKernelHistoryThreadParams{//needed for parallelism across types
			int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			int run;
			unsigned int type_id_offset; //id of the first type params of the thread
			unsigned int nof_types; //number of types whose params the thread will update
			const std::vector<std::map<double,Event *>> & hs;
			std::vector<std::map<double,Event *>> & s;
			double *** &h;
			unsigned int nof_threads;
			computeKernelHistoryThreadParams(unsigned int i, GeneralizedHawkesProcess *p, int r, unsigned int o, unsigned int n, const std::vector<std::map<double,Event *>> & hs2, std::vector<std::map<double,Event *>> & s2, double *** &h2, unsigned int nt):
				thread_id(i),
				ghp(p),
				run(r),
				type_id_offset(o),
				nof_types(n),
				hs(hs2),
				s(s2),
				h(h2), 
				nof_threads(nt){};
		};
		
		struct computeKernelHistoryThreadParams_{//needed for parallelism across events
			int thread_id;//the id of the thread which executes the mcmc run
			GeneralizedHawkesProcess *ghp;//calling object
			int run;
			unsigned int event_id_offset;
			unsigned int nof_events;
			const std::vector<std::map<double,Event *>> & hs;
			const std::vector<Event *> & s;
			unsigned int k;
			double **&h;
			computeKernelHistoryThreadParams_(unsigned int i, GeneralizedHawkesProcess *p, int r, unsigned int o, unsigned int n, const std::vector<std::map<double,Event *>> & hs2, const std::vector<Event *> & s2, unsigned int k2, double ** &h2):
				thread_id(i),
				ghp(p),
				run(r),
				event_id_offset(o), 
				nof_events(n),
				hs(hs2),
				s(s2),
				k(k2),
				h(h2){};
		};

		//	--------------------------------------------------------------------------  synchronization variables  --------------------------------------------------------------------------
			//lock for modifying the event sequence by adding the sequence of the thinned events

			std::vector<pthread_mutex_t *> sample_thinned_events_mutex;
			unsigned int fit_nof_threads;
			unsigned int nof_fit_threads_done; //number of threads which have finished the inference
			pthread_mutex_t fit_mtx;
			pthread_cond_t fit_con;
			std::vector<pthread_mutex_t *> save_samples_mtx;


	protected:
		/****************************************************************************************************************************************************************************
		 * Model protected Inference Methods
		******************************************************************************************************************************************************************************/
		
		void fit_(GeneralizedHawkesProcessMCMCParams const * const mcmc_params);
	
		void mcmcInit(MCMCParams const * const mcmc_params) override;
		
		void mcmcSynchronize(unsigned int fit_nof_threads);
		
		void mcmcPolling(const std::string &dir_path_str, unsigned int fit_nof_threads);
		
		void mcmcStartRun(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads) override;
		
		void mcmcStep( unsigned int thread_id, unsigned int run,unsigned int step, bool save, unsigned int iter_nof_threads, void * (*mcmcUpdateExcitationKernelParams_)(void *)=0, bool profiling=false) override;
		
		void mcmcStepSampleThinnedEvents(unsigned int seq_id, unsigned int run,unsigned int step, unsigned int thread_id, unsigned int iter_nof_threads, bool profiling=false);
		
		void mcmcStepClear(unsigned int run_id); //clear the current mcmc state from the latent variables of the previous step
		
		void sampleThinnedEvents(unsigned int thread_id, unsigned int seq_id, unsigned int run, unsigned int iter_nof_threads);//multithreaded, it distributes per type the thinned events to be sampled
		
		void mcmcStepSamplePolyaGammas(unsigned int thread_id, unsigned seq_id, unsigned int run_id, unsigned int iter_nof_threads);//multithreaded, it distributes per type of events the polyagammas to be sampled
		
		void mcmcUpdateThinKernelParams(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads,  unsigned int step_id, bool save);
		
		void mcmcHistoryKernelsUpdate(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads,  unsigned int step_id, bool save);
		
		//initialize the sparse normal gammas of the logistic kernels of the model
		void mcmcStartSparseNormalGammas(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads);
		

		//inference methods needed for multithreaded programming
		static  void* fit_( void *p); //it executes a batch of mcmc runs, needed for multithreaded programming
		
		static void* mcmcStepSampleInhibitionPart_(void *p); //it samples the inhibition weights and the polyagammas needed for that
		
		static void* mcmcStepSampleExcitationPart_(void *p); //it samples the parameters of the intensity function of the mutually exciting poisson processes
		
		static void* mcmcUpdateExcitationKernelParams_(void *p);//it samples the excitatotion parameters, it overrides the method of the base for the case where the calling object is a generalized hawkes process

		static void* mcmcSamplePolyaGammas_( void *p);//it samples the latent polya-gamma variables for each event for a range of types, needed for multithreaded programming
		
		static void* mcmcSamplePolyaGammas__( void *p);//it samples the latent polya-gamma variables for a range of events of a certain type, needed for multithreaded programming
		
		static void* mcmcUpdateThinKernelParams_(void *p);

		//it initializes the thinning weights from the prior for a batch of event types, needed for multithreaded programming
		static void* mcmcStartRun_(void *p);
		
		static void* sampleThinnedEvents_(void *p); //distribute across the types of the thinned events that will be sampled
		
		static void *sampleThinnedEvents__(void *p); //distribute across the observed events/ parents of the thinned events
		
		static void *computeKernelHistory_(void *p); //it computes the history of events whose type lies within a range of types
		
		static void *computeKernelHistory__(void *p); //it computes the history of a set of events of a specific type
		
		/****************************************************************************************************************************************************************************
		 * Posterior Processing Methods
		******************************************************************************************************************************************************************************/
		
		//for setting both mutual excitation and inhibition
		static void setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
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
		);
		//for setting only inhibition and constant excitation
		static void setPosteriorParams(std::vector<ConstantKernel*> & post_mean_mu, 
				   	   	   	   	   std::vector<LogisticKernel *> & post_mean_pt,
								   std::vector<Kernel *> & post_mean_psi,
								   std::vector<ConstantKernel*> & post_mode_mu,
								   std::vector<LogisticKernel *> & post_mode_pt,
								   std::vector<Kernel *> & post_mode_psi,
								   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & mu_samples,       
								   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_samples,
								   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_samples
		); 
		
		//for seting only inhibition
		static void setPosteriorParams(
				                       std::vector<LogisticKernel *> & post_mean_pt, 
									   std::vector<Kernel *> & post_mean_psi,
									   std::vector<LogisticKernel *> & post_mode_pt, 
									   std::vector<Kernel *> & post_mode_psi,
									   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_posterior_samples,
									   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_posterior_samples
									);

		
		//for computing only inhibition
		void computePosteriorParams(std::vector<std::vector<double>> & mean_pt_param, std::vector<std::vector<double>> & mode_pt_param, std::vector<std::vector<double>> & mean_psi_param, std::vector<std::vector<double>> & mode_psi_param);
		
		static void computePosteriorParams(std::vector<std::vector<double>> & mean_pt_param, 
										   std::vector<std::vector<double>> & mode_pt_param, 
										   std::vector<std::vector<double>> & mean_psi_param,
										   std::vector<std::vector<double>> & mode_psi_param,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & pt_posterior_samples,
										   const std::vector<std::vector<std::vector<boost::numeric::ublas::vector<double>>>> & psi_posterior_samples
										  );
		
				
	private:
			
		/****************************************************************************************************************************************************************************
		 * Model private mcmc plot methods
		******************************************************************************************************************************************************************************/
		void plotPosteriorIntensity(const std::string & dir_path_str, unsigned int k, const EventSequence &s,  bool true_intensity, unsigned int sequence_id=0, unsigned int nofps=NOF_PLOT_POINTS, bool write_png_file=true, bool write_csv_file=false) const;
		
		void generateMCMCplots_(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter=NOF_BURNIN_ITERS, bool write_png_file=true, bool write_csv_file=false);
			
		void generateMCMCTracePlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
			
		void generateMCMCMeanPlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
			
		void generateMCMCAutocorrelationPlots_(const std::string & dir_path_str, bool write_png_file=true, bool write_csv_file=false) const;
			
		void generateMCMCPosteriorPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const;
			
		void generateMCMCIntensityPlots_(const std::string & dir_path_str, bool true_values, bool write_png_file=true, bool write_csv_file=false) const;

		void generateMCMCLikelihoodPlot(const std::string & dir_path_str, unsigned int samples_step, bool true_values,  const EventSequence & seq, bool write_png_file=true, bool write_csv_file=false, bool normalized=true) const;
		
		struct generateMCMCLikelihoodPlotThreadParams{
			const GeneralizedHawkesProcess *ghp;//calling object
			const EventSequence & seq;
			unsigned int *nof_samples;
			bool *endof_samples;
			unsigned int samples_step;
			std::vector<double> & post_mean_loglikelihood;
			std::vector<double> & post_mode_loglikelihood;
			pthread_mutex_t *mtx;
			bool normalized;
			generateMCMCLikelihoodPlotThreadParams(const GeneralizedHawkesProcess *p, const EventSequence & s, unsigned int *n, bool *e, unsigned int ss, std::vector<double> &mel, std::vector<double> &mol, pthread_mutex_t *m, bool nr):
				ghp(p), 
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

#endif /* GENERALIZEDHAWKESPROCESS_HPP_ */
