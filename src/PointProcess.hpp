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

#ifndef POINTPROCESS_HPP_
#define POINTPROCESS_HPP_

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
#include <boost/chrono/include.hpp>
#include <boost/timer/timer.hpp>
#include <boost/accumulators/statistics/stats.hpp> 
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "EventSequence.hpp"
#include "LearningParams.hpp"

using namespace boost::accumulators;



class PointProcess{

	public:

		std::string name;
		unsigned int K; //number of event types
		double start_t; //start of the observation interval
		double end_t;   //end of the observation interval
		
		//TODO: make the inference support multiple eventsequences
		std::vector<EventSequence> train_data;
		std::vector<EventSequence> test_data;
		//EventSequence data;
		/****************************************************************************************************************************************************************************
		 * Model Construction and Destruction Methods
		******************************************************************************************************************************************************************************/
		PointProcess ();
		PointProcess(unsigned int k, double s, double e);
		PointProcess(std::ifstream &file); //constructor which loads a point process from a serialized file
		PointProcess (std::string name);
		PointProcess(std::string name, unsigned int k, double s, double e);
		
		virtual ~PointProcess();
		/****************************************************************************************************************************************************************************
		 * Model Utility Methods
		******************************************************************************************************************************************************************************/
		virtual std::string createModelDirectory() const=0;
		
		virtual std::string createModelInferenceDirectory() const=0;
		
		/****************************************************************************************************************************************************************************
		 * Model Generation Methods
		******************************************************************************************************************************************************************************/
		//generate an instance of the model from the priors
		virtual void generate()=0;
		
		/****************************************************************************************************************************************************************************
		 * Model Likelihood  Methods
		******************************************************************************************************************************************************************************/
		
		//if normalized is true it divides by the number of events (observed or not) in the sequence
		virtual double likelihood(const EventSequence & seq, bool normalized, double t0=-1) const=0;

		//likelihood (of only mutual excitation part)
	    virtual double likelihood(const EventSequence & seq, int k, int k2, double t0=-1) const=0;

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		virtual double loglikelihood(const EventSequence & seq, bool normalized, double t0=-1) const=0;

		//full log-likelihood (of both the background intensity and the mutual excitation part)
		virtual void posteriorModeLoglikelihood(EventSequence & seq, std::vector<double> & logl, bool normalized, double t0=-1) =0;

		//log-likelihood (of only the mutual excitation part)
		virtual double loglikelihood(const EventSequence & seq, int k, int k2, double t0=-1) const=0;
		
		/****************************************************************************************************************************************************************************
		 * Model Simulation Methods
		******************************************************************************************************************************************************************************/
		//simulation of the cluster point process
		virtual EventSequence simulate(unsigned int id=0)=0;
		
		//simulation of the cluster point process for interval dt starting from the end of the event sequence seq
		virtual void simulate(EventSequence & seq, double dt)=0;
		
		//simulation of the cluster point process of type k, starting from event e, for the time interval [te,ts]
		virtual void simulate(Event *e, int k, double ts, double te, EventSequence &s)=0;
		
		//simulation of the cluster point process to get only the next event in the interval [start_t, end_t] when the events are observed up to time th and there are no events in [th, start_t]
		//virtual void simulateNxt(const EventSequence & seq, double th, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N=1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) =0;
		
		//simulation of the cluster point process to get only the next event in the interval [start_t, end_t]
		virtual void simulateNxt(const EventSequence & seq, double start_t, double end_t, std::vector<Event *> & nxt_events, unsigned int N=1) =0;
	
		/****************************************************************************************************************************************************************************
		 * Model Memory Methods
		******************************************************************************************************************************************************************************/
		//load model parameters and samples from the posterior
		virtual void load(std::ifstream &file)=0;
				
		//save model parameters and samples from the posterior
		 virtual void save(std::ofstream &file )const=0;
		 
		 //loads data from file
		 virtual void loadData(std::ifstream &file, std::vector<EventSequence> & data, unsigned int file_type=0, const std::string & name="", double t0=-1,double t1=-1)=0;//TODO: better if the name of the file is given???

		 virtual void loadData(std::vector<std::ifstream> &file, std::vector<EventSequence> & data, unsigned int file_type=0, double t0=-1,double t1=-1)=0;//TODO: better if the name of the file is given???

		 //loads data from file
		 //virtual void loadData(std::vector<std::ifstream> &file, unsigned int file_type=0)=0;//TODO: should I keep only the std::vector version???
		
		//save model point parameters
		virtual void saveParameters(std::ofstream &file)const=0;
	
		//print model point parameters
		virtual void print(std::ostream &file)const=0;
		/****************************************************************************************************************************************************************************
		 * ModelIntensity Methods
		******************************************************************************************************************************************************************************/

		//compute intensity of type k at time t
		virtual double computeIntensity(unsigned int k, const EventSequence &s, double t) const=0;

		//compute intensity at equally spaced "nofps" timepoints within the observation window
		virtual void computeIntensity(unsigned int k, const EventSequence &s, boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v, unsigned int nofps) const=0;
		
		//compute intensity at determined timepoints t
		virtual void computeIntensity(unsigned int k, const EventSequence &s, const boost::numeric::ublas::vector<double>  &t, boost::numeric::ublas::vector<long double> & v) const=0;
				
		virtual void plotIntensity(const std::string &filename, unsigned int k, const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const=0;

		virtual void plotIntensities(const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const=0;

		virtual void plotIntensities(const std::string & dir_path_str, const EventSequence &s,  unsigned int nofps, bool write_png_file, bool write_csv_file) const=0;
		
		
		/****************************************************************************************************************************************************************************
		 * Model Inference Methods
		******************************************************************************************************************************************************************************/
		//it creates samples from the posterior given the data
	    virtual void fit(MCMCParams const * const p)=0;
	    
		//sets the parameters to the posterior mode and mean after fitting the model
		virtual void setPosteriorParams()=0;
				
		virtual void setPosteriorParents()=0;

		virtual void setPosteriorParents(EventSequence & seq)=0;

		//it discards the first samples of the mcmc inference, the burnin samples may have to be preserved for plotting purposes
		virtual void flushBurninSamples(unsigned int nof_burnin)=0;
		
		virtual void initProfiling(unsigned int runs,  unsigned int max_num_iter) =0;
		
		virtual void writeProfiling(const std::string & dir_path_str) const =0;
		
		/****************************************************************************************************************************************************************************
		 * Model Testing Methods
		******************************************************************************************************************************************************************************/
		
	
		virtual void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences,  double t0, double t1, unsigned int burnin_samples, const std::string & true_logl_summary, bool normalized) =0;
		
		virtual void testLogLikelihood(const std::string &test_dir_name, const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples, const std::string & true_logl_summary, bool normalized) =0;
		
		void testPredictions(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples, unsigned int offset=0);
		
		virtual void testPredictions(EventSequence & seq, const std::string &test_dir_path_str, double & mode_rmse, double & mean_rmse, double & mode_errorate, double & mean_errorate, double t0) =0;

		/****************************************************************************************************************************************************************************
		 * Model Prediction Methods
		******************************************************************************************************************************************************************************/
		
		void predictEventSequences(const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int offset_id=0, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES);
		
		virtual void predictEventSequence(const EventSequence & seq , const std::string &seq_dir_path_str, double t0=-1, unsigned int nof_samples=MODEL_SIMULATION_SAMPLES) =0;
	
		/****************************************************************************************************************************************************************************
		 * Model Inference Diagnostic-Plot Methods
		******************************************************************************************************************************************************************************/

		//utility functions for plotting mcmc diagnostics and inference results
		virtual void generateMCMCplots(unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool write_png_file, bool write_csv_file) =0 ;
		
		virtual void generateMCMCplots(const std::string & dir_path_str, unsigned int samples_step, bool true_values, unsigned int burnin_num_iter, bool write_png_file, bool write_csv_file) =0;
		
		virtual void generateMCMCTracePlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCTracePlots(bool write_png_file, bool write_csv_file)  const =0;
		
		virtual void generateMCMCMeanPlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCMeanPlots(bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCAutocorrelationPlots(const std::string & dir_path_str, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCAutocorrelationPlots(bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCPosteriorPlots(const std::string & dir_path_str, bool true_values, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCPosteriorPlots(bool true_values, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCIntensityPlots(const std::string & dir_path_str, bool true_values, bool write_png_file, bool write_csv_file) const =0;
		
		virtual void generateMCMCIntensityPlots(bool true_values, bool write_png_file, bool write_csv_file) const =0;

		virtual void generateMCMCTrainLikelihoodPlots(unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true, const std::string & plot_dir_path_str="") const =0;
		
		virtual void generateMCMCTestLikelihoodPlots(const std::string &seq_dir_name, const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int samples_step, bool true_values, bool write_png_file, bool write_csv_file=false, bool normalized=true,  double t0=-1, double t1=-1, const std::string & plot_dir_path_str="") const =0;

	private:
	
		/****************************************************************************************************************************************************************************
		 * Model Inference Private Methods
		******************************************************************************************************************************************************************************/
		
		//one step of bayesian inference for the model parameters and the latent variables
		virtual void mcmcStep(unsigned int thread_id, unsigned int run, unsigned int step, bool save, unsigned int iter_nof_threads,void * (*mcmcUpdateParams_)(void *), bool profiling)=0;
		
		//it prepares the structures for holding the posterior samples
		virtual void mcmcInit(MCMCParams const * const p)=0;
		
		//initialization for bayesian inference for the model parameters and the latent variables
		virtual void mcmcStartRun(unsigned int thread_id, unsigned int run_id, unsigned int iter_nof_threads)=0;
		
		/****************************************************************************************************************************************************************************
		 * Model Serialization Methods
		******************************************************************************************************************************************************************************/
	    friend class boost::serialization::access;
	    
	    void serialize( boost::archive::text_oarchive &ar, unsigned int );
	    
	    void serialize(boost::archive::text_iarchive &ar, unsigned int );
	    
	    void serialize( boost::archive::binary_oarchive &ar, unsigned int );
	    
	    void serialize(boost::archive::binary_iarchive &ar, unsigned int );

};

#endif /* POINTPROCESS_HPP_ */
