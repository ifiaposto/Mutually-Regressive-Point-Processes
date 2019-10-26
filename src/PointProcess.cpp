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


BOOST_SERIALIZATION_ASSUME_ABSTRACT(PointProcess)



/****************************************************************************************************************************************************************************
 *
 * Cluster Point Process
 *
******************************************************************************************************************************************************************************/

PointProcess::PointProcess()=default;

PointProcess::PointProcess(unsigned int k, double s, double e): K{k}, start_t{s}, end_t{e}{};

PointProcess::PointProcess(std::ifstream &file){
	boost::archive::binary_iarchive ia{file};
	ia >> *this;
}


PointProcess::PointProcess(std::string n): name{n} {};

PointProcess::PointProcess(std::string n, unsigned int k, double s, double e): name{n}, K{k}, start_t{s}, end_t{e}{};

PointProcess::~PointProcess(){
	for(auto seq_iter=train_data.begin();seq_iter!=train_data.end();seq_iter++)
		seq_iter->flush();
	train_data.clear();
	
	for(auto seq_iter=test_data.begin();seq_iter!=test_data.end();seq_iter++)
		seq_iter->flush();
	test_data.clear();
}

void PointProcess::serialize( boost::archive::text_oarchive &ar, unsigned int ){
	
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & train_data;
	ar & test_data;
	
	
};

void PointProcess::serialize(boost::archive::text_iarchive &ar, unsigned int ){
	
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & train_data;
	ar & test_data;

	
};

void PointProcess::serialize( boost::archive::binary_oarchive &ar, unsigned int ){
	
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & train_data;
	ar & test_data;	
};

void PointProcess::serialize(boost::archive::binary_iarchive &ar, unsigned int ){
	
	ar & K;
	ar & start_t;
	ar & end_t;
	ar & train_data;
	ar & test_data;
};

void PointProcess::predictEventSequences(const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int offset_id, unsigned int nof_samples){
	for(unsigned int n=0;n<nof_test_sequences;n++){
	
		//open sequence file and load
		std::string csv_seq_filename{seq_dir_path_str+seq_prefix+std::to_string(offset_id+n)+".csv"};

		std::ifstream test_seq_file{csv_seq_filename};
		EventSequence test_seq;

		test_seq.name=seq_prefix+std::to_string(offset_id+n);
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
		
		test_seq.start_t=start_t;
		test_seq.end_t=end_t;
		test_seq_file.close();
		
	    predictEventSequence(test_seq , seq_dir_path_str, -1, nof_samples);
	}	
}

void PointProcess::testPredictions(const std::string &test_dir_path_str, const std::string &seq_dir_path_str,const std::string &seq_prefix, unsigned int nof_test_sequences, unsigned int burnin_samples, unsigned int offset_id){

	//flush the burnin samples if needed, compute point posterior estimates (mean and mode) with the rest of the samples
	flushBurninSamples(burnin_samples);
	setPosteriorParams();

	//open file with the rmse metrics
	std::string test_filename=test_dir_path_str+"test_sequences_prediction.csv"; //in each row it will hold the mean rmse for the predicition of the arrival time of the next event for each test event sequence, one column for post mean/ post mode
	std::ofstream test_file{test_filename};
	test_file<<"sequence name, mode rmse, mean rmse, mode error rate, mean error rate"<<std::endl;//total rmse across all time-steps of the event sequence
	
	//open file with the error rate metrics

	
	//accumulators for the metrics across all the test event sequences
	accumulator_set<double, stats<tag::mean,tag::variance>> mode_rmse_acc;
	accumulator_set<double, stats<tag::mean,tag::variance>> mean_rmse_acc;
	accumulator_set<double, stats<tag::mean,tag::variance>> mode_errorrate_acc;
	accumulator_set<double, stats<tag::mean,tag::variance>> mean_errorrate_acc;


	for(unsigned int n=0;n<nof_test_sequences;n++){
		//open sequence file and load
		std::string csv_seq_filename{seq_dir_path_str+seq_prefix+std::to_string(offset_id+n)+".csv"};
		std::ifstream test_seq_file{csv_seq_filename};
		EventSequence test_seq;

		test_seq.name=seq_prefix+std::to_string(offset_id+n);
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
		
		test_seq.start_t=start_t;
		test_seq.end_t=end_t;
		test_seq_file.close();
		

		double mode_rmse=0.0;
		double mean_rmse=0.0;
		double mode_errorrate=0.0;
		double mean_errorrate=0.0;

		testPredictions(test_seq, test_dir_path_str, mode_rmse, mean_rmse, mode_errorrate, mean_errorrate, -1);
		mode_rmse_acc(mode_rmse);
		mean_rmse_acc(mean_rmse);
		mode_errorrate_acc(mode_errorrate);
		mean_errorrate_acc(mean_errorrate);
		
		test_file<<test_seq.name<<","<<mode_rmse<<","<<mean_rmse<<","<<mode_errorrate<<","<<mean_errorrate<<std::endl;
	}
	test_file<<"all, mode rmse, std mode rmse, mean rmse, std mean rmse, mode error rate, std mode error rate, mean error rate, std mean error rate"<<std::endl;
	test_file<<"summary,"<<mean(mode_rmse_acc)<<","<<std::sqrt(variance(mode_rmse_acc))<<","<<mean(mean_rmse_acc)<<","<<std::sqrt(variance(mean_rmse_acc))<<","<<mean(mode_errorrate_acc)<<","<<std::sqrt(variance(mode_errorrate_acc))<<","<<mean(mean_errorrate_acc)<<","<<std::sqrt(variance(mean_errorrate_acc))<<std::endl;
	test_file.close();
}

