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
#include <stdio.h>
#include <stdlib.h>

#include "HawkesProcess.hpp"
#include "GeneralizedHawkesProcess.hpp"
#include "EventSequence.hpp"
#include "Kernels.hpp"
#include "stat_utils.hpp"
#include "gnuplot-iostream/gnuplot-iostream.h"
#include "debug.hpp"
#define NOF_SEQUENCES_PER_RUN 1
#define NOF_TEST_SEQUENCES 1
#define NOF_TRAIN_SEQUENCES 1



#define DEBUG 0

#if DEBUG_LEARNING
std::ofstream debug_learning_file{"debug_learning_report.txt"};
#endif
enum PriorFamily {GAMMA, EXPONENTIAL};
std::map<std::string, PriorFamily> prior_types = {{"GAMMA", GAMMA}, {"EXP", EXPONENTIAL}};

// for the polyagamma sampling: we used the library https://github.com/jwindle/BayesLogit/tree/master/Code


//compile command
// g++ -g -std=c++11 -Wall LearningParams.cpp Kernels.cpp PointProcess.cpp HawkesProcess.cpp  GeneralizedHawkesProcess.cpp EventSequence.cpp  polyagammaSampler/PolyaGamma.cpp polyagammaSampler/RNG.cpp polyagammaSampler/GRNG.cpp stat_utils.cpp plot_utils.cpp math_utils.cpp fit_sparse_ghp_models.cpp -lboost_iostreams -lboost_filesystem -lboost_system -lboost_serialization  -lboost_timer -lpthread -lgsl -lcblas -llapack -o fit_sghp



//synthetic experiments



// ./fit_sghp 1 simulation_study/ simulation_study/train_sequences/ghp_example_sequence_0.csv simulation_study/test_sequences/ csv_test_sequence_ 1000       ghp_example     2 0 20000 EXP EXP  true  7 1000  EXP  100       EXP 10       10 1   100 10 5    1000 0.01 100   1000 10 1     100000 0.015 100000 0.015 10000 5000 0 20000





/// multineuron experiments

//./fit_hp 5 spike_train_ghp/ spike_data_spont_activity_area17/train_sequences/multineuron_spike_train_4000.csv spike_data_spont_activity_area17/test_sequences/ multineuron_spike_test_4000_ 1  spike_data_ghp_4000 25 0 22236 EXP EXP true 0.100000 0.100000 GAMMA 10.000000 1000.000000 EXP 0.1 10.000000 1.000000 1.000000 100.000000 5.000000 100.000000 0.001000 10.000000 1000.000000 1.000000 10000.000000 10000 5 10000 0.001 10000 5000 22236 -1






enum KernelFamily {CONST, EXP, SIG, PWL, RAY};//TODO: store kernel type  variable for each derived class
std::map<std::string, KernelFamily> kernel_types = {{"CONST", CONST}, {"EXP", EXP}, {"SIG", SIG}, {"PWL", PWL}, {"RAY", RAY}};



void createIntensityKernels(char*** argvp, unsigned int & arg_id, unsigned int K, std::vector<ConstantKernel *> & m, std::vector<std::vector<Kernel *>> & phi, std::string & infer_dir_path_str){

	char **argv=*argvp;

	//read and create priors for the parameters of the excitation kernels
	double alpha_mu=atof(argv[arg_id++]);
	std::cout<<"alpha mu "<<alpha_mu<<std::endl;
	double beta_mu=atof(argv[arg_id++]);
	std::cout<<"beta mu "<<beta_mu<<std::endl;
	
	
	//create prior
	std::vector<std::vector< const DistributionParameters *>> p_mu_v(K);
	for(unsigned int k=0;k!=K;k++){
		p_mu_v[k].push_back(const_cast<GammaParameters *>(new GammaParameters{alpha_mu, beta_mu}));
	}
	infer_dir_path_str+="_"+std::to_string(alpha_mu);
	infer_dir_path_str+="_"+std::to_string(beta_mu);

	//create constant kernel for the exogenous intensity
     //background intensities
    for(unsigned int k=0;k<K;k++){
	   ConstantKernel *mu_k=new ConstantKernel{p_mu_v[k]};
	   m.push_back(mu_k);
    }
  
    // prior distributions for the parameters of the excitatory kernel
    std::vector<std::vector<std::vector< const DistributionParameters *>>> p_phi_v(K,std::vector<std::vector< const DistributionParameters *>>(K));
        
    //prior type for the multiplicative coefficient of the mutually exciting intensities
    const PriorFamily phi_prior_type=prior_types[argv[arg_id++]]; 
    
    switch(phi_prior_type){
			case GAMMA:{
				//sample from a gamma distribution
				double alpha_phi_k_kp=atof(argv[arg_id++]);
				double beta_phi_k_kp=atof(argv[arg_id++]);
				std::cout<<"alpha phi "<<alpha_phi_k_kp<<std::endl;
				std::cout<<"beta phi "<<beta_phi_k_kp<<std::endl;
				
				infer_dir_path_str+="_gm_";
				infer_dir_path_str+="_"+std::to_string(alpha_phi_k_kp);
				infer_dir_path_str+="_"+std::to_string(beta_phi_k_kp);
				
			
				for(unsigned int k=0;k!=K;k++){
					for(unsigned int k2=0;k2!=K;k2++){
						//for the multiplicative coefficient
						p_phi_v[k][k2].push_back(const_cast<GammaParameters *>(new GammaParameters(alpha_phi_k_kp, beta_phi_k_kp)));
					}
				}
				break;
			}
			case EXPONENTIAL:{
				double c_k_kp=atof(argv[arg_id++]);
				
				infer_dir_path_str+="_exp_";
				infer_dir_path_str+="_"+std::to_string(c_k_kp);
				
				for(unsigned int k=0;k!=K;k++){
					for(unsigned int k2=0;k2!=K;k2++){
						//for the multiplicative coefficient
						p_phi_v[k][k2].push_back(const_cast<ExponentialParameters *>(new ExponentialParameters(c_k_kp)));
					}
				}
				break;
			}
    }
	
    //prior type for the decaying coefficient of the mutually exciting intensities
    std::cout<<"parse the decaying coefficient\n";
    const PriorFamily phi_exp_prior_type=prior_types[argv[arg_id++]]; 
 
    switch(phi_exp_prior_type){
			case GAMMA:{
				 //  std::cout<<"gamma distribution\n";
				//sample from a gamma distribution
				double alpha_phi_k_kp=atof(argv[arg_id++]);
				double beta_phi_k_kp=atof(argv[arg_id++]);
				std::cout<<"alpha phi delta "<<alpha_phi_k_kp<<std::endl;
				
				std::cout<<"beta phi delta "<<beta_phi_k_kp<<std::endl;
				
				infer_dir_path_str+="_gm_";
				infer_dir_path_str+="_"+std::to_string(alpha_phi_k_kp);
				infer_dir_path_str+="_"+std::to_string(beta_phi_k_kp);
			
				for(unsigned int k=0;k!=K;k++){
					for(unsigned int k2=0;k2!=K;k2++){
						//for the decaying coefficient
						p_phi_v[k][k2].push_back(const_cast<GammaParameters *>(new GammaParameters(alpha_phi_k_kp, beta_phi_k_kp)));
					}
				}
				break;
			}
			case EXPONENTIAL:{
				  std::cout<<"exponential distribution\n";
				//sample from a exponential distribution
				double lambda_phi_k_kp=atof(argv[arg_id++]);
				
				infer_dir_path_str+="_exp_";
				infer_dir_path_str+="_"+std::to_string(lambda_phi_k_kp);
			
				
				for(unsigned int k=0;k!=K;k++){
					for(unsigned int k2=0;k2!=K;k2++){
						//for the decaying coefficient
						p_phi_v[k][k2].push_back(const_cast<ExponentialParameters *>(new ExponentialParameters(lambda_phi_k_kp)));
					}
				}
				break;
			}
    }

	//trigerring kernels
    for(unsigned int k=0;k<K;k++){
	   std::vector<Kernel *> phi_k;
	   //read the prior for the excitation/inhibition kernels
	   for(unsigned int kp=0;kp<K;kp++){

		   //todo: cases with power law and exp
		   Kernel *phi_k_kp=new ExponentialKernel{ p_phi_v[k][kp]};
		   phi_k.push_back(phi_k_kp);

      }
      phi.push_back(phi_k);
   }

}

void createHistoryKernels(char*** argvp, unsigned int & arg_id, KernelFamily type, std::vector<Kernel *> & h, std::string & infer_dir_path_str){

	char **argv=*argvp;

	switch(type){
		case EXP:{
			double alpha_h=atof(argv[arg_id++]);
			double beta_h=atof(argv[arg_id++]);
			double lambda_h=atof(argv[arg_id++]);
			std::cout<<"exponential history\n";
			std::cout<<alpha_h<<std::endl;
			std::cout<<beta_h<<std::endl;
			std::cout<<lambda_h<<std::endl;

			//todo: cases with rayleigh and exp
			std::vector< const DistributionParameters *> p_h_v;
			GammaParameters ghp{alpha_h, beta_h};
			p_h_v.push_back(const_cast<GammaParameters *>(&ghp));

			//read prior for history kernels

			//the prior for the decaying coefficient
			ExponentialParameters ehp{lambda_h};
			p_h_v.push_back(const_cast<ExponentialParameters *>(&ehp));

			//create history kernels
			Kernel *h_d=new ExponentialKernel{ p_h_v};

			h.push_back(h_d);
		 	//generate mcmc inference plots

			 infer_dir_path_str+="_"+std::to_string(alpha_h);
		 	 infer_dir_path_str+="_"+std::to_string(beta_h);
		 	 infer_dir_path_str+="_"+std::to_string(lambda_h);
			break;
		}

		case RAY:{
			double alpha_h=atof(argv[arg_id++]);
			double beta_h=atof(argv[arg_id++]);
			double lambda_h=atof(argv[arg_id++]);

			//todo: cases with rayleigh and exp
			std::vector< const DistributionParameters *> p_h_v;
			GammaParameters ghp{alpha_h, beta_h};
			p_h_v.push_back(const_cast<GammaParameters *>(&ghp));

			//read prior for history kernels

			//the prior for the decaying coefficient
			ExponentialParameters ehp{lambda_h};
			p_h_v.push_back(const_cast<ExponentialParameters *>(&ehp));

			//create history kernels
			Kernel *h_d=new RayleighKernel{ p_h_v};

			h.push_back(h_d);

		 	//generate mcmc inference plots

			 infer_dir_path_str+="_"+std::to_string(alpha_h);
		 	 infer_dir_path_str+="_"+std::to_string(beta_h);
		 	 infer_dir_path_str+="_"+std::to_string(lambda_h);
			break;
		}

		default:{

			break;
		}

	}


}

void createLogisticKernel(char*** argvp, unsigned int & arg_id, bool hierarchical, unsigned int K, std::vector<LogisticKernel *> &psi, std::string & infer_dir_path_str){
	char **argv=*argvp;

	//read prior parameters for the bias term
	double mu_0=atof(argv[arg_id++]);
	double sigma_0=atof(argv[arg_id++]);
	std::cout<<"variance for bias term from the consode "<<sigma_0<<std::endl;
	//create prior
	std::vector<std::vector<const DistributionParameters *>> p_w_v(K);
	double mu_d=0.0;
	double sigma_d=1.0;
	if(!hierarchical){
		mu_d=atof(argv[arg_id++]);
		sigma_d=atof(argv[arg_id++]);
	}

	for(unsigned int k=0;k!=K;k++){
		//for the bias term
	    p_w_v[k].push_back(const_cast<NormalParameters *>(new NormalParameters{mu_0, sigma_0}));
	    //std::cout<<"type of prior for weight "<<p_w_v[k][0]->type<<std::endl;
	    //for the interaction weights
	    for(unsigned int k2=0;k2!=K;k2++)//TODO: in case of a flat function, read the mean and var (instead of using 0.0, 1.0)
	    	p_w_v[k].push_back(const_cast<NormalParameters *>(new NormalParameters{mu_d, sigma_d}));
	}

    //this is K-dimensional
    for(unsigned int k=0;k<K;k++){
    	LogisticKernel *psi_k=new LogisticKernel{ p_w_v[k]};
       psi.push_back(psi_k);
    }
 	//generate mcmc inference plots

 	 infer_dir_path_str+="_"+std::to_string(mu_0);
 	 infer_dir_path_str+="_"+std::to_string(sigma_0);
}

void createLogisticKernelPrior(char*** argvp, unsigned int & arg_id, unsigned int K, std::vector<std::vector<SparseNormalGammaParameters *> > & pt_hp, std::string & infer_dir_path_str){

	char **argv=*argvp;

	//read the hyperpriors for the logistic part, todo: option for hierarchical vs flat point process
	double nu_mu=atof(argv[arg_id++]);
	std::cout<<"nu mu "<<nu_mu<<std::endl;
	double alpha_mu2=atof(argv[arg_id++]);
	std::cout<<"alpha mu2 "<<alpha_mu2<<std::endl;
	double lambda=atof(argv[arg_id++]);
	std::cout<<"lambda "<<lambda<<std::endl;

	double nu_tau=atof(argv[arg_id++]);
	std::cout<<"nu_tau "<<nu_tau<<std::endl;
	double alpha_tau=atof(argv[arg_id++]);
	std::cout<<"alpha_tau "<<alpha_tau<<std::endl;
	double beta_tau=atof(argv[arg_id++]);
	std::cout<<"beta_tau "<<beta_tau<<std::endl;

	double delta_tau=atof(argv[arg_id++]);
	std::cout<<"delta_tau "<<delta_tau<<std::endl;
	double x_tau=atof(argv[arg_id++]);
	std::cout<<"x_tau "<<x_tau<<std::endl;
	double delta_mu=atof(argv[arg_id++]);
	std::cout<<"delta_mu "<<delta_mu<<std::endl;
	double x_mu=atof(argv[arg_id++]);
	std::cout<<"x_mu "<<x_mu<<std::endl;
	ActivationFunction *phi_tau=new GeneralSigmoid(x_tau,delta_tau);
	ActivationFunction *phi_mu=new GeneralSigmoid(x_mu, delta_mu);

	std::cout<<"hyperpriors created\n";

    for(unsigned int k=0;k<K;k++){
    	std::vector<SparseNormalGammaParameters *> pt_hp_k(K,new SparseNormalGammaParameters(lambda, alpha_tau, beta_tau, nu_tau, alpha_mu2, nu_mu, phi_tau, phi_mu));
        pt_hp.push_back(pt_hp_k);
    }

 	//generate mcmc inference plots
 	 infer_dir_path_str+="_"+std::to_string(nu_mu);
 	 infer_dir_path_str+="_"+std::to_string(alpha_mu2);
 	 infer_dir_path_str+="_"+std::to_string(lambda);
 	 infer_dir_path_str+="_"+std::to_string(nu_tau);
 	 infer_dir_path_str+="_"+std::to_string(alpha_tau);
 	 infer_dir_path_str+="_"+std::to_string(beta_tau);
	 infer_dir_path_str+="_"+std::to_string(delta_tau);
	 infer_dir_path_str+="_"+std::to_string(x_tau);
	 infer_dir_path_str+="_"+std::to_string(delta_mu);
	 infer_dir_path_str+="_"+std::to_string(x_mu);

}

int main(int argc, char* argv[]){
	
	unsigned int nof_threads=atoi(argv[1]);//nof testing sequences
	
	const std::string pp_dir=argv[2]; //read the output directory name 
	std::cout<<"pp dir "<<pp_dir<<std::endl;
	const std::string pp_train_file=argv[3];//read the train sequence filename
	std::cout<<"train file "<<pp_train_file<<std::endl;
    const std::string pp_test_dir=argv[4];//read the directory with the testing timeseries 
    std::cout<<"test file dir "<<pp_test_dir<<std::endl;
	const std::string test_seqfilename_prefix=argv[5];//read the directory with the testing timeseries
	std::cout<<"test file prefix "<<test_seqfilename_prefix<<std::endl;
	unsigned int nof_test_sequences=atoi(argv[6]);//nof testing sequences
	std::cout<<"nof test sequence "<<nof_test_sequences<<std::endl;
	
	boost::filesystem::path curr_path_boost=boost::filesystem::current_path();
	std::string curr_path_str=curr_path_boost.string();
	std::string ghp_dir_path_str=curr_path_str+"/"+pp_dir;
	std::string  pp_train_file_path_str=curr_path_str+"/"+pp_train_file;
	  


	const std::string model_name=argv[7];//read the model name
	std::cout<<"model name "<<model_name<<std::endl;
	
	std::string infer_dir_path_str=ghp_dir_path_str+model_name+"_ip2_";
	std::string pp_test_dir_path_str=curr_path_str+"/"+pp_test_dir;
	
	
	const unsigned int K=atoi(argv[8]);//read the number of types
	std::cout<<"nof types "<<K<<std::endl;
	const double T_begin=atof(argv[9]);//read the start of the observation window of the process
	std::cout<<"start of interval "<<T_begin<<std::endl;
	const double T_end=atof(argv[10]);//read the end of the observation window of the process
	std::cout<<"end of interval "<<T_end<<std::endl;
	
	//const KernelFamily phi_type=kernel_types[argv[11]]; //type for the mutually exciting intensities
	const KernelFamily psi_type=kernel_types[argv[12]]; //type for the history kernel functions


	std::string model_type=argv[13];//whether a hierarchical model for imposing constraints between inhibitory vs excitatory relationship
	std::cout<<"model type "<<model_type<<std::endl;
	bool hier_model=!model_type.compare("true");
	    //read prior parameters and create kernels for the exogenous and excitatory intensity
	unsigned int arg_id=14;
	std::vector<ConstantKernel *>  m;
	std::vector<std::vector<Kernel *>>  phi;
	createIntensityKernels(&argv, arg_id, K, m, phi, infer_dir_path_str);
	std::cout<<"intensity kernels created\n";

	//create logistic kernels
	std::vector<LogisticKernel *> psi;
	createLogisticKernel(&argv, arg_id, hier_model, K, psi, infer_dir_path_str);
	std::cout<<"logistic kernels created\n";


	//create history kernel functions
	std::vector<Kernel *> h;
	createHistoryKernels(&argv,arg_id,psi_type, h, infer_dir_path_str);
    std::cout<<"history kernel functions created\n";


	 //create prior for the logistic kernel
	 std::vector<std::vector<SparseNormalGammaParameters *> > pt_hp;
	 if(hier_model)
		 createLogisticKernelPrior(&argv, arg_id, K, pt_hp, infer_dir_path_str);
	 std::cout<<"hyperpriors created\n";


	 GeneralizedHawkesProcess *ghp_model;
	 if(hier_model){
		 std::cout<<"hierarchical model\n";
		 ghp_model=new GeneralizedHawkesProcess{model_name,K,T_begin,T_end, m,phi,psi, h, pt_hp};
	 }
	 else{
		std::cout<<"flat model\n";
		ghp_model=new GeneralizedHawkesProcess{model_name,K,T_begin,T_end, m,phi,psi, h};
	}
	std::cout<<"model created\n";
  	
 	 //load model with data
 	 std::ifstream train_seq_file{pp_train_file_path_str};
 	 ghp_model->loadData(train_seq_file, ghp_model->train_data,1,model_name+"_train_seq", T_begin, T_end);
 	 std::cout<<"load data done\n";
 	 train_seq_file.close();
 	 
	 unsigned int nof_iters=atoi(argv[arg_id++]);//nof mcmc iterations
	 unsigned int nof_burnin_iters=atoi(argv[arg_id++]);//nof burnin mcmc iterations
	 const double test_begin=atof(argv[arg_id++]); //read the start of the testing window of the process
	 std::cout<<"start of testing interval "<<T_begin<<std::endl;

	 const double test_end=atof(argv[arg_id]);


 	 infer_dir_path_str+="/";
 		  if(!boost::filesystem::is_directory(infer_dir_path_str) && !boost::filesystem::create_directory(infer_dir_path_str))
 			  std::cerr<<"Couldn't create auxiliary folder."<<std::endl;
 		  std::cout<<"create dir done \n";
 	 
 	 //fit model
	 GeneralizedHawkesProcessMCMCParams mcmc_params(RUNS,0, nof_iters, SAMPLES_STEP, MCMC_SAVE_PERIOD, NOF_SEQUENCES_PER_RUN, true, infer_dir_path_str, nof_threads, nof_threads, PLOT_NOF_THREADS, NOF_PLOT_POINTS);
	 
	 	 
	 std::ofstream mcmc_params_file{ infer_dir_path_str+"mcmc_params.txt"};
	 mcmc_params.print(mcmc_params_file);
	 mcmc_params_file.close();
	 	
	 ghp_model->fit(&mcmc_params); 
	 std::cout<<"model fitted\n";


     //generate mcmc plots (marginal posterior disrtibutions and mcmc diagnostics)
	// ghp_model->generateMCMCplots(infer_dir_path_str, SAMPLES_STEP, false, nof_burnin_iters,true, true);
	 
	 ghp_model->flushBurninSamples(nof_burnin_iters);//flush the burning samples
	 ghp_model->generateMCMCPosteriorPlots(infer_dir_path_str,false, true, true);


	 //get the testloglikelihood with the point (posterior mean and mode) estimates
	 ghp_model->testLogLikelihood(infer_dir_path_str,pp_test_dir_path_str,test_seqfilename_prefix,nof_test_sequences, test_begin, test_end ,0, "",true);
	 
	 //save learned model
	 std::string ghp_ser_file_str_2= infer_dir_path_str+"ghp_ser_learned_file.txt";
	 std::cout<<"serialized learned model filename "<<ghp_ser_file_str_2<<std::endl;
	 std::ofstream ghp_ser_file_2{ghp_ser_file_str_2};
	 ghp_model->save(ghp_ser_file_2);
	 ghp_ser_file_2.close();
		
	 //print learned model	    
	 std::string pp_learned= infer_dir_path_str+"ghp_learned_file.txt";
	 std::cout<<"serialized learned model filename "<<pp_learned<<std::endl;
	 std::ofstream pp_learned_file{pp_learned};
	 ghp_model->print(pp_learned_file);
	 pp_learned_file.close();


	 return 0;
}
