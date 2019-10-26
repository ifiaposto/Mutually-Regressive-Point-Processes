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

#include "LearningParams.hpp"


BOOST_CLASS_EXPORT(MCMCParams)
BOOST_CLASS_EXPORT(HawkesProcessMCMCParams)
BOOST_CLASS_EXPORT(GeneralizedHawkesProcessMCMCParams)
	
MCMCParams::MCMCParams(){};

MCMCParams::MCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string  d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp): 
		runs(r), 
		nof_burnin_iters(b), 
		max_num_iter(m), 
		samples_step(s), 
		mcmc_save_period(sp), 
		nof_sequences(1), 
		profiling(p), 
		dir_path_str(d),
		mcmc_fit_nof_threads(ft),
		mcmc_iter_nof_threads(it),
		plot_nof_threads(pt),
		nof_plot_points(pp)
		{};
MCMCParams::MCMCParams(unsigned int n, bool p, const std::string & d):nof_sequences(n), profiling(p), dir_path_str(d){};

MCMCParams::MCMCParams(const MCMCParams &p): 
		runs(p.runs),
		nof_burnin_iters(p.nof_burnin_iters),
		max_num_iter(p.max_num_iter),
		samples_step(p.samples_step),
		mcmc_save_period(p.mcmc_save_period){};
	
	
void MCMCParams::print(std::ostream & file){
		file<<"runs "<<runs<<std::endl;
		file<<"nof burnin iters "<<nof_burnin_iters<<std::endl;
		file<<"max num iter "<<max_num_iter<<std::endl;
		file<<"samples step "<<samples_step<<std::endl;
		file<<"mcmc save perior "<<mcmc_save_period<<std::endl;
		
	}
	

void MCMCParams::serialize( boost::archive::text_oarchive &ar, const unsigned int version){
//    	ar & runs;
//    	ar & nof_burnin_iters;
//    	ar & max_num_iter;
//    	ar & samples_step;
//    	ar & mcmc_save_period;
//    	ar & nof_sequences;
//    	ar & profiling;
//    	ar & dir_path_str;
//    	ar & mcmc_fit_nof_threads;
//    	ar & mcmc_iter_nof_threads;
//    	ar & plot_nof_threads;
//    	ar & nof_plot_points;   	
 }

void MCMCParams::serialize( boost::archive::text_iarchive &ar, const unsigned int version){
 //   	ar & runs;
//    	ar & nof_burnin_iters;
//    	ar & max_num_iter;
//    	ar & samples_step;
//    	ar & mcmc_save_period;
//    	ar & nof_sequences;
//    	ar & profiling;
//    	ar & dir_path_str;
//    	ar & mcmc_fit_nof_threads;
//    	ar & mcmc_iter_nof_threads;
//    	ar & plot_nof_threads;
//    	ar & nof_plot_points;
    }
void MCMCParams::serialize( boost::archive::binary_oarchive &ar, const unsigned int version){
//    	ar & runs;
//    	ar & nof_burnin_iters;
//    	ar & max_num_iter;
//    	ar & samples_step;
//    	ar & mcmc_save_period;
//    	ar & nof_sequences;
//    	ar & profiling;
//    	ar & dir_path_str;
//    	ar & mcmc_fit_nof_threads;
//    	ar & mcmc_iter_nof_threads;
//    	ar & plot_nof_threads;
//    	ar & nof_plot_points;
    }
void MCMCParams::serialize( boost::archive::binary_iarchive &ar, const unsigned int version){
//    	ar & runs;
//    	ar & nof_burnin_iters;
//    	ar & max_num_iter;
//    	ar & samples_step;
//    	ar & mcmc_save_period;
//    	ar & nof_sequences;
//    	ar & profiling;
//    	ar & dir_path_str;
//    	ar & mcmc_fit_nof_threads;
//    	ar & mcmc_iter_nof_threads;
//    	ar & plot_nof_threads;
//    	ar & nof_plot_points;
    }


HawkesProcessMCMCParams::HawkesProcessMCMCParams()=default;
	
HawkesProcessMCMCParams::HawkesProcessMCMCParams(const HawkesProcessMCMCParams &p): 
		MCMCParams(p),
		d_t_am_sd(p.d_t_am_sd),
		d_t_am_t0(p.d_t_am_t0),
		d_t_am_c0(p.d_t_am_t0){};
	
HawkesProcessMCMCParams::HawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp):
		MCMCParams(r,b,m,s,sp,n,p,d, ft, it, pt, pp){};


HawkesProcessMCMCParams::HawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp,double sd, double t0, double c0): 
		MCMCParams(r,b,m,s,sp,n,p,d, ft, it, pt, pp), 
		d_t_am_sd(sd), 
		d_t_am_t0(t0), 
		d_t_am_c0(c0){};

HawkesProcessMCMCParams::HawkesProcessMCMCParams(unsigned int n, bool p, const std::string & d):MCMCParams(n,p,d){};

	
	
void HawkesProcessMCMCParams::print(std::ostream & file){
	MCMCParams::print(file);
	file<<"adaptive metropolis hastings regularization parameter for the decaying coefficient in the trigerred intensity "<<d_t_am_sd<<std::endl;
	file<<"adaptive metropolis hastings initial period for the decaying coefficient in the trigerred intensity "<<d_t_am_t0<<std::endl;
	file<<"adaptive metropolis hastings variance for the decaying coefficient in the trigerred intensity "<<d_t_am_c0<<std::endl;
}
	
void HawkesProcessMCMCParams::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
//	    boost::serialization::void_cast_register<HawkesProcessMCMCParams,MCMCParams>();
//	    ar & boost::serialization::base_object<MCMCParams>(*this);
//	    ar & d_t_am_sd;
//	    ar & d_t_am_t0;
//	    ar & d_t_am_c0;
	}
	
void HawkesProcessMCMCParams::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
//	    boost::serialization::void_cast_register<HawkesProcessMCMCParams,MCMCParams>();
//	    ar & boost::serialization::base_object<MCMCParams>(*this);
//	    ar & d_t_am_sd;
//	    ar & d_t_am_t0;
//	    ar & d_t_am_c0;
}
	
void HawkesProcessMCMCParams::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
//	    boost::serialization::void_cast_register<HawkesProcessMCMCParams,MCMCParams>();
//	    ar & boost::serialization::base_object<MCMCParams>(*this);
//	    ar & d_t_am_sd;
//	    ar & d_t_am_t0;
//	    ar & d_t_am_c0;
	}
	
void HawkesProcessMCMCParams::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
//	    boost::serialization::void_cast_register<HawkesProcessMCMCParams,MCMCParams>();
//	    ar & boost::serialization::base_object<MCMCParams>(*this);
//	    ar & d_t_am_sd;
//	    ar & d_t_am_t0;
//	    ar & d_t_am_c0;
	}


GeneralizedHawkesProcessMCMCParams::GeneralizedHawkesProcessMCMCParams(){};
	
GeneralizedHawkesProcessMCMCParams::GeneralizedHawkesProcessMCMCParams(const GeneralizedHawkesProcessMCMCParams &p): 
		HawkesProcessMCMCParams(p),
		c_t_am_sd(p.c_t_am_sd),
		c_t_am_t0(p.c_t_am_t0),
		c_t_am_c0(p.c_t_am_c0),
		c_h_am_sd(p.c_h_am_sd),
		c_h_am_t0(p.c_h_am_t0),
		c_h_am_c0(p.c_h_am_c0),
		d_h_am_sd(p.d_h_am_sd),
		d_h_am_t0(p.d_h_am_t0),
		d_h_am_c0(p.d_h_am_c0),
		t_w_am_sd(p.t_w_am_sd),
		t_w_am_t0(p.t_w_am_t0),
		t_w_am_c0(p.t_w_am_c0)
		{};
	
	
GeneralizedHawkesProcessMCMCParams::GeneralizedHawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp):
		HawkesProcessMCMCParams(r,b,m,s,sp,n,p,d, ft, it, pt, pp){};
		
GeneralizedHawkesProcessMCMCParams::GeneralizedHawkesProcessMCMCParams(unsigned int n, bool p, const std::string & d):HawkesProcessMCMCParams(n,p, d){};
	
GeneralizedHawkesProcessMCMCParams::GeneralizedHawkesProcessMCMCParams(unsigned int r, unsigned int b, unsigned int m, unsigned int s, unsigned int sp, unsigned int n, bool p, const std::string & d, unsigned int ft, unsigned int it, unsigned int pt, unsigned int pp, double sd, double t0, double c0,
									   double tcsd, double tct0, double tcc0,
									   double hcsd, double hct0, double hcc0,
									   double hdsd, double hdt0, double hdc0,
									   double twsd, double twt0, double twc0): 
										   HawkesProcessMCMCParams(r,b,m,s,sp,n,p,d, ft, it, pt, pp, sd,t0,c0),
										   c_t_am_sd(tcsd), c_t_am_t0(tct0), c_t_am_c0(tcc0),
										   c_h_am_sd(hcsd), c_h_am_t0(hct0), c_h_am_c0(hcc0),
										   d_h_am_sd(hdsd), d_h_am_t0(hdt0), d_h_am_c0(hdc0),
										   t_w_am_sd(twsd), t_w_am_t0(twt0), t_w_am_c0(twc0)
										{};
	

void GeneralizedHawkesProcessMCMCParams::print(std::ostream & file){
	HawkesProcessMCMCParams::print(file);
	file<<"adaptive metropolis hastings regularization parameter for the multiplicative coefficient in the trigerred intensity "<<c_t_am_sd<<std::endl;
	file<<"adaptive metropolis hastings initial period for the multiplicative coefficient in the trigerred intensity "<<c_t_am_t0<<std::endl;
	file<<"adaptive metropolis hastings variance for the multiplicative coefficient in the trigerred intensity "<<c_t_am_c0<<std::endl;
	
	file<<"adaptive metropolis hastings regularization parameter for the multiplicative coefficient in the history kernel function "<<c_h_am_sd<<std::endl;
	file<<"adaptive metropolis hastings initial period for the multiplicative coefficient in the history kernel function "<<c_h_am_t0<<std::endl;
	file<<"adaptive metropolis hastings variance for the multiplicative coefficient in the history kernel function "<<c_h_am_c0<<std::endl;
	
	file<<"adaptive metropolis hastings regularization parameter for the decaying coefficient in the history kernel function "<<d_h_am_sd<<std::endl;
	file<<"adaptive metropolis hastings initial period for the decaying coefficient in the history kernel function "<<d_h_am_t0<<std::endl;
	file<<"adaptive metropolis hastings variance for the decaying coefficient in the history kernel function "<<d_h_am_c0<<std::endl;
	
	file<<"adaptive metropolis hastings regularization parameter for the precision of the interaction weights in the logistic kernel "<<t_w_am_sd<<std::endl;
	file<<"adaptive metropolis hastings initial period for the precision of the interaction weights in the logistic kernel "<<t_w_am_t0<<std::endl;
	file<<"adaptive metropolis hastings variance for the precision of the interaction weights in the logistic kernel "<<t_w_am_c0<<std::endl;
	
}

	
void GeneralizedHawkesProcessMCMCParams::serialize(boost::archive::text_oarchive &ar,  unsigned int version){
//		boost::serialization::void_cast_register<GeneralizedHawkesProcessMCMCParams,HawkesProcessMCMCParams>();
//		ar & boost::serialization::base_object<HawkesProcessMCMCParams>(*this);
//		ar & c_t_am_sd;
//		ar & c_t_am_t0;
//		ar & c_t_am_c0;
//		ar & c_h_am_sd;
//		ar & c_h_am_t0;
//		ar & c_h_am_c0;
//		ar & d_h_am_sd;  
//		ar & d_h_am_t0;
//		ar & d_h_am_c0;  
//		ar & t_w_am_sd;    
//		ar & t_w_am_t0;  
//		ar & t_w_am_c0;  
}
	
void GeneralizedHawkesProcessMCMCParams::serialize(boost::archive::text_iarchive &ar,  unsigned int version){
//		boost::serialization::void_cast_register<GeneralizedHawkesProcessMCMCParams,HawkesProcessMCMCParams>();
//		ar & boost::serialization::base_object<HawkesProcessMCMCParams>(*this);
//		ar & c_t_am_sd;
//		ar & c_t_am_t0;
//		ar & c_t_am_c0;
//		ar & c_h_am_sd;
//		ar & c_h_am_t0;
//		ar & c_h_am_c0;
//		ar & d_h_am_sd;  
//		ar & d_h_am_t0;
//		ar & d_h_am_c0;  
//		ar & t_w_am_sd;    
//		ar & t_w_am_t0;  
//		ar & t_w_am_c0;
}
	
void GeneralizedHawkesProcessMCMCParams::serialize(boost::archive::binary_iarchive &ar,  unsigned int version){
//		boost::serialization::void_cast_register<GeneralizedHawkesProcessMCMCParams,HawkesProcessMCMCParams>();
//		ar & boost::serialization::base_object<HawkesProcessMCMCParams>(*this);
//		ar & c_t_am_sd;
//		ar & c_t_am_t0;
//		ar & c_t_am_c0;
//		ar & c_h_am_sd;
//		ar & c_h_am_t0;
//		ar & c_h_am_c0;
//		ar & d_h_am_sd;  
//		ar & d_h_am_t0;
//		ar & d_h_am_c0;  
//		ar & t_w_am_sd;    
//		ar & t_w_am_t0;  
//		ar & t_w_am_c0;
}
	
void GeneralizedHawkesProcessMCMCParams::serialize(boost::archive::binary_oarchive &ar,  unsigned int version){
//		boost::serialization::void_cast_register<GeneralizedHawkesProcessMCMCParams,HawkesProcessMCMCParams>();
//		ar & boost::serialization::base_object<HawkesProcessMCMCParams>(*this);
//		ar & c_t_am_sd;
//		ar & c_t_am_t0;
//		ar & c_t_am_c0;
//		ar & c_h_am_sd;
//		ar & c_h_am_t0;
//		ar & c_h_am_c0;
//		ar & d_h_am_sd;  
//		ar & d_h_am_t0;
//		ar & d_h_am_c0;  
//		ar & t_w_am_sd;    
//		ar & t_w_am_t0;  
//		ar & t_w_am_c0;
}

