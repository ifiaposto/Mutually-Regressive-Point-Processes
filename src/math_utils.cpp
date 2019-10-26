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

#include "math_utils.hpp"

BOOST_SERIALIZATION_ASSUME_ABSTRACT(ActivationFunction)
BOOST_CLASS_EXPORT(GeneralSigmoid)

/****************************************************************************************************************************************************************************
 * utility functions
******************************************************************************************************************************************************************************/
void linspace(const boost::numeric::ublas::vector<double> &samples, boost::numeric::ublas::vector<double> &points, unsigned int nofxs){
	accumulator_set<double, stats<tag::mean,tag::variance,tag::min,tag::max>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));

	//find the range for which the posterior will be plotted;[min(min_of_samples,mean-2*std), max(max_of_samples,mean+2*std)]
	double mean_samples=mean(acc);
	double sigma_samples=sqrt(variance(acc));
	double max_sample=extract_result<tag::max>(acc);

	double start_x=0.0;
	double end_x=std::max(max_sample, mean_samples+2*sigma_samples);
	double step=(end_x-start_x)/(nofxs-1);

	boost::numeric::ublas::scalar_vector<double> start_x_v(nofxs,start_x);
	boost::numeric::ublas::vector<double> step_x_v(nofxs);
	std::iota(step_x_v.begin(),step_x_v.end(), 0);
	if(points.size()<nofxs)
		points.resize(nofxs);
	points=start_x_v+step_x_v*step;
}


double gsl_mul_all(gsl_vector * x){
	double prod=1.0;
	for(unsigned int i=0;i<x->size;i++){
		prod*=gsl_vector_get(x,i);
	}
	return prod;
}

double gsl_cholesky_det(gsl_matrix *L){
	gsl_vector_view diag=gsl_matrix_diagonal (L);
	double c=gsl_mul_all(&diag.vector);
	return c*c;
}

static void * monte_carlo_integral_(void *p){
	
	std::unique_ptr<MonteCarloIntegralThreadParams> params(static_cast< MonteCarloIntegralThreadParams * >(p));
	unsigned int nof_mcmc_iters=params->nof_mcmc_iters; //number of monte carlo iterations to be computed by the thread
	double &mc_sum=params->mc_sum; //variable to store the partial sum of the monte carlo iterations
	std::function<double(double)> f=params->f; //function to be integrated
	double t0=params->t0; //low bound of the integration interval
	double t1=params->t1; //upper bound of the integration interval
	
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> uniform_distribution(t0,t1);//this is approximation, oo is the right limit
	mc_sum=0.0;
	for(unsigned int s=0;s<nof_mcmc_iters;s++){
	
			double x=uniform_distribution(gen);
			mc_sum+=f(x);
	}
	return 0;
}

double monte_carlo_integral(std::function<double(double)> f, double t0, double t1, unsigned int nof_samples, unsigned int nof_threads){
	if(t1<=t0){
		std::cerr<<"wrong integration intervals\n";
		std::cerr<<"t0 "<<t0<<std::endl;
		std::cerr<<"t1 "<<t0<<std::endl;
	}
	
	pthread_t mc_threads[nof_threads];
	unsigned int nof_samples_thread[nof_threads];
	double *monte_carlo_ps=(double*) calloc (nof_threads,sizeof(double));
	//distribute the events across the threads
	unsigned int nof_samples_q=nof_samples/nof_threads;
	unsigned int nof_samples_m=nof_samples%nof_threads;
	for(unsigned int t=0;t<nof_threads;t++){
		nof_samples_thread[t]=nof_samples_q;
		if(t<nof_samples_m)
			nof_samples_thread[t]++;
	}
	
	//create threads
	for(unsigned int thread_id=0;thread_id<nof_threads;thread_id++){
		if(!nof_samples_thread[thread_id])
			break;
	
		MonteCarloIntegralThreadParams* p;
		//use the history of realized and thinned events which correspond to type k2
		p= new MonteCarloIntegralThreadParams(thread_id, nof_samples_thread[thread_id], monte_carlo_ps[thread_id], f, t0, t1);
		int rc = pthread_create(&mc_threads[thread_id], NULL, monte_carlo_integral_, (void *)p);
		if (rc){
			 std::cerr<< "Error:unable to create thread," << rc << std::endl;
			 exit(-1);
		}
	}
	//wait for all the partial sums to be computed
	for (unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t])
			pthread_join (mc_threads [t], NULL);
	}
	
	double monte_carlo_sum=0.0; 
	for(unsigned int t = 0; t <nof_threads; t++){
		if(nof_samples_thread[t])
			monte_carlo_sum+=monte_carlo_ps[t];//TODO, TODO, TODO: uncomment this
	}
	monte_carlo_sum*=(t1-t0)/nof_samples;
	return monte_carlo_sum;
	
}



/****************************************************************************************************************************************************************************
 * activation functions
******************************************************************************************************************************************************************************/

double sigmoid(double x){
	return 1/(1+exp(-x));
}
ActivationFunction::ActivationFunction()=default;

ActivationFunction::ActivationFunction(unsigned int nofp2): nofp{nofp2}{};

void ActivationFunction::serialize( boost::archive::text_oarchive &ar, unsigned int ){
	ar & nofp;	
};

void ActivationFunction::serialize( boost::archive::text_iarchive &ar, unsigned int ){
	ar & nofp;	
};


void ActivationFunction::serialize( boost::archive::binary_oarchive &ar, unsigned int ){
	ar & nofp;	
};


void ActivationFunction::serialize( boost::archive::binary_iarchive &ar, unsigned int ){
	ar & nofp;	
};

GeneralSigmoid::GeneralSigmoid()=default;

GeneralSigmoid::GeneralSigmoid(double x, double d): ActivationFunction(2), x0{x}, d0{d} {};

void GeneralSigmoid::serialize( boost::archive::text_oarchive &ar, unsigned int ){
    boost::serialization::void_cast_register<GeneralSigmoid,ActivationFunction>();
    ar & boost::serialization::base_object<ActivationFunction>(*this);
	ar & x0;
	ar & d0;	
};

void GeneralSigmoid::serialize( boost::archive::text_iarchive &ar, unsigned int ){
    boost::serialization::void_cast_register<GeneralSigmoid,ActivationFunction>();
    ar & boost::serialization::base_object<ActivationFunction>(*this);
	ar & x0;
	ar & d0;	
};

void GeneralSigmoid::serialize( boost::archive::binary_oarchive &ar, unsigned int ){
    boost::serialization::void_cast_register<GeneralSigmoid,ActivationFunction>();
    ar & boost::serialization::base_object<ActivationFunction>(*this);
	ar & x0;
	ar & d0;	
};

void GeneralSigmoid::serialize( boost::archive::binary_iarchive &ar, unsigned int ){
    boost::serialization::void_cast_register<GeneralSigmoid,ActivationFunction>();
    ar & boost::serialization::base_object<ActivationFunction>(*this);
	ar & x0;
	ar & d0;	
};

double GeneralSigmoid::op(double a, double x, double b) const{
	return a/(1+exp(-d0*(x-x0)))+b;
}


void GeneralSigmoid::print(std::ostream &file) const {
	
	file<<"general sigmoid\n";
	file<<"a/(1+exp(-d0*(x-x0)))+b";
	file<<"d0 "<<d0<<std::endl;
	file<<"x0 "<<x0<<std::endl;
	
	
}

double sigmoid(double x, bool l){
	double e_x=exp(x);
	return (l?e_x:1.0)/(1+e_x);
}

double linear(double a, double x, double b){
	return a*x+b;
}

double sqrtlinear(double a, double x, double b){
	return a*sqrt(x)+b;
}

double squarelinear(double a, double x, double b){
	return a*x*x+b;
}


