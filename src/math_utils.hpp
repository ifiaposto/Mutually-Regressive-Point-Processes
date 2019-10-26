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

#ifndef MATH_UTILS_HPP_
#define MATH_UTILS_HPP_


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
#include <random>
#include <iostream>
#include <math.h>

#define EPS 1e-10
#define INF 1e10

using namespace boost::accumulators;
using namespace std;


const unsigned int  MONTE_CARLO_NOF_SAMPLES=100; //number of samples for the monte carlo integration
const unsigned int  MONTE_CARLO_NOF_THREADS=2; //nof threads for the monte carlo integration


/****************************************************************************************************************************************************************************
 * utility functions
******************************************************************************************************************************************************************************/

void linspace(const boost::numeric::ublas::vector<double> &samples, boost::numeric::ublas::vector<double> &points, unsigned int nofxs);

//it returns the value of the sigmoid function (if l=true) (or 1-signoid, if l=false) at x, according to the label l of x
double sigmoid(double x, bool l);

//it computes the product of the elements of the vector x
double gsl_mul_all(gsl_vector  * x);

//it computes the determinant of the matrix from its cholesk decomposition L (as the product of square of the diagonal elements)
double gsl_cholesky_det(gsl_matrix *L);


struct MonteCarloIntegralThreadParams{
	unsigned int thread_id; //the id of the thread which executes the subcomputation
	unsigned int nof_mcmc_iters; //number of monte carlo iterations to be computed by the thread
	double &mc_sum; //variable to store the partial sum of the monte carlo iterations
	std::function<double(double)> f; //function to be integrated
	double t0; //low bound of the integration interval
	double t1; //upper bound of the integration interval
	MonteCarloIntegralThreadParams(unsigned int i, unsigned int n, double &s, std::function<double(double)> f2, double t00, double t11): thread_id(i), nof_mcmc_iters(n), mc_sum(s), f(f2), t0(t00), t1(t11) {};
};


double monte_carlo_integral(std::function<double(double)> f, double t0, double t1, unsigned int nof_samples=MONTE_CARLO_NOF_SAMPLES, unsigned int nof_threads=MONTE_CARLO_NOF_THREADS);

/****************************************************************************************************************************************************************************
 * activation functions
******************************************************************************************************************************************************************************/
class ActivationFunction{
	
	unsigned int nofp;
	
	public:
		ActivationFunction(unsigned int nofp2);
		ActivationFunction();
		virtual double op(double a, double x, double b) const=0;
		virtual void print(std::ostream &file) const=0;
	private:
		friend class boost::serialization::access;
		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);


};
////it returns the value of the sigmoid function at x
double sigmoid(double x);

class GeneralSigmoid: public ActivationFunction{
	public:
		double x0;
		double d0;
		double op(double a, double x, double b) const override;
		GeneralSigmoid(double x, double d);
		GeneralSigmoid();
		void print(std::ostream &file) const override;
	private:
		friend class boost::serialization::access;
		void serialize( boost::archive::text_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::text_iarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_oarchive &ar, const unsigned int version);
		void serialize( boost::archive::binary_iarchive &ar, const unsigned int version);

	
};

#endif /* MATH_UTILS_HPP_ */
