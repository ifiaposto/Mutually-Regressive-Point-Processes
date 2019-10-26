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
#ifndef PLOT_UTILS_HPP_
#define PLOT_UTILS_HPP_

#include <boost/numeric/ublas/vector.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <string>

#include "gnuplot-iostream/gnuplot-iostream.h"

#define NOF_PLOT_POINTS  100000

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, double start_y, double end_y, std::string xaxis_label, std::string yaxis_label);

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, double start_y, double end_y);

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x);

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, std::string xaxis_label, std::string yaxis_label);

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs);

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, std::string xaxis_label, std::string yaxis_label);

void plotTraces(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> &samples, bool write_png_file=true, bool write_csv_file=false);

void plotTrace(const std::string &filename, const boost::numeric::ublas::vector<double> &samples, bool write_png_file=true, bool write_csv_file=false);

void plotMeans(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samples, bool write_png_file=true, bool write_csv_file=false);

void plotMean(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, bool write_png_file=true, bool write_csv_file=false);

void plotAutocorrelations(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samples, bool write_png_file=true, bool write_csv_file=false);

void plotAutocorrelation(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, bool write_png_file=true, bool write_csv_file=false);

//it computes by gaussian kernel density estimationand plots multiple distributions 
void plotDistributions(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samplesv, bool write_png_file=true, bool write_csv_file=false);
//it computes by gaussian kernel density estimationa nd plots from samples a distribution, by denoting its mean and mode 
void plotDistribution(const std::string &filename, const boost::numeric::ublas::vector<double> & samplesv, bool write_png_file=true, bool write_csv_file=false);
//it computes by gaussian kernel density estimationa nd plots from samples a distribution, by denoting its mean, mode and a specific value 
void plotDistribution(const std::string &filename, const boost::numeric::ublas::vector<double> & samplesv, double value, bool write_png_file=true, bool write_csv_file=false);

#endif /* PLOT_UTILS_HPP_ */
