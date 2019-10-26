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


#include "plot_utils.hpp"
#include "stat_utils.hpp"

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, double start_y, double end_y){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xrange ["+std::to_string(start_x)+":"+std::to_string(end_x)+"]\nset yrange ["+std::to_string(start_y)+":"+std::to_string(end_y)+"]\n";

	gp<<"set xtics font \"Helvetica,15\"\n"; 
	gp<<"set ytics font \"Helvetica,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0) {
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}


}


void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, double start_y, double end_y, std::string xaxis_label, std::string yaxis_label){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xrange ["+std::to_string(start_x)+":"+std::to_string(end_x)+"]\nset yrange ["+std::to_string(start_y)+":"+std::to_string(end_y)+"]\n";

	gp<<"set xlabel '"+xaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set ylabel '"+yaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set xtics font \"Helvetica,15\"\n"; 
	gp<<"set ytics font \"Helvetica,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0) {
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}


}

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x, std::string xaxis_label, std::string yaxis_label){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xrange ["+std::to_string(start_x)+":"+std::to_string(end_x)+"]\n";
	gp<<"set xlabel '"+xaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set ylabel '"+yaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set xtics font \"Verdana,15\"\n"; 
	gp<<"set ytics font \"Verdana,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0){
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}

}

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs, double start_x, double end_x){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xrange ["+std::to_string(start_x)+":"+std::to_string(end_x)+"]\n";
	gp<<"set xtics font \"Helvetica,15\"\n"; 
	gp<<"set ytics font \"Helvetica,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0){
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}

}

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xtics font \"Helvetica,15\"\n"; 
	gp<<"set ytics font \"Helvetica,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0){
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}

}

void create_gnuplot_script(Gnuplot & gp, const std::string &output_file, const std::vector<std::string> & plot_specs,std::string xaxis_label, std::string yaxis_label){

	//create output file for the gnuplot-separate per type the sequences
	gp<<"set terminal png\n";
	gp<<"set output '"+output_file+"'\n";
	gp<<"set xlabel '"+xaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set ylabel '"+yaxis_label+"' font \"Helvetica,20\"\n";
	gp<<"set xtics font \"Verdana,15\"\n"; 
	gp<<"set ytics font \"Verdana,15\"\n"; 
	unsigned int K=plot_specs.size();
	if(K>0){
		std::string plot_cmnd="plot ";
		for(unsigned int k=0;k<K;k++){
			plot_cmnd+=" '-' " + plot_specs[k];
			if(k!=K-1)
				plot_cmnd+=",";
		}
		plot_cmnd+="\n";
		gp <<plot_cmnd;
	}

}

void plotTraces(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	//create steps of samples
	long unsigned int nof_samples=samples[0].size();
	long unsigned int nof_runs=samples.size();
		
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	//unsigned int nof_runs=samples.size();
	
	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		
		for(unsigned int r=0;r<nof_runs;r++){
			plot_specs.push_back("with lines lw 2 notitle");
		}
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		for(unsigned int r=0;r<nof_runs;r++){
			gp.send1d(boost::make_tuple(iter,samples[r]));
		}
	}
	
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		//write the headers
		csv_file<<"nof_samples";
		for(unsigned int r=0;r<nof_runs;r++){
			csv_file<<",trace_"<<std::to_string(r);
		}
		csv_file<<"\n";
		//write x and estimated value of the posterior per run
		//unsigned int nof_samples=p_x[0].size();
		for(long unsigned int s=0;s!=nof_samples;s++){
			csv_file<<iter[s];
			for (long unsigned int r=0;r!=nof_runs;r++){
				csv_file<<","<<samples[r][s];
			}
			csv_file<<"\n";
		}
		csv_file.close();
	}
}

void plotTrace(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	//create steps of samples
	unsigned int nof_samples=samples.size();
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	
	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines lw 2 notitle");
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		gp.send1d(boost::make_tuple(iter,samples));
	}
	
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		csv_file<<"sample, value \n";
	    for(unsigned int s=0;s!=nof_samples;s++){
	    	csv_file<<iter[s]<<","<<samples[s]<<std::endl;
	    }
		csv_file.close();
	}

	
}

void plotMeans(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	long unsigned int nof_samples=samples[0].size();
	long unsigned int nof_runs=samples.size();
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	std::vector<std::vector<double>> meanplot_v;
	meanplot_v.resize(nof_runs);
	//compute meanplots per run
	for(long unsigned int r=0;r<nof_runs;r++){	
		meanplot(samples[r],meanplot_v[r]);
	}
	

	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
	
		for(long unsigned int r=0;r<nof_runs;r++){
			plot_specs.push_back("with lines lw 2 notitle");
		}
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		for(long unsigned int r=0;r<nof_runs;r++){	
			gp.send1d(boost::make_tuple(iter,meanplot_v[r]));
		}
	}
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		//write the headers
		csv_file<<"nof_samples";
		for(unsigned int r=0;r<nof_runs;r++){
			csv_file<<",meanplot_"<<std::to_string(r);
		}
		csv_file<<"\n";
		//write x and estimated value of the posterior per run
		//unsigned int nof_samples=p_x[0].size();
		for(long unsigned int s=0;s!=nof_samples;s++){
			csv_file<<iter[s];
			for (long unsigned int r=0;r!=nof_runs;r++){
				csv_file<<","<<meanplot_v[r][s];
			}
			csv_file<<"\n";
		}
		csv_file.close();
	}
}

void plotMean(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	unsigned int nof_samples=samples.size();

	//create steps of samples
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	//compute meanplot
	std::vector<double> meanplot_v;
	meanplot(samples,meanplot_v);
	
	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines lw 2 notitle");
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		gp.send1d(boost::make_tuple(iter,meanplot_v));
	}
	
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		csv_file<<"nof samples, autocorrelation\n";
	    for(unsigned int s=0;s!=nof_samples;s++){
	    	csv_file<<iter[s]<<","<<meanplot_v[s]<<std::endl;
	    }
		csv_file.close();
	}	
}

void plotAutocorrelations(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	unsigned int nof_runs=samples.size();
	unsigned int nof_samples=samples[0].size();
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	std::vector<std::vector<double>> acf_v;
	acf_v.resize(nof_runs);
	
	for(unsigned int r=0;r<nof_runs;r++){
		acf(samples[r],acf_v[r]);
		//gp.send1d(boost::make_tuple(iter,acf_v));
	}
	
	if(write_png_file){
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		for(unsigned int r=0;r<nof_runs;r++){
			plot_specs.push_back("with lines lw 2 notitle");
		}
		std::string png_filename=filename+".png";
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		for(unsigned int r=0;r<nof_runs;r++){
			gp.send1d(boost::make_tuple(iter,acf_v[r]));
		}
	}
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		//write the headers
		csv_file<<"nof_samples";
		for(unsigned int r=0;r<nof_runs;r++){
			csv_file<<",acf_"<<std::to_string(r);
		}
		csv_file<<"\n";
		//write x and estimated value of the posterior per run
		//unsigned int nof_samples=p_x[0].size();
		for(unsigned int s=0;s!=nof_samples;s++){
			csv_file<<iter[s];
			for (unsigned int r=0;r!=nof_runs;r++){
				csv_file<<","<<acf_v[r][s];
			}
			csv_file<<"\n";
		}
		
		csv_file.close();
	}
}

void plotAutocorrelation(const std::string &filename, const boost::numeric::ublas::vector<double> & samples,bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	unsigned int nof_samples=samples.size();
	//compute autocorrelation
	std::vector<double> acf_v;
	acf(samples,acf_v);
	//create steps of samples
	std::vector<double> iter(nof_samples);
	std::iota( std::begin(iter), std::end(iter),1);
	
	//unsigned int runs=samples.size();
	if(write_png_file){
		Gnuplot gp;
		std::string png_filename=filename+".png";
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines lw 2 notitle");
		create_gnuplot_script(gp,png_filename,plot_specs,1,nof_samples);
		gp.send1d(boost::make_tuple(iter,acf_v));
	}
	
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		csv_file<<"nof samples, autocorrelation\n";
	    for(unsigned int s=0;s!=nof_samples;s++){
	    	csv_file<<iter[s]<<","<<acf_v[s]<<std::endl;
	    }
		csv_file.close();
	}
}

void plotDistributions(const std::string &filename, const std::vector<boost::numeric::ublas::vector<double>> & samplesv,bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	//compute the distirbutions 
	unsigned int nof_runs=samplesv.size();
	boost::numeric::ublas::vector<double> x;
	std::vector<boost::numeric::ublas::vector<double>> p_x;
	p_x.resize(nof_runs);
	for(unsigned int r=0;r<nof_runs;r++){
		gkr(samplesv[r],x,p_x[r], NOF_PLOT_POINTS);
	}

	//plot the distributions
	if(write_png_file){
		std::string png_filename=filename+".png";
		Gnuplot gp;
		std::vector<std::string> plot_specs;
		for(unsigned int r=0;r<nof_runs;r++){
			plot_specs.push_back("with lines lw 2 notitle");
		}
		create_gnuplot_script(gp,png_filename,plot_specs);
		for(unsigned int r=0;r<nof_runs;r++){
			//gkr(samplesv[r],x,p_x, NOF_PLOT_POINTS);
			gp.send1d(boost::make_tuple(x,p_x[r]));
		}
	}
	if(write_csv_file){
		//open csv file
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		//write the headers
		csv_file<<"x";
		for(unsigned int r=0;r<nof_runs;r++){
			csv_file<<",p_"<<std::to_string(r);
		}
		csv_file<<"\n";
		//write x and estimated value of the posterior per run
		unsigned int nof_samples=p_x[0].size();
		for(unsigned int s=0;s!=nof_samples;s++){
			csv_file<<x[s];
			for (unsigned int r=0;r!=nof_runs;r++){
				csv_file<<","<<p_x[r][s];
			}
			csv_file<<"\n";
		}
		
		csv_file.close();
	}
	
}

void plotDistribution(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	boost::numeric::ublas::vector<double> x;
	boost::numeric::ublas::vector<double> p_x;
	gkr(samples,x,p_x,NOF_PLOT_POINTS);
	//find the posterior mode
    auto mode_p_x_iter=boost::max_element(p_x);
    std::vector<double> mode_p_x;
    mode_p_x.push_back(*mode_p_x_iter);
    std::vector<double> mode_x;
	mode_x.push_back(x(std::distance(p_x.begin(), mode_p_x_iter)));
	//get the posterior mean
	accumulator_set<double, stats<tag::mean>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));
	boost::numeric::ublas::vector<double> mean_x(1);
	mean_x(0)=mean(acc);
	boost::numeric::ublas::vector<double> mean_p_x;
	gkr(samples,mean_x,mean_p_x);

	if(write_png_file){
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines ls 1 lw 2 notitle");
		plot_specs.push_back("with points pt 7 ps 2 title 'Posterior Mode'");
		plot_specs.push_back("with points pt 3 ps 1 title 'Posterior Samples'");
		plot_specs.push_back("with points pt 7 ps 2 title 'Posterior Mean'");
		Gnuplot gp;
		std::string png_filename=filename+".png";
		//create_gnuplot_script(gp,filename,plot_specs,x(0),x(NOF_PLOT_POINTS-1));
		create_gnuplot_script(gp,png_filename,plot_specs,"x","p(x)");
		gp.send1d(boost::make_tuple(x,p_x));
		gp.send1d(boost::make_tuple(mode_x,mode_p_x));
		std::vector<double> samples_y(samples.size(),0.0);
		gp.send1d(boost::make_tuple(samples,samples_y));
		gp.send1d(boost::make_tuple(mean_x,mean_p_x));
	}

	if(write_csv_file){
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		csv_file<<"x,p_x,mode_x,mode_p_x,mean_x,mean_p_x"<<std::endl;
		csv_file<<x[0]<<","<<p_x[0]<<","<<mode_x[0]<<","<<mode_p_x[0]<<","<<mean_x[0]<<","<<mean_p_x[0]<<std::endl;
		
		for(unsigned int i=1;i<p_x.size();i++){
			csv_file<<x[i]<<","<<p_x[i]<<std::endl;
		}
		csv_file.close();
	}	
}

void plotDistribution(const std::string &filename, const boost::numeric::ublas::vector<double> & samples, double true_value, bool write_png_file, bool write_csv_file){
	
	if(!write_png_file && !write_csv_file)
		return;
	
	//gaussian kernel density estimation 
	boost::numeric::ublas::vector<double> x;
	boost::numeric::ublas::vector<double> p_x;
	gkr(samples,x,p_x,NOF_PLOT_POINTS);
	
	//find the posterior mode
    auto mode_p_x_iter=boost::max_element(p_x);
    std::vector<double> mode_p_x;
    mode_p_x.push_back(*mode_p_x_iter);
    std::vector<double> mode_x;
	mode_x.push_back(x(std::distance(p_x.begin(), mode_p_x_iter)));
	//get the posterior mean
	accumulator_set<double, stats<tag::mean>> acc;
	for_each(samples.begin(),samples.end(), bind<void>(ref(acc),std::placeholders::_1));
	boost::numeric::ublas::vector<double> mean_x(1);
	mean_x(0)=mean(acc);
	boost::numeric::ublas::vector<double> mean_p_x;
	gkr(samples,mean_x,mean_p_x);
	//plot the true value
	boost::numeric::ublas::vector<double> value_x(1);
	value_x(0)=true_value;
	boost::numeric::ublas::vector<double> value_p_x;
	gkr(samples,value_x,value_p_x);




	if(write_csv_file){
		std::ofstream csv_file;
		std::string csv_filename=filename+".csv";
		csv_file.open(csv_filename);
		csv_file<<"x,p_x,mode_x,mode_p_x,mean_x,mean_p_x,true_x,true_p_x"<<std::endl;
		csv_file<<x[0]<<","<<p_x[0]<<","<<mode_x[0]<<","<<mode_p_x[0]<<","<<mean_x[0]<<","<<mean_p_x[0]<<","<<value_x[0]<<","<<value_p_x[0]<<std::endl;
		for(unsigned int i=1;i<p_x.size();i++){
			csv_file<<x[i]<<","<<p_x[i]<<std::endl;
		}
		csv_file.close();
	}
	
	if(write_png_file){
		std::string png_filename=filename+".png";
		std::vector<std::string> plot_specs;
		plot_specs.push_back("with lines ls 1 lw 2 notitle");
		plot_specs.push_back("with points pt 7 ps 2 title 'Posterior Mode'");
		plot_specs.push_back("with points pt 3 ps 1 title 'Posterior Samples'");
		plot_specs.push_back("with points pt 7 ps 2 title 'Posterior Mean'");
		plot_specs.push_back("with points pt 7 ps 2 title 'True'");
		Gnuplot gp;
		//create_gnuplot_script(gp,filename,plot_specs,x(0),x(NOF_PLOT_POINTS-1));
		create_gnuplot_script(gp,png_filename,plot_specs,"x","p(x)");
		gp.send1d(boost::make_tuple(x,p_x));
		gp.send1d(boost::make_tuple(mode_x,mode_p_x));
		std::vector<double> samples_y(samples.size(),0.0);
		gp.send1d(boost::make_tuple(samples,samples_y));
		gp.send1d(boost::make_tuple(mean_x,mean_p_x));
		gp.send1d(boost::make_tuple(value_x,value_p_x));
	}
	
}
