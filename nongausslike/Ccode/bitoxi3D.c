#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <fstream>
#include <iostream>

using namespace std;

double pstoxi(const int nps, double* klin, double* pslin, double ra, int ell){
/* 
	A sample program for computing the two-point correlation function, xi(r),
	from the linear or non-linear power spectrum in real or redshift space.
	The program generates spherically-averaged correlation functions.
	NOTE: the power spectrum is smoothed by a Gaussian with the width of 
	sigma_smooth = 1h^-1 Mpc, i.e., P(k) -> P(k)*exp(-k^2*sigma_smooth^2), 
	in order to suppress the aliasing (oscillations).   
*/

	double dk, factor, dummy;
	double ps, k;
	
	double log_k[nps], log_ps[nps];
	double klim = 10;
	/*
	int n = 0;
	for(int i = 0; i < nps; i++){
		
		if(pslin[i] > 1.e-8 && klin[i] < klim && i == n){
			
			log_k[n] = log10(klin[i]);
			log_ps[n] = log10(pslin[i]);

			n++;
		}
		else if(pslin[i] < 1.e-8){
		
			log_k[n] = log10(klin[i]);
			log_ps[n] = log10(1.e-8);
			
			n++;
		}
	}
	n--;

//--------------------------------------------------------------------------------//

	// interpolation
	gsl_interp *pspec;
	gsl_interp_accel *accel;
	// options
	// 1. gsl_interp_polynomial
	// 2. gsl_interp_linear
	// 3. gsl_interp_cspline
	// 4. gsl_interp_cspline_periodic
	// 5. gsl_interp_akima
	// 6. gsl_interp_akima_periodic
	pspec = gsl_interp_alloc(gsl_interp_linear, n);
	accel = gsl_interp_accel_alloc();
	gsl_interp_init(pspec, log_k, log_ps, n);
*/
//-------------------------------------------------------------------------------//

	k = klin[1];
	// index of root (it could be that we don't start with 0)
	// depends mainly on the initial k
	dk = 0.005;
	double sigma_smooth = 0.5, x1;

	double xi = 0.;
	int i;
	for(i = 0; i < nps; i++){
		
		dk = klin[i] - klin[i-1];
		ps = pslin[i];//pow(10.,gsl_interp_eval(pspec, log_k, log_ps, log10(k), accel)); 
		//pks = linterp(nmp,ak_lin,pk_lin,k)*exp(-pow(k*sigma_smooth,2));
		//ps *= exp(-pow(klin[i]*sigma_smooth,2));
		
		x1 = klin[i]*ra;
		
		if(ell == 0) dummy = sin(x1)/x1;
		else if(ell == 2) dummy = (3./(x1*x1) - 1.)*sin(x1)/x1 - 3.*cos(x1)/(x1*x1);
		else if(ell == 4) dummy = (10./x1 - 105./(x1*x1*x1))*cos(x1)/x1 + (1. - 45./(x1*x1) + 105./(x1*x1*x1*x1))*sin(x1)/x1;
		
		factor = klin[i]*klin[i]*dk/(2.*M_PI*M_PI);
		//xi += factor*ps*gsl_sf_bessel_jl(ell,klin[i]*ra);
		xi += factor*ps*dummy;

		//cout<<klin[i]<<" "<<xi<< endl;
		
		//k += dk;
	}
	//}while(k < klim-1 && k < klin[n-1]);

    //gsl_interp_free(pspec);
	if(ell == 0 || ell == 4) return xi;
	else if(ell == 2) return -xi;
}
