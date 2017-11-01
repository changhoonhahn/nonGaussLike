#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <fstream>
#include <iostream>

using namespace std;

double xitops(const int nxi, double* slin, double* xilin, double k, int ell){
/* 
	A sample program for computing the two-point correlation function, xi(r),
	from the linear or non-linear power spectrum in real or redshift space.
	The program generates spherically-averaged correlation functions.
	NOTE: the power spectrum is smoothed by a Gaussian with the width of 
	sigma_smooth = 1h^-1 Mpc, i.e., P(k) -> P(k)*exp(-k^2*sigma_smooth^2), 
	in order to suppress the aliasing (oscillations).   
*/

	double ds, factor, dummy;
	double xi, s;
	double slim = 700.;//10000.;

//--------------------------------------------------------------------------------//
/*
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
	pspec = gsl_interp_alloc(gsl_interp_linear, nxi);
	accel = gsl_interp_accel_alloc();
	gsl_interp_init(pspec, slin, xilin, nxi);
*/
	
	//dummyl_0 = tsin(idummy)/x1;
	//dummyl_2 = (3./(x1*x1) - 1.)*dummyl_0 - 3.*tcos(idummy)/(x1*x1);
	//dummyl_4 = (10./x1 - 105./(x1*x1*x1))*tcos(idummy)/x1 + 
	//		   (1. - 45./(x1*x1) + 105./(x1*x1*x1*x1))*dummyl_0;
	
//-------------------------------------------------------------------------------//

	s = slin[1];
	// index of root (it could be that we don't start with 0)
	// depends mainly on the initial k
	//ds = 1.;
	double sigma_smooth = 0.0004, x1;

	double ps = 0.;
	int i;
	for(i = 1; i < nxi; i++){

		ds = slin[i] - slin[i-1];
		x1 = k*slin[i];
		
		xi = xilin[i];//gsl_interp_eval(pspec, slin, xilin, s, accel); 
		//pks = linterp(nmp,ak_lin,pk_lin,k)*exp(-pow(k*sigma_smooth,2));
		//xi *= exp(-pow(sigma_smooth*slin[i],2));
		factor = slin[i]*slin[i]*ds*4.*M_PI;
		
		if(ell == 0) dummy = sin(x1)/x1;
		else if(ell == 2) dummy = (3./(x1*x1) - 1.)*sin(x1)/x1 - 3.*cos(x1)/(x1*x1);
		else if(ell == 4) dummy = (10./x1 - 105./(x1*x1*x1))*cos(x1)/x1 + (1. - 45./(x1*x1) + 105./(x1*x1*x1*x1))*sin(x1)/x1;
		
		//if(slin[i] < slim) ps += factor*xi*gsl_sf_bessel_jl(ell,k*slin[i]);
		if(slin[i] < slim) ps += factor*xi*dummy;
		
		//s += ds;
		
		//cout<<k<<" "<<xi<< endl;
	}
	//}while(s < slim && s < slin[nxi-1]);

    //gsl_interp_free(pspec);

    //cout<<k<<" "<<ps<< endl;

	if(ell == 0 || ell == 4) return ps;
	else if(ell == 2) return -ps;
}
