//double *back4 = make_1D_double(4);
//double *back3 = make_1D_double(2);
//double *back2 = make_1D_double(4);
//double *back = make_1D_double(4);

double growth_fac(double omegam, double omegal, double a);
double DADT(double a, double omegam, double omegal);
double DTDA3(double a, double omegam, double omegal);
double INTEGRATE(double fp1, double fp2, double a, double b, double dxinit, double eps);
double* RUNGE5VAR(double y, double dydx, double x, double htry, double eps, double yscale, double hnext, double fp1, double fp2);
double* RUNGE(double y, double dydx, double x, double h, double fp1, double fp2);
double sign(double x, double y);
double max(double x, double y);

// fitting form from Lahav et al. 1991, Carroll et al. 1992, Eisenstein & Hu 1997
double gfac_eisen(double z, double omegam, double omegal, double h, double Theta){
	
	// Theta is defined as TCMB/2.7
	double zeq = 2.5e4*omegam*h*h*pow(Theta,-4.);
	double g2 = omegam*pow(1.+z,3) + (1.-omegam-omegal)*pow(1.+z,2) + omegal;
	double omegam_z = omegam*pow(1.+z,3)/g2;
	double omegal_z = omegal/g2;
	
	//cout<<zeq<< endl;
	
	double D1 = (1./(1.+z))*5.*omegam_z/(2.*(pow(omegam_z,4./7.) - omegal_z + (1.+omegam_z/2.)*(1.+omegal_z/70.)));
	//double D0 = ((1.+zeq)/(1.+0.))*5.*omegam/(2.*(pow(omegam,4./7.) - omegal + (1.+omegam/2.)*(1.+omegal/70.)));
	
	return D1;//D0;
}

/*
     Calculate the linear growth factor for a given cosmology.
     Normalized so that D\approx a at small a.
     from Komatsu's webpage... with slide changes
     It gives D(z), while the growth factor is defined as g(z) = D(z)(1+z)
*/
double growth_fac(double omegam, double omegal, double a){
	
    double fact1, fact2;

    fact1 = INTEGRATE(omegam, omegal, 0., a, a/10., 1.e-8);
    fact2 = DADT(a, omegam, omegal)/a;

	return 2.5*omegam*fact1*fact2;//a;
   //return fact2;
}

double D(double omegam, double omegal, double a){
	
	double fact1, fact2;
	
	fact1 = INTEGRATE(omegam, omegal, 0., a, a/10., 1.e-8);
	fact2 = DADT(a, omegam, omegal)/a;
	
	return 2.5*omegam*fact1*fact2;//a;
	//return fact2;
}

/*
     Find da/dt given a, from the Friedmann Equation.  This is exact for
     any isotropic-metric cosmology consistent with General Relativity.
     Here, "t" is understood to be in units of the inverse Hubble constant
     (i.e. "t" = H0*t).

     Definitions for parameters are as in Peebles 1993, eqn (5.53).
*/
double DADT(double a, double omegam, double omegal){
   double omegak;

   omegak = 1. - omegam - omegal;

   return sqrt(omegam/a + omegal*a*a + omegak);
}

/*
     Find (dt/da)**3 given a, from the Friedmann Equation.  This is exact for
     any isotropic-metric cosmology consistent with General Relativity.
     Here, "t" is understood to be in units of the inverse Hubble constant
     (i.e. "t" = H0*t).

     This is an integrand, so it must contain two arguments, even if one is
     ignored.

     Definitions for parameters are as in Peebles 1993, eqn (5.53).
*/
double DTDA3(double a, double omegam, double omegal){
   double omegak, DTDA3;

   omegak = 1. - omegam - omegal;
   DTDA3 = pow(a/(omegam + omegal*a*a*a + omegak*a),1.5);

   return DTDA3;
}

/*
     Quadrature using fifth order Runge-Kutta with adaptive step size.
     Based on Press et al, Numerical Recipes in C, 2nd ed, pp 719-722.

     Runge-Kutta driver with adaptive stepsize control.  Integrate starting
     value y from a to b with accuracy eps, storing intermediate results in
     global variables.  dxinit should be set as a guessed first stepsize.

     Pass a second parameter to FUNC in fparm.
*/
double INTEGRATE(double fp1, double fp2, double a, double b, double dxinit, double eps){
	
    int maxsteps = 100000000;
    double x, dx, dxnext, y, dydx, yscale;
    int Nstep;
	double *back4 = make_1D_double(4);

    x = a;
    dx = dxinit;
    y = 0.;
    Nstep = 0;

    while((x-b)*(b-a) < 0. && Nstep < maxsteps){
       Nstep = Nstep + 1;
       dydx = DTDA3(x,fp1,fp2);
       // yscale is the scaling used to monitor accuracy.  This general-purpose
       // choice can be modified if need be.

       yscale = max(fabs(y) + fabs(dx*dydx), 1.e-12);
       if((x+dx-b)*(x+dx-a) > 0.)  dx = b - x;
 
       //cout<<"1 = "<<fabs(y) + fabs(dx*dydx)<<" yscale = "<<yscale<< endl;

       back4 = RUNGE5VAR(y,dydx,x,dx,eps,yscale,dxnext,fp1,fp2);

       y = back4[0];
       dydx = back4[1];
       x = back4[2];
       dx = back4[3];
       //cout<<"y = "<<y<<" dydx = "<<dydx<<" x = "<<x<<" dx = "<<dx<< endl;
       //while(true);
    }

//   cout<<"Nstep = "<<Nstep<<" y = "<<y<<" dx = "<<dx<<" x = "<<x<< endl;
//   while(true);

    if(Nstep >= maxsteps) printf("WARNING: failed to converge in INTEGRATE.");

    return y;
}

/*
     Fifth-order Runge-Kutta step with monitoring of local truncation error
     to ensure accuracy and adjust stepsize.  Input are the dependent
     variable y and its derivative dydx at the starting value of the
     independent variable x.  Also input are the stepsize to be attempted
     htry, the required accuracy eps, and the value yscale, against which the
     error is scaled.  On output, y and x are replaced by their new values.
     hdid is the stepsize that was actually accomplished, and hnext is the
     estimated next stepsize.  DERIVS is the user-supplied routine that
     computes right-hand-side derivatives.  The argument fparm is for an
     optional second argument to DERIVS (NOT integrated over).
*/
double* RUNGE5VAR(double y, double dydx, double x, double htry, double eps, double yscale, double hnext, double fp1, double fp2){
    double errmax, h, hold, htemp, xnew, yerr, ytemp;
    double safety, pgrow, pshrink, errcon;
	double *back2 = make_1D_double(4);
	double *back3 = make_1D_double(4);
   safety = 0.9;
   pgrow = -0.2;
   pshrink = -0.25;
   errcon = 1.89e-4;

   //cout<<"htry = "<<htry<<" x = "<<x<< endl;

   h = htry;                       // Set step to initial accuracy.
   errmax = 10.;
   while(errmax > 1.){
      back3 = RUNGE(y,dydx,x,h,fp1,fp2);
      yerr = back3[0];
      ytemp = back3[1];

      errmax = fabs(yerr/yscale)/eps;   // Scale rel. to required accuracy.
      if(errmax > 1.){                  // Trunc. error too large; reduce h
         htemp = safety*h*pow(errmax,pshrink);
         hold = h;
         //cout<<"h = "<<h<<" yerr = "<<yerr<<" yscale = "<<yscale<<" fabs(htemp) = "<<fabs(htemp)<<" 0.1*fabs(h) = "<<0.1*fabs(h)<< endl;
         h = sign(max(fabs(htemp),0.1*fabs(h)),h);  // >= factor of 10
         //cout<<"h = "<<h<< endl;
         xnew = x + h;
         if(xnew == x){
            printf("WARNING: ','Stepsize underflow in RUNGE5VAR().");
            h = hold;
            errmax = 0.;
         }
      }
   }

   // Step succeeded.  Compute estimated size of next step.
   if(errmax > errcon){
      hnext = safety*h*(pow(errmax,pgrow));
   }
   else{
      hnext = 5.*h;              // <= factor of 5 increase
   }
   x = x + h;

   y = ytemp;

   back2[0] = ytemp;
   back2[1] = dydx;
   back2[2] = x;
   back2[3] = hnext;

   //cout<<"errmax = "<<errmax<<" y = "<<ytemp<<" dydx = "<<dydx<<" x = "<<x<<" hnext = "<<hnext<< endl;

   return back2;
}

/*
     Given values for a variable y and its derivative dydx known at x, use
     the fifth-order Cash-Karp Runge-Kutta method to advance the solution
     over an interval h and return the incremented variables as yout.  Also
     return an estimate of the local truncation error in yout using the
     embedded fourth order method.  The user supplies the routine
     DERIVS(x,y,dydx), which returns derivatives dydx at x.
*/
double* RUNGE(double y, double dydx, double x, double h, double fp1, double fp2){
    double ak3, ak4, ak5, ak6, yout;
    double a2, a3, a4, a5, a6;
    double c1, c3, c4, c6, dc1, dc3, dc4, dc5, dc6;
	double *back = make_1D_double(4);

   a2  = 0.2;
   a3  = 0.3;
   a4  = 0.6;
   a5  = 1.;
   a6  = 0.875;
   c1  = 37./378.;
   c3  = 250./621.;
   c4  = 125./594.;
   c6  = 512./1771.;
   dc1 = c1 - 2825./27648.;
   dc3 = c3 - 18575./48384.;
   dc4 = c4 - 13525./55296.;
   dc5 = -277./14336.;
   dc6 = c6 - 0.25;

   ak3 = DTDA3(x+a3*h, fp1, fp2);
   ak4 = DTDA3(x+a4*h, fp1, fp2);
   ak5 = DTDA3(x+a5*h, fp1, fp2);
   ak6 = DTDA3(x+a6*h, fp1, fp2);

   // Estimate the fifth order value.

   yout = y + h*(c1*dydx + c3*ak3 + c4*ak4  + c6*ak6);

   // Estimate error as difference between fourth and fifth order

   back[0] = h*(dc1*dydx + dc3*ak3 + dc4*ak4 + dc5*ak5 + dc6*ak6);
   back[1] = yout;

   return back;
}

double sign(double x, double y){
   if(x <= 0 && y <= 0) return x;
   if(x > 0 && y > 0) return x;
   else return 0;
}

double max(double x, double y){
   if(x > y) return x;
   else return y;
}
