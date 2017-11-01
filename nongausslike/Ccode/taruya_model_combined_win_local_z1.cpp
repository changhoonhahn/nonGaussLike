#include <math.h>
#include <iostream>
#include <cstdlib>

#include "linterp.c"
#include "pstoxi2.c"
#include "xitops2.c"

using namespace std;

//********************************************************************************
// Function DXX_MPCH_ARR - returns comoving distance dxx(z) in Mpc/h for an 
// array of redshifts (saves a huge amount of time).
extern "C" double* taruya_model_combined_win_local(const int mubins, const int binrange1, const int binrange2, 
													  const int binrange3, const int maxbin1, double *x, 
													  double alpha_perp, double alpha_para, double fsig8, double b1NGCsig8, double b1SGCsig8, 
													  double b2NGCsig8, double b2SGCsig8, double N_NGC, double N_SGC, double sigmav_NGC, double sigmav_SGC){
	
	// input files
	char AAinput[200] = "/Users/chang/projects/clustools/input/pkstd_corr2_tree_0_38.dat";
	char BBinput[200] = "/Users/chang/projects/clustools/input/pkstd_corr_tree_0_38.dat";
	
	char ddinput[200] = "/Users/chang/projects/clustools/input/pk_RegPT_spectrum_fid_0_38_0_82_dd.dat";
	char dtinput[200] = "/Users/chang/projects/clustools/input/pk_RegPT_spectrum_fid_0_38_0_82_dt.dat";
	char ttinput[200] = "/Users/chang/projects/clustools/input/pk_RegPT_spectrum_fid_0_38_0_82_tt.dat";
	
	char psshuninput[200] = "/Users/chang/projects/clustools/input/fid_0_38_shun_0_82_matterpower.dat";
	
	// if the power spectrum is not kept fixed 
	// sigma_8(z) should be calculated here
	double fudge = 1.03;
	double Dz = 0.81905*fudge;  //z1
	//double Dz = 0.76644*fudge;  //z2
	//double Dz = 0.729246*fudge; //z3
	double sig8z_fid = 0.82*Dz;
	double betaNGC = fsig8/b1NGCsig8;
	double betaSGC = fsig8/b1SGCsig8;
	double b1NGC = b1NGCsig8/sig8z_fid;
	double b1SGC = b1SGCsig8/sig8z_fid;
	double b2NGC = b2NGCsig8/sig8z_fid;
	double b2SGC = b2SGCsig8/sig8z_fid;
	double FAP = alpha_para/alpha_perp;
	double rs_ratio = 0.98726;
	int i, j;
	
	double bs2NGC = -4.*(b1NGC - 1.)/7.;
	double b3nlNGC = 32.*(b1NGC - 1.)/315.;
	double bs2SGC = -4.*(b1SGC - 1.)/7.;
	double b3nlSGC = 32.*(b1SGC - 1.)/315.;
	
	const int totbinrange = binrange1+binrange2+binrange3;
	
	FILE *stream;
	
//---------------------------------------------------------------------------------------//
	
	double Nmodes_SGC[120][100], Nmodes_norm_SGC[120];
	for(i = 0; i < 120; i++){
		Nmodes_norm_SGC[i] = 0.;
	}
	float dummyk, dummymu, dummy;
	char modesfiles_SGC[200] = "/Users/chang/projects/clustools/input/modes_DR12_patchy_SGC_z1_recon_modes_90_180_102_120.dat";
	//char modesfiles_SGC[200] = "/Users/chang/projects/clustools/input/modes_DR12_patchy_SGC_z1_recon_modes_180_360_204_120.dat";
	//char modesfiles_SGC[200] = "/Users/fbeutler/github/clustools/input/modes_DR12_patchy_SGC_z2_recon_modes_150_416_234_120.dat";
	//char modesfiles_SGC[200] = "/Users/fbeutler/github/clustools/input/modes_DR12_patchy_SGC_z3_recon_modes_90_245_136_120.dat";
	
	stream = fopen(modesfiles_SGC, "r");
	if(stream == NULL) {
		printf("Cannot open file modesfile. (modes SGC)\n");
		exit(1);
	}
	int n = 0;
	int n1 = 0;
	while((fscanf(stream, "%f %f %f\n", 
				  &dummyk, &dummymu, &dummy)) != EOF){
		
		//printf("dummy = %f\n", dummy);
		
		Nmodes_SGC[n1][n] = dummy;
		Nmodes_norm_SGC[n1] += dummy;
		
		n++;
		if(n == 100){
			n = 0;
			n1++;
		}
	}
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double Nmodes_NGC[120][100], Nmodes_norm_NGC[120];
	for(i = 0; i < 120; i++){
		Nmodes_norm_NGC[i] = 0.;
	}
	char modesfiles_NGC[200] = "/Users/chang/projects/clustools/input/modes_DR12_patchy_NGC_z1_recon_modes_125_230_126_120.dat";
	//char modesfiles_NGC[200] = "/Users/fbeutler/github/clustools/input/modes_DR12_patchy_NGC_z2_recon_modes_140_270_150_120.dat";
	//char modesfiles_NGC[200] = "/Users/fbeutler/github/clustools/input/modes_DR12_patchy_NGC_z3_recon_modes_175_320_180_120.dat";
	
	stream = fopen(modesfiles_NGC, "r");
	if(stream == NULL) {
		printf("Cannot open file modesfile. (modes NGC)\n");
		exit(1);
	}
	n = 0;
	n1 = 0;
	while((fscanf(stream, "%f %f %f\n", 
				  &dummyk, &dummymu, &dummy)) != EOF){
		
		//printf("dummy = %f\n", dummy);
		
		Nmodes_NGC[n1][n] = dummy;
		Nmodes_norm_NGC[n1] += dummy;
		
		n++;
		if(n == 100){
			n = 0;
			n1++;
		}
	}
	fflush(stdout);
	fclose(stream);	

//---------------------------------------------------------------------------------------//
//----------------------------------- read in file  -------------------------------------//
//---------------------------------------------------------------------------------------//
	
	const int ns = 1000;
	
	char cdummy[200];
	double BBk[ns], BB111[ns], BB112[ns], BB122[ns];
	double BB211[ns], BB212[ns], BB222[ns], BB312[ns];
	double BB322[ns], BB422[ns];
	double kdummy, mudummy, dummy111, dummy112, dummy122, dummy211, dummy212, dummy222;
	double dummy312, dummy322, dummy422;
	
	stream = fopen(BBinput, "r");
	if(stream == NULL) {
		printf("Cannot open file BB. (ps-grid)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
				  &kdummy, &dummy111, &dummy112, &dummy122, &dummy211, 
				  &dummy212, &dummy222, &dummy312, &dummy322, &dummy422)) != EOF){
		
		BBk[n] = kdummy;
		BB111[n] = dummy111;
		BB112[n] = dummy112;
		BB122[n] = dummy122;
		BB211[n] = dummy211;
		BB212[n] = dummy212;
		BB222[n] = dummy222;
		BB312[n] = dummy312;
		BB322[n] = dummy322;
		BB422[n] = dummy422;
		
		if(BBk[n] < 0.3) n++;
	}
	n--;
	int nBB = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double AAk[ns], AA11[ns], AA12[ns], AA22[ns], AA23[ns], AA33[ns];
	double dummy11, dummy12, dummy22, dummy23, dummy33;
	
	stream = fopen(AAinput, "r");
	if(stream == NULL) {
		printf("4Cannot open file. (ps-grid)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %f %lf %lf %lf\n", 
				  &kdummy, &dummy11, &dummy12, &dummy22, &dummy23, &dummy33)) != EOF){
		
		AAk[n] = kdummy;
		AA11[n] = dummy11;
		AA12[n] = dummy12;
		AA22[n] = dummy22;
		AA23[n] = dummy23;
		AA33[n] = dummy33;
		
		if(AAk[n] < 0.3) n++;
	}
	n--;
	int nAA = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double kdd[ns], pkdd[ns];
	double psdummy;
	
	stream = fopen(ddinput, "r");
	if(stream == NULL) {
		printf("5Cannot open file. (ps-grid)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf\n", 
				  &kdummy, &dummy, &dummy, &psdummy, &dummy)) != EOF){
		
		kdd[n] = kdummy;
		pkdd[n] = psdummy/(Dz*Dz);
		
		if(kdd[n] < 0.3) n++;
	}
	n--;
	int ndd = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double kdt[ns], pkdt[ns];
	
	stream = fopen(dtinput, "r");
	if(stream == NULL) {
		printf("6Cannot open file. (ps-grid)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf\n", 
				  &kdummy, &dummy, &dummy, &psdummy, &dummy)) != EOF){
		
		kdt[n] = kdummy;
		pkdt[n] = psdummy/(Dz*Dz);
		
		if(kdt[n] < 0.3) n++;
	}
	n--;
	int ndt = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double ktt[ns], pktt[ns];
	
	stream = fopen(ttinput, "r");
	if(stream == NULL) {
		printf("7Cannot open file. (ps-grid)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf\n", 
				  &kdummy, &dummy, &dummy, &psdummy, &dummy)) != EOF){
		
		ktt[n] = kdummy;
		pktt[n] = psdummy/(Dz*Dz);
		
		if(ktt[n] < 0.3) n++;
	}
	n--;
	int ntt = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//
	
	double kshun[ns], pkb2delta[ns], pkb2theta[ns], pkb22[ns];
	double pkbs2delta[ns], sigma3pkterm[ns], pkb2s2[ns], pkbs22[ns];
	double pkbs2theta[ns];
	double dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8;
	
	stream = fopen(psshuninput, "r");
	if(stream == NULL) {
		printf("8Cannot open file. (ps_shun)\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 
				  &kdummy, &dummy, &dummy, &dummy, &dummy, &dummy1, &dummy2, &dummy3, &dummy4, &dummy5, &dummy6, &dummy7, &dummy8, &dummy, &dummy, &dummy, &dummy)) != EOF){
		
		kshun[n] = kdummy;
		pkb2delta[n] = dummy1;
		pkb2theta[n] = dummy2;
		pkb22[n] = dummy3;
		
		pkbs2delta[n] = dummy4;
		pkbs2theta[n] = dummy5;
		pkb2s2[n] = dummy6;
		pkbs22[n] = dummy7;
		sigma3pkterm[n] = dummy8;
		
		if(kshun[n] < 0.3) n++;
	}
	n--;
	int nshun = n;
	fflush(stdout);
	fclose(stream);
	
//---------------------------------------------------------------------------------------//	
//---------------------------------------------------------------------------------------//
//--------------------------------------------------------------------------------------//
	
	double dmu = 1./mubins, mu, mudash, kdash, test0, test2, test4;
	double L0, L2, L4;
	double dummy_pkdd, dummy_pkdt, dummy_pktt;
	double dummy_AA11, dummy_AA12, dummy_AA22, dummy_AA23, dummy_AA33;
	double dummy_BB111, dummy_BB112, dummy_BB121, dummy_BB122, dummy_BB211;
	double dummy_BB212, dummy_BB221, dummy_BB222, dummy_BB312, dummy_BB321;
	double dummy_BB322, dummy_BB422;
	double dummy_pkb2delta, dummy_pkb2theta, dummy_pkb22;
	double dummy_pkdeltadeltaSGC, dummy_pkdeltathetaSGC, dummy_pkdeltadeltaNGC, dummy_pkdeltathetaNGC, pkmuSGC, pkmuNGC;
	double dummy_pkbs2delta, dummy_pkbs2theta, dummy_sigma3pkterm, dummy_pkb2s2, dummy_pkbs22;
	
	double AAkmuSGC, BBkmuSGC, Damp_NGC, Damp_SGC, factor, sigvarg, AAkmuNGC, BBkmuNGC;
	
	double *pk0_SGC = new double[60];
	double *pk2_SGC = new double[60];
	double *pk4_SGC = new double[60];
	
	double *pk0_NGC = new double[60];
	double *pk2_NGC = new double[60];
	double *pk4_NGC = new double[60];
	double *kps = new double[60];
	
	for(i = 0; i < 60; i++){
		
		kps[i] = i*0.005 + 0.005/2.;
		
		pk0_SGC[i] = 0.;
		pk2_SGC[i] = 0.;
		pk4_SGC[i] = 0.;
		pk0_NGC[i] = 0.;
		pk2_NGC[i] = 0.;
		pk4_NGC[i] = 0.;
		for(j = 0; j < mubins; j++){
			
			mu = j*dmu + dmu/2.;
			
			factor = sqrt(1. + mu*mu*((1./(FAP*FAP)) - 1.));
			kdash = kps[i]*factor/alpha_perp;
			
			if(kdash > AAk[0]){
				
				dummy_AA11 = linterp(nAA, AAk, AA11, kdash);
				dummy_AA12 = linterp(nAA, AAk, AA12, kdash);
				dummy_AA22 = linterp(nAA, AAk, AA22, kdash);
				dummy_AA23 = linterp(nAA, AAk, AA23, kdash);
				dummy_AA33 = linterp(nAA, AAk, AA33, kdash);
				
				dummy_BB111 = linterp(nBB, BBk, BB111, kdash);
				dummy_BB112 = linterp(nBB, BBk, BB112, kdash);
				//dummy_BB121 = linterp(nBB, BBk, BB121, kdash);
				dummy_BB122 = linterp(nBB, BBk, BB122, kdash);
				dummy_BB211 = linterp(nBB, BBk, BB211, kdash);
				dummy_BB212 = linterp(nBB, BBk, BB212, kdash);
				//dummy_BB221 = linterp(nBB, BBk, BB221, kdash);
				dummy_BB222 = linterp(nBB, BBk, BB222, kdash);
				dummy_BB312 = linterp(nBB, BBk, BB312, kdash);
				//dummy_BB321 = linterp(nBB, BBk, BB321, kdash);
				dummy_BB322 = linterp(nBB, BBk, BB322, kdash);
				dummy_BB422 = linterp(nBB, BBk, BB422, kdash);
			
				dummy_pkdd = linterp(ndd, kdd, pkdd, kdash);
				dummy_pkdt = linterp(ndt, kdt, pkdt, kdash);
				dummy_pktt = linterp(ntt, ktt, pktt, kdash);
				
				dummy_pkb2delta = linterp(nshun, kshun, pkb2delta, kdash);
				dummy_pkb2theta = linterp(nshun, kshun, pkb2theta, kdash);
				dummy_pkb22 = linterp(nshun, kshun, pkb22, kdash);
			
				dummy_pkbs2delta = linterp(nshun, kshun, pkbs2delta, kdash);
				dummy_pkbs2theta = linterp(nshun, kshun, pkbs2theta, kdash);
				dummy_sigma3pkterm = linterp(nshun, kshun, sigma3pkterm, kdash);
				dummy_pkb2s2 = linterp(nshun, kshun, pkb2s2, kdash);
				dummy_pkbs22 = linterp(nshun, kshun, pkbs22, kdash);
			
				dummy_pkdeltadeltaNGC = b1NGC*b1NGC*dummy_pkdd + 2.*b2NGC*b1NGC*dummy_pkb2delta + 
										2.*bs2NGC*b1NGC*dummy_pkbs2delta + 2.*b3nlNGC*b1NGC*dummy_sigma3pkterm + 
										b2NGC*b2NGC*dummy_pkb22 + 2.*b2NGC*bs2NGC*dummy_pkb2s2 + 
										bs2NGC*bs2NGC*dummy_pkbs22 + N_NGC;
				dummy_pkdeltathetaNGC = b1NGC*dummy_pkdt + b2NGC*dummy_pkb2theta + 
										bs2NGC*dummy_pkbs2theta + b3nlNGC*dummy_sigma3pkterm;
				
				dummy_pkdeltadeltaSGC = b1SGC*b1SGC*dummy_pkdd + 2.*b2SGC*b1SGC*dummy_pkb2delta + 
										2.*bs2SGC*b1SGC*dummy_pkbs2delta + 2.*b3nlSGC*b1SGC*dummy_sigma3pkterm + 
										b2SGC*b2SGC*dummy_pkb22 + 2.*b2SGC*bs2SGC*dummy_pkb2s2 + 
										bs2SGC*bs2SGC*dummy_pkbs22 + N_SGC;
				dummy_pkdeltathetaSGC = b1SGC*dummy_pkdt + b2SGC*dummy_pkb2theta + 
										bs2SGC*dummy_pkbs2theta + b3nlSGC*dummy_sigma3pkterm;
				
				L0 = 1.*dmu;
				L2 = 0.5*(3.*mu*mu - 1.)*dmu;
				L4 = (1./8.)*(35.*mu*mu*mu*mu - 30.*mu*mu + 3.)*dmu;
				//L0 = 1.;
				//L2 = 0.5*(3.*mu*mu - 1.);
				//L4 = (1./8.)*(35.*mu*mu*mu*mu - 30.*mu*mu + 3.);
				
				mudash = mu/(FAP*factor);
				
				AAkmuNGC = dummy_AA11*pow(mudash, 2*1)*pow(betaNGC, 1);
				AAkmuNGC += dummy_AA12*pow(mudash, 2*1)*pow(betaNGC, 2);
				AAkmuNGC += dummy_AA22*pow(mudash, 2*2)*pow(betaNGC, 2);
				AAkmuNGC += dummy_AA23*pow(mudash, 2*2)*pow(betaNGC, 3);
				AAkmuNGC += dummy_AA33*pow(mudash, 2*3)*pow(betaNGC, 3);
				
				BBkmuNGC = dummy_BB111*pow(mudash, 2*1)*pow(-betaNGC, 1+1);
				BBkmuNGC += dummy_BB112*pow(mudash, 2*1)*pow(-betaNGC, 1+2);
				//BBkmu += dummy_BB121*pow(mudash, 2*1)*pow(-beta, 2+1);
				BBkmuNGC += dummy_BB122*pow(mudash, 2*1)*pow(-betaNGC, 2+2);
				BBkmuNGC += dummy_BB211*pow(mudash, 2*2)*pow(-betaNGC, 1+1);
				BBkmuNGC += dummy_BB212*pow(mudash, 2*2)*pow(-betaNGC, 1+2);
				//BBkmu += dummy_BB221*pow(mudash, 2*2)*pow(-beta, 2+1);
				BBkmuNGC += dummy_BB222*pow(mudash, 2*2)*pow(-betaNGC, 2+2);
				BBkmuNGC += dummy_BB312*pow(mudash, 2*3)*pow(-betaNGC, 1+2);
				//BBkmu += dummy_BB321*pow(mudash, 2*3)*pow(-beta, 2+1);
				BBkmuNGC += dummy_BB322*pow(mudash, 2*3)*pow(-betaNGC, 2+2);
				BBkmuNGC += dummy_BB422*pow(mudash, 2*4)*pow(-betaNGC, 2+2);
				
				AAkmuSGC = dummy_AA11*pow(mudash, 2*1)*pow(betaSGC, 1);
				AAkmuSGC += dummy_AA12*pow(mudash, 2*1)*pow(betaSGC, 2);
				AAkmuSGC += dummy_AA22*pow(mudash, 2*2)*pow(betaSGC, 2);
				AAkmuSGC += dummy_AA23*pow(mudash, 2*2)*pow(betaSGC, 3);
				AAkmuSGC += dummy_AA33*pow(mudash, 2*3)*pow(betaSGC, 3);
				
				BBkmuSGC = dummy_BB111*pow(mudash, 2*1)*pow(-betaSGC, 1+1);
				BBkmuSGC += dummy_BB112*pow(mudash, 2*1)*pow(-betaSGC, 1+2);
				//BBkmu += dummy_BB121*pow(mudash, 2*1)*pow(-beta, 2+1);
				BBkmuSGC += dummy_BB122*pow(mudash, 2*1)*pow(-betaSGC, 2+2);
				BBkmuSGC += dummy_BB211*pow(mudash, 2*2)*pow(-betaSGC, 1+1);
				BBkmuSGC += dummy_BB212*pow(mudash, 2*2)*pow(-betaSGC, 1+2);
				//BBkmu += dummy_BB221*pow(mudash, 2*2)*pow(-beta, 2+1);
				BBkmuSGC += dummy_BB222*pow(mudash, 2*2)*pow(-betaSGC, 2+2);
				BBkmuSGC += dummy_BB312*pow(mudash, 2*3)*pow(-betaSGC, 1+2);
				//BBkmu += dummy_BB321*pow(mudash, 2*3)*pow(-beta, 2+1);
				BBkmuSGC += dummy_BB322*pow(mudash, 2*3)*pow(-betaSGC, 2+2);
				BBkmuSGC += dummy_BB422*pow(mudash, 2*4)*pow(-betaSGC, 2+2);
				
				Damp_NGC = 1./(1. + (kdash*mudash*sigmav_NGC)*(kdash*mudash*sigmav_NGC)/2.);
				Damp_SGC = 1./(1. + (kdash*mudash*sigmav_SGC)*(kdash*mudash*sigmav_SGC)/2.);
				//sigvarg = beta*b1*kdash*mudash*sigmav/(alpha_para*alpha_para);
				//Damp = exp(-sigvarg*sigvarg);
				
				pkmuNGC = Damp_NGC*(dummy_pkdeltadeltaNGC + 2.*betaNGC*b1NGC*mudash*mudash*dummy_pkdeltathetaNGC + 
							    betaNGC*betaNGC*b1NGC*b1NGC*mudash*mudash*mudash*mudash*dummy_pktt + 
							    b1NGC*b1NGC*b1NGC*AAkmuNGC + b1NGC*b1NGC*b1NGC*b1NGC*BBkmuNGC);
				pkmuSGC = Damp_SGC*(dummy_pkdeltadeltaSGC + 2.*betaSGC*b1SGC*mudash*mudash*dummy_pkdeltathetaSGC + 
							    betaSGC*betaSGC*b1SGC*b1SGC*mudash*mudash*mudash*mudash*dummy_pktt + 
							    b1SGC*b1SGC*b1SGC*AAkmuSGC + b1SGC*b1SGC*b1SGC*b1SGC*BBkmuSGC);
				
				if(Nmodes_norm_NGC[i] > 0){
					pk0_NGC[i] += 100.*L0*pkmuNGC*Nmodes_NGC[i][j]/Nmodes_norm_NGC[i];
					pk2_NGC[i] += 100.*L2*pkmuNGC*Nmodes_NGC[i][j]/Nmodes_norm_NGC[i];
					pk4_NGC[i] += 100.*L4*pkmuNGC*Nmodes_NGC[i][j]/Nmodes_norm_NGC[i];
				}
				if(Nmodes_norm_SGC[i] > 0){
					pk0_SGC[i] += 100.*L0*pkmuSGC*Nmodes_SGC[i][j]/Nmodes_norm_SGC[i];
					pk2_SGC[i] += 100.*L2*pkmuSGC*Nmodes_SGC[i][j]/Nmodes_norm_SGC[i];
					pk4_SGC[i] += 100.*L4*pkmuSGC*Nmodes_SGC[i][j]/Nmodes_norm_SGC[i];
				}
			}
		}
		
		pk0_SGC[i] *= 1.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
		pk2_SGC[i] *= 5.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
		pk4_SGC[i] *= 9.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
		pk0_NGC[i] *= 1.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
		pk2_NGC[i] *= 5.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
		pk4_NGC[i] *= 9.*rs_ratio/(alpha_perp*alpha_perp*alpha_para);
	}
//---------------------------------------------------------------------------------------//
	
	double s_NGC[5000], RR0_NGC[5000], RR2_NGC[5000], RR4_NGC[5000], RR6_NGC[5000], RR8_NGC[5000];
	double s_SGC[5000], RR0_SGC[5000], RR2_SGC[5000], RR4_SGC[5000], RR6_SGC[5000], RR8_SGC[5000];
	double xi0_NGC[5000], xi2_NGC[5000], xi4_NGC[5000];
	double xi0_SGC[5000], xi2_SGC[5000], xi4_SGC[5000];
	double sdummy, RR0dummy, RR2dummy, RR4dummy, RR6dummy, RR8dummy;
	
	char RRfile[200], RRfile_SGC[200];
	//sprintf(RRfile_SGC, "/Users/chang/projects/clustools/input/wilson_random_win_patchy_z1_0_20.dat");
	//sprintf(RRfile_SGC, "/Users/chang/projects/clustools/input/wilson_random_win_patchy_z1_all_SGC.dat");
	sprintf(RRfile, "/Users/chang/projects/nonGaussLike/dat/Beutler/public_material_RSD/Beutleretal_window_z1_NGC_.dat");
	sprintf(RRfile_SGC, "/Users/chang/projects/nonGaussLike/dat/Beutler/public_material_RSD/Beutleretal_window_z1_SGC_.dat");

	stream = fopen(RRfile, "r");
	if(stream == NULL) {
		printf("Cannot open RR file\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf\n", 
				  &sdummy, &dummy, &RR0dummy, &RR2dummy, &RR4dummy, &RR6dummy, &RR8dummy)) != EOF){
		
		s_NGC[n] = sdummy;
		RR0_NGC[n] = RR0dummy/(sdummy*sdummy*sdummy);
		RR2_NGC[n] = RR2dummy/(sdummy*sdummy*sdummy);
		RR4_NGC[n] = RR4dummy/(sdummy*sdummy*sdummy);
		RR6_NGC[n] = RR6dummy/(sdummy*sdummy*sdummy);
		RR8_NGC[n] = RR8dummy/(sdummy*sdummy*sdummy);
		
		xi0_NGC[n] = pstoxi(60-1, kps, pk0_NGC, s_NGC[n], 0);
		xi2_NGC[n] = pstoxi(60-1, kps, pk2_NGC, s_NGC[n], 2);
		xi4_NGC[n] = pstoxi(60-1, kps, pk4_NGC, s_NGC[n], 4);
		
		n++;
	}
	fclose(stream);
	
	stream = fopen(RRfile_SGC, "r");
	if(stream == NULL) {
		printf("Cannot open RR_SGC file\n");
		exit(1);
	}
	n = 0;
	while((fscanf(stream, "%lf %lf %lf %lf %lf %lf %lf\n", 
				  &sdummy, &dummy, &RR0dummy, &RR2dummy, &RR4dummy, &RR6dummy, &RR8dummy)) != EOF){
		
		s_SGC[n] = sdummy;
		RR0_SGC[n] = RR0dummy/(sdummy*sdummy*sdummy);
		RR2_SGC[n] = RR2dummy/(sdummy*sdummy*sdummy);
		RR4_SGC[n] = RR4dummy/(sdummy*sdummy*sdummy);
		RR6_SGC[n] = RR6dummy/(sdummy*sdummy*sdummy);
		RR8_SGC[n] = RR8dummy/(sdummy*sdummy*sdummy);
		
		xi0_SGC[n] = pstoxi(60-1, kps, pk0_SGC, s_SGC[n], 0);
		xi2_SGC[n] = pstoxi(60-1, kps, pk2_SGC, s_SGC[n], 2);
		xi4_SGC[n] = pstoxi(60-1, kps, pk4_SGC, s_SGC[n], 4);
		
		n++;
	}
	fclose(stream);
	int nr = n;
	
	delete [] pk0_NGC;
	delete [] pk2_NGC;
	delete [] pk4_NGC;
	delete [] pk0_SGC;
	delete [] pk2_SGC;
	delete [] pk4_SGC;
	
	double norm_SGC = 0.;
	int counter_SGC = 0;
	for(int i = 0; i < nr; i++){
		
		if(s_SGC[i] > 10 && s_SGC[i] < 12){
			
			norm_SGC += RR0_SGC[i];
			counter_SGC++;
		}
	}
	norm_SGC /= counter_SGC;
	
	double norm_NGC = 0.;
	int counter_NGC = 0;
	for(int i = 0; i < nr; i++){
		
		if(s_NGC[i] > 10 && s_NGC[i] < 12){
			
			norm_NGC += RR0_NGC[i];
			counter_NGC++;
		}
	}
	norm_NGC /= counter_NGC;
	
	for(i = 0; i < nr; i++){
		
		RR0_SGC[i] /= norm_SGC;
		RR2_SGC[i] /= norm_SGC;
		RR4_SGC[i] /= norm_SGC;
		RR6_SGC[i] /= norm_SGC;
		RR8_SGC[i] /= norm_SGC;
		
		if(s_SGC[i] < 10){
			
			RR0_SGC[i] = 1.;
			RR2_SGC[i] = 0.;
			RR4_SGC[i] = 0.;
			RR6_SGC[i] = 0.;
			RR8_SGC[i] = 0.;
		}
	}
	
	for(i = 0; i < nr; i++){
		
		RR0_NGC[i] /= norm_NGC;
		RR2_NGC[i] /= norm_NGC;
		RR4_NGC[i] /= norm_NGC;
		RR6_NGC[i] /= norm_NGC;
		RR8_NGC[i] /= norm_NGC;
		
		if(s_NGC[i] < 10){
			
			RR0_NGC[i] = 1.;
			RR2_NGC[i] = 0.;
			RR4_NGC[i] = 0.;
			RR6_NGC[i] = 0.;
			RR8_NGC[i] = 0.;
		}
	}

	double xi0conv_NGC[5000], xi2conv_NGC[5000], xi4conv_NGC[5000];
	double xi0conv_SGC[5000], xi2conv_SGC[5000], xi4conv_SGC[5000];
	
	for(i = 0; i < nr; i++){
		
		xi0conv_NGC[i] = xi0_NGC[i]*RR0_NGC[i] + xi2_NGC[i]*RR2_NGC[i]/5. + xi4_NGC[i]*RR4_NGC[i]/9.;
		xi2conv_NGC[i] = xi0_NGC[i]*RR2_NGC[i] + 
                    xi2_NGC[i]*(RR0_NGC[i] + 2.*RR2_NGC[i]/7. + 2.*RR4_NGC[i]/7.) + 
                    xi4_NGC[i]*(2.*RR2_NGC[i]/7. + 100.*RR4_NGC[i]/693. + 25.*RR6_NGC[i]/143.);
                xi4conv_NGC[i] = xi0_NGC[i]*RR4_NGC[i] + 
                    xi2_NGC[i]*(18.*RR2_NGC[i]/35. + 20.*RR4_NGC[i]/77. + 45.*RR6_NGC[i]/143.) + 
                    xi4_NGC[i]*(RR0_NGC[i] + 20.*RR2_NGC[i]/77. + 162.*RR4_NGC[i]/1001. + 20.*RR6_NGC[i]/143. + 490.*RR8_NGC[i]/2431.);

		xi0conv_SGC[i] = xi0_SGC[i]*RR0_SGC[i] + xi2_SGC[i]*RR2_SGC[i]/5. + xi4_SGC[i]*RR4_SGC[i]/9.;
		xi2conv_SGC[i] = xi0_SGC[i]*RR2_SGC[i] + 
                    xi2_SGC[i]*(RR0_SGC[i] + 2.*RR2_SGC[i]/7. + 2.*RR4_SGC[i]/7.) + 
                    xi4_SGC[i]*(2.*RR2_SGC[i]/7. + 100.*RR4_SGC[i]/693. + 25.*RR6_SGC[i]/143.);
                xi4conv_SGC[i] = xi0_SGC[i]*RR4_SGC[i] + 
                    xi2_SGC[i]*(18.*RR2_SGC[i]/35. + 20.*RR4_SGC[i]/77. + 45.*RR6_SGC[i]/143.) + 
                    xi4_SGC[i]*(RR0_SGC[i] + 20.*RR2_SGC[i]/77. + 162.*RR4_SGC[i]/1001. + 20.*RR6_SGC[i]/143. + 490.*RR8_SGC[i]/2431.);
	}
	
	double* pk = new double[2*totbinrange];
	
	for(i = 0; i < binrange1; i++){
			
		pk[i] = xitops(nr-1, s_NGC, xi0conv_NGC, x[i], 0); 
		if(i < binrange2) pk[i+binrange1] = xitops(nr-1, s_NGC, xi2conv_NGC, x[i], 2);
		if(i < binrange3) pk[i+binrange1+binrange2] = xitops(nr-1, s_NGC, xi4conv_NGC, x[i], 4);
		pk[i+totbinrange] = xitops(nr-1, s_SGC, xi0conv_SGC, x[i], 0); 
		if(i < binrange2) pk[i+totbinrange+binrange1] = xitops(nr-1, s_SGC, xi2conv_SGC, x[i], 2);
		if(i < binrange3) pk[i+totbinrange+binrange1+binrange2] = xitops(nr-1, s_SGC, xi4conv_SGC, x[i], 4);
	}
	return pk;
}
