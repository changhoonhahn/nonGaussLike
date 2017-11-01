#include <math.h>

double linterp(int n, double x[], double y[], double xnew){

   int j;
   int found, range;
   double xperc;
   double ynew = 0.;

   if(xnew < x[0]){
      xperc = fabs(100.*fabs(xnew - x[0])/xnew);
      if(xperc < 20){
         //cout<<"1 OUTSIDE OF RANGE...  "<<xnew<<" <= "<<x[0]<< endl;
      }
      else{
		  printf("1 x range: %f %f\n", x[0], x[n]);
		  printf("1 y range: %f %f\n", y[0], y[n]);
		  printf("new = %f %f\n", xnew, x[0]);
         //cout<<"1 x range: "<<x[0]<<" "<<x[n]<< endl;
         //cout<<"1 y range: "<<y[0]<<" "<<y[n]<< endl;
         //cout<<"new = "<<xnew<<" < "<<x[0]<< endl;
      }
      ynew = y[0];
   }
   else if(xnew > x[n]){ 
      xperc = fabs(100.*fabs(xnew - x[n])/xnew);
      if(xperc < 20){
         //cout<<"2 OUTSIDE OF RANGE...  "<<xnew<<" >= "<<x[n]<< endl;
      }
      else{
		  printf("2 x range: %f %f\n", x[0], x[n]);
		  printf("2 y range: %f %f\n", y[0], y[n]);
		  printf("new = %f %f\n", xnew, x[0]);
      }
      ynew = y[n];
   }
   else{  // if everything went well we end up here
      j = 1;
      found = 1;
      range = 1;
      while(found){ // go through the array till xnew is localized
         if(xnew >= x[j-1] && xnew <= x[j]){ 
            found = 0;
         }
         else if(j < n){
            j++; 
         }
         else{ // if nothing foound
            found = 0;
            range = 0;
         }
      }
      if(range == 1){
         ynew = (y[j] - y[j-1])/(x[j] - x[j-1])*(xnew - x[j-1]) + y[j-1];
      }
      else if(range == 0){
         //cout<<"3 x range: "<<x[0]<<" "<<x[n]<< endl;
         //cout<<"3 y range: "<<y[0]<<" "<<y[n]<< endl;
		  printf("3 x range: %f %f\n", x[0], x[n]);
		  printf("3 y range: %f %f\n", y[0], y[n]);
      }
   }

   return ynew;
}
