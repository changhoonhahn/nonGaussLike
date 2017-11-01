#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

int* make_1D_int(long int xdim) {
	int* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
	arr1d=(int*)calloc(xdim, sizeof(int) );
  	return arr1d;
    /* This is fine because the array itself will live on. */
}

void destroy_1D_int(int* arr1d) {
	free(&arr1d[0]);
}

//long int* make_1D_ulint(long int xdim) {
//	unsigned long int* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
//	arr1d=(unsigned long int*)calloc(xdim, sizeof(unsigned long int) );
//  	return arr1d;
    /* This is fine because the array itself will live on. */
//}

//void destroy_1D_ulint(unsigned long int* arr1d) {
//	free(&arr1d[0]);
//}

long int* make_1D_long_int(long int xdim) {
	long int* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
	arr1d=(long int*)calloc(xdim, sizeof(long int) );
  	return arr1d;
    /* This is fine because the array itself will live on. */
}

void destroy_1D_long_int(long int* arr1d) {
	free(&arr1d[0]);
}

//bool* make_1D_bool(long int xdim) {
//	bool* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double *//
//	arr1d=(bool*)calloc(xdim, sizeof(bool) );
//  	return arr1d;
    /* This is fine because the array itself will live on. */
//}

//void destroy_1D_bool(bool* arr1d) {
//	free(&arr1d[0]);
//}

double* make_1D_double(long int xdim) {
	double* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
	arr1d=(double*)calloc(xdim, sizeof(double) );
  	return arr1d;
    /* This is fine because the array itself will live on. */
}

void destroy_1D_double(double* arr1d) {
	free(&arr1d[0]);
}

float* make_1D_float(long int xdim) {
	float* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
	arr1d=(float*)calloc(xdim, sizeof(float) );
  	return arr1d;
    /* This is fine because the array itself will live on. */
}

void destroy_1D_float(float* arr1d) {
	free(&arr1d[0]);
}

long double* make_1D_long_double(long int xdim) {
	long double* arr1d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
	arr1d=(long double*)calloc(xdim, sizeof(long double) );
  	return arr1d;
    /* This is fine because the array itself will live on. */
}

void destroy_1D_long_double(long double* arr1d) {
	free(&arr1d[0]);
}

int** make_2D_int(int xdim, int ydim) {
	int** arr2d;
	int* flatarr=(int*)calloc(xdim*ydim, sizeof(int));
	int i;
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 2D_float array");
        exit(0);
    }
    arr2d=(int**)malloc(xdim*sizeof(int*));
    for (i=0; i<xdim; i++) {
    	arr2d[i]=flatarr+i*ydim;
    }
 	return arr2d;   
}

void destroy_2D_int(int** arr2d, int xdim) {
	/* rip up the main array */
	free(&arr2d[0][0]);	
	free(&arr2d[0]);
}

long int** make_2D_long_int(long int xdim, long int ydim) {
	long int** arr2d;
	long int* flatarr=(long int*)calloc(xdim*ydim, sizeof(long int));
	long int i;
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 2D_float array");
        exit(0);
    }
    arr2d=(long int**)malloc(xdim*sizeof(long int*));
    for (i=0; i<xdim; i++) {
    	arr2d[i]=flatarr+i*ydim;
    }
 	return arr2d;   
}

void destroy_2D_long_int(long int** arr2d, long int xdim) {
	/* rip up the main array */
	free(&arr2d[0][0]);	
	free(&arr2d[0]);
}

float** make_2D_float(long int xdim, long int ydim) {
	float** arr2d;
	float* flatarr=(float*)calloc(xdim*ydim, sizeof(float));
	long int i;
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 2D_float array");
        exit(0);
    }
    arr2d=(float**)malloc(xdim*sizeof(float*));
    for (i=0; i<xdim; i++) {
    	arr2d[i]=flatarr+i*ydim;
    }
 	return arr2d;   
}

void destroy_2D_float(float** arr2d, long int xdim) {
	/* rip up the main array */
	free(&arr2d[0][0]);	
	free(&arr2d[0]);
}

double** make_2D_double(long int xdim, long int ydim) {
	double** arr2d;
	double* flatarr=(double*)calloc(xdim*ydim, sizeof(double));
	long int i;
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 2D_float array");
        exit(0);
    }
    arr2d=(double**)malloc(xdim*sizeof(double*));
    for (i=0; i<xdim; i++) {
    	arr2d[i]=flatarr+i*ydim;
    }
 	return arr2d;   
}

void destroy_2D_double(double** arr2d, long int xdim) {
	/* rip up the main array */
	free(&arr2d[0][0]);	
	free(&arr2d[0]);
}

long double** make_2D_long_double(long int xdim, long int ydim) {
	long double** arr2d;
	long double* flatarr=(long double*)calloc(xdim*ydim, sizeof(long double));
	long int i;
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 2D_float array");
        exit(0);
    }
    arr2d=(long double**)malloc(xdim*sizeof(long double*));
    for (i=0; i<xdim; i++) {
    	arr2d[i]=flatarr+i*ydim;
    }
 	return arr2d;   
}

void destroy_2D_long_double(long double** arr2d, long int xdim) {
	/* rip up the main array */
	free(&arr2d[0][0]);	
	free(&arr2d[0]);
}

int*** make_3D_int(long int xdim, long int ydim, long int zdim) {
	int ***arr3d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
  	long int i, j;
	int* flatarr;
	flatarr=(int*)calloc(xdim*ydim*zdim, sizeof(int) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 3D_double array");
        exit(0);
    }

	/* next we allocate room for the pointers to the rows */
	arr3d=(int***)malloc(xdim * sizeof(int **));
	
	/* and for each of these we assign a pointer to a newly
    allocated array of pointers to a row */
	for (i=0; i<xdim; i++) {
		arr3d[i]=(int**)malloc(ydim*sizeof(int *));
		/* and for each space in this array we put a pointer to
        the first element of each row in the array space
        originally allocated */
		for (j=0; j<ydim; j++) {
			arr3d[i][j] = flatarr + (i*ydim*zdim+j*zdim);
		}
	}
	
  	return arr3d;
    /* This is fine because the array itself will live on. */
}

void destroy_3D_int(int ***arr3d, long int xdim) {
	/* free the main array */
	long int x;
	/* rip up the main array */
	free(&arr3d[0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
		free(&arr3d[x][0]);
	}		
	free(&arr3d[0]);
}

double*** make_3D_double(long int xdim, long int ydim, long int zdim) {
	double ***arr3d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
  	long int i, j;
	double* flatarr;
	flatarr=(double*)calloc(xdim*ydim*zdim, sizeof(double) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the 3D_double array");
        exit(0);
    }

	/* next we allocate room for the pointers to the rows */
	arr3d=(double***)malloc(xdim * sizeof(double **));
	
	/* and for each of these we assign a pointer to a newly
    allocated array of pointers to a row */
	for (i=0; i<xdim; i++) {
		arr3d[i]=(double**)malloc(ydim*sizeof(double *));
		/* and for each space in this array we put a pointer to
        the first element of each row in the array space
        originally allocated */
		for (j=0; j<ydim; j++) {
			arr3d[i][j] = flatarr + (i*ydim*zdim+j*zdim);
		}
	}
	
  	return arr3d;
    /* This is fine because the array itself will live on. */
}

void destroy_3D_double(double ***arr3d, long int xdim) {
	/* free the main array */
	long int x;
	/* rip up the main array */
	free(&arr3d[0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
		free(&arr3d[x][0]);
	}		
	free(&arr3d[0]);
}

float*** make_3D_float(long int xdim, long int ydim, long int zdim) {
	float ***arr3d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
  	long int i, j;
	float* flatarr;
	flatarr=(float*)calloc(xdim*ydim*zdim, sizeof(float) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the array");
        exit(0);
    }

	/* next we allocate room for the pointers to the rows */
	arr3d=(float***)malloc(xdim * sizeof(float **));
	
	/* and for each of these we assign a pointer to a newly
    allocated array of pointers to a row */
	for (i=0; i<xdim; i++) {
		arr3d[i] = (float**)malloc(ydim*sizeof(float *));
		/* and for each space in this array we put a pointer to
        the first element of each row in the array space
        originally allocated */
		for (j=0; j<ydim; j++) {
			arr3d[i][j] = flatarr + (i*ydim*zdim+j*zdim);
		}
	}
  	return arr3d;
    /* This is fine because the array itself will live on. */
}

void destroy_3D_float(float*** arr3d, long int xdim) {
	/* free the main array */
	long int x;
	/* rip up the main array */
	free(&arr3d[0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
		free(&arr3d[x][0]);
	}		
	free(&arr3d[0]);
}


int**** make_4D_int(long int xdim, long int ydim, long int zdim, long int kdim) {
	int ****arr4d; // Declaration of arr3d as: pointer-to-pointer-to-pointer of double 
  	long int i, j, w;
	int* flatarr;
	flatarr = (int*)calloc(xdim*ydim*zdim*kdim, sizeof(int) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the array");
        exit(0);
    }

	// next we allocate room for the pointers to the rows 
	arr4d = (int****)malloc(xdim * sizeof(int ***));
	
	// and for each of these we assign a pointer to a newly
    //allocated array of pointers to a row 
	for (i=0; i<xdim; i++) {
		arr4d[i] = (int***)malloc(ydim*sizeof(int **));
		// and for each space in this array we put a pointer to
        //the first element of each row in the array space
        //originally allocated 
		for (j=0; j<ydim; j++) {
			arr4d[i][j] = (int**)malloc(zdim*sizeof(int *));
			for (w=0; w<zdim; w++) {
				arr4d[i][j][w] = flatarr + (i*ydim*zdim*kdim + j*zdim*kdim + w*kdim);
                        }
		}
	}
  	return arr4d;
    // This is fine because the array itself will live on. 
}

void destroy_4D_int(int ****arr4d, long int xdim, long int ydim) {
	/* free the main array */
	long int x, v;
	/* rip up the main array */
	free(&arr4d[0][0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
           for (v=0; v<ydim; v++) {
			free(&arr4d[x][v][0]);
           }
	   free(&arr4d[x][0]);
	}		
	free(&arr4d[0]);
}


float**** make_4D_float(long int xdim, long int ydim, long int zdim, long int kdim) {
	float ****arr4d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
  	long int i, j, w;
	float* flatarr;
	flatarr = (float*)calloc(xdim*ydim*zdim*kdim, sizeof(float) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the array");
        exit(0);
    }

	/* next we allocate room for the pointers to the rows */
	arr4d = (float****)malloc(xdim * sizeof(float ***));
	
	/* and for each of these we assign a pointer to a newly
    allocated array of pointers to a row */
	for (i=0; i<xdim; i++) {
		arr4d[i] = (float***)malloc(ydim*sizeof(float **));
		/* and for each space in this array we put a pointer to
        the first element of each row in the array space
        originally allocated */
		for (j=0; j<ydim; j++) {
			arr4d[i][j] = (float**)malloc(zdim*sizeof(float *));
			for (w=0; w<zdim; w++) {
				arr4d[i][j][w] = flatarr + (i*ydim*zdim*kdim + j*zdim*kdim + w*kdim);
                        }
		}
	}
  	return arr4d;
    /* This is fine because the array itself will live on. */
}


void destroy_4D_float(float ****arr4d, long int xdim, long int ydim) {
	/* free the main array */
	long int x, v;
	/* rip up the main array */
	free(&arr4d[0][0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
           for (v=0; v<ydim; v++) {
			free(&arr4d[x][v][0]);
           }
	   free(&arr4d[x][0]);
	}		
	free(&arr4d[0]);
}

double**** make_4D_double(long int xdim, long int ydim, long int zdim, long int kdim) {
	double ****arr4d; /* Declaration of arr3d as: pointer-to-pointer-to-pointer of double */
  	long int i, j, w;
	double* flatarr;
	flatarr = (double*)calloc(xdim*ydim*zdim*kdim, sizeof(double) );
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the array");
        exit(0);
    }
	
	/* next we allocate room for the pointers to the rows */
	arr4d = (double****)malloc(xdim * sizeof(double ***));
	
	/* and for each of these we assign a pointer to a newly
	 allocated array of pointers to a row */
	for (i=0; i<xdim; i++) {
		arr4d[i] = (double***)malloc(ydim*sizeof(double **));
		/* and for each space in this array we put a pointer to
		 the first element of each row in the array space
		 originally allocated */
		for (j=0; j<ydim; j++) {
			arr4d[i][j] = (double**)malloc(zdim*sizeof(double *));
			for (w=0; w<zdim; w++) {
				arr4d[i][j][w] = flatarr + (i*ydim*zdim*kdim + j*zdim*kdim + w*kdim);
			}
		}
	}
  	return arr4d;
    /* This is fine because the array itself will live on. */
}

void destroy_4D_double(double ****arr4d, long int xdim, long int ydim) {
	/* free the main array */
	long int x, v;
	/* rip up the main array */
	free(&arr4d[0][0][0][0]);
	
	/* now rip up the pointer arrays */
	for (x=0; x<xdim; x++) {
		for (v=0; v<ydim; v++) {
			free(&arr4d[x][v][0]);
		}
		free(&arr4d[x][0]);
	}		
	free(&arr4d[0]);
}

double***** make_5D_double(long int x0, long int x1, long int x2, long int x3, long int x4) {
	double***** arr5d; /* Declaration of arr5d as: pointer-to-pointer-to-pointer of double */
  	long int u, v, w, x;
	double* flatarr; 
	flatarr=(double*)calloc(x0*x1*x2*x3*x4, sizeof(double));
	
	if (flatarr == NULL)
    {
        puts("\nFailure to allocate room for the array");
        exit(0);
    }

	/* next we allocate room for the pointers to the rows */
	arr5d=(double *****)malloc(x0*sizeof(double ****));

	/* and for each of these we assign a pointer to a newly
    allocated array of pointers to a row */
	for (u=0; u<x0; u++) {
		arr5d[u] = (double ****)malloc(x1*sizeof(double ***));
		/* and for each space in this array we put a pointer to
        the first element of each row in the array space
        originally allocated */
		for (v=0; v<x1; v++) {
			arr5d[u][v] = (double ***)malloc(x2*sizeof(double **));
			for (w=0; w<x2; w++) {
				arr5d[u][v][w] = (double **)malloc(x3*sizeof(double *));
				for (x=0; x<x3; x++) {
					arr5d[u][v][w][x]=flatarr + x4*(x+x3*(w+x2*(v+(x1*u))));
				}
			}
		}			
	}	
		
  	return arr5d;
    /* This is fine because the array itself will live on. */
}

void destroy_5D_double(double***** arr5d, long int x0, long int x1, long int x2) {
	long int u, v, w;
	/* free the big array first */
	free(&arr5d[0][0][0][0][0]);
	
	/* then free the other guys */
	
	for (u=0; u<x0; u++) {
		for (v=0; v<x1; v++) {
			for (w=0; w<x2; w++) {
				free(&arr5d[u][v][w][0]);
			}
			free(&arr5d[u][v][0]);
		}
		free(&arr5d[u][0]);
	}
	free(&arr5d[0]); 
}
