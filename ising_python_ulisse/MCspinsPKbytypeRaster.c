#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

//const int RAND_MAX = 32767;

extern "C" {
int offset(int i, int length)
{
    return (length + i * (length - 2) - (i * (i - 1)) / 2 - 1);
}
double intPow(double b,int exp)
{
 int i;
 double result=b;
 for(i=1;i<exp;i++){ result=result*b; }
 return result;
}
}

extern "C" {
void pyMC(int N, int Ntype, int B, int first_steps, int logTd, double *J, int *idxtype, int *latticeIn, double* p, double* pK, int *latticeOut, int *magn_t)
{
    // random generator seed
    FILE *devran = fopen("/dev/urandom","r");
    unsigned int myrand;
    fread(&myrand, 4, 1, devran);
    fclose(devran);
    /* myrand=1234567 */;
    srand(myrand);



    int D = (N*(N+1))/2;
    int i,j,k,n;
    for (i=0;i<D;i++) { p[i]  = 0.0; } // magnetisation, pairwise corr, dimension D

    int lattice[N];
    for(i=0;i<N;i++)   lattice[i]=latticeIn[i];

    double expJ[N][N];
    for (i=0;i<N;i++) {
        expJ[i][i] = 1.0;
        int off=offset(i, N);
        for (j=i+1;j<N;j++) {
            expJ[i][j] = exp( J[off + j] );
            expJ[j][i] = expJ[i][j];
        } // end for on i (number of neurons)
    } // end for on j



    double h_local[N];
    double expHlocal[N];

    int decTime =intPow(2,logTd);

    for (i=0;i<N;i++) {
        expHlocal[i] = exp(-J[i] );
        for (j=0;j<N;j++) if(lattice[j]==1) expHlocal[i] /= expJ[i][j];
    }


    int nWindows,steps,loopTd;

    /*  Thermalization  */
    for (nWindows=0;nWindows< first_steps;nWindows++) {
        for(steps=intPow(2,nWindows); steps < intPow(2,nWindows+1); steps++ ){
            for (loopTd=0; loopTd< intPow(2,logTd);loopTd++){
                for(n=0; n<N;n++){
                    k=rand()%N;
                    double expMinusDelta = (lattice[k]==1 ? expHlocal[k] : 1.0/expHlocal[k]); /* = exp( -DeltaE ) */
                    if ( expMinusDelta>1.0 || rand() < expMinusDelta * RAND_MAX  ) {
                        lattice[k]=1.0 - lattice[k];
                        if(lattice[k]==1) for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] / expJ[k][i];
                        else for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] * expJ[k][i];
                    }
                }
            }
        }
    }


    /*  Sampling  */

    int b;
    //int magn_t[B];
    for (b=0;b< B;b++) {
        for(steps=0; steps < intPow(2,logTd); steps++ ){
            for(n=0; n<N;n++){
                k=rand()%N;
                double expMinusDelta = (lattice[k]==1 ? expHlocal[k] : 1.0/expHlocal[k]); /* = exp( -DeltaE ) */
                if ( expMinusDelta>1.0 || rand() < expMinusDelta * RAND_MAX  ) {
                    lattice[k]=1-lattice[k];
                    if(lattice[k]==1) for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] / expJ[k][i];
                    else for(i=0;i<N;i++) expHlocal[ i ] = expHlocal[ i ] * expJ[k][i];
                } // end if
            }// end for on neurons N
        }// end for on steps
	// p(K)
	//int magn = 0; // magnetisation := number of ON spins in 1 time bin
	int ii;
        for (ii=0;ii<Ntype;ii++) {
            i = idxtype[ii]; // only cells of given type
            p[i] = p[i] + lattice[i];
            if (lattice[i]==1) {
                int off=offset(i, N);
                for (j=i+1;j<N;j++) p[off + j] += lattice[j];
		//magn += 1;
		magn_t[b] += 1;
            } //endif on ON spins
        } //end for on cells of chosen type
	pK[magn_t[b]] += 1.0/B;

// full raster as 1D vector
    for (i=0;i<N;i++){
        latticeOut[b*N+i]=lattice[i];
        //cout << latticeOut[i] << endl;
    } // end for on cells

    }//end for on time bins
    for (i=0;i<D;i++)  p[i]/= B;


}
}





