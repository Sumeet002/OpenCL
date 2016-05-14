#ifndef MIRE_H
#define MIRE_H

#include <vector>

using namespace std;

float *MIRE(float[] , float ,int , int);
void MIRE_Automatic(float[],int,int,int,int,float);
float TV_column_norm(float[],int,int,float);
void specify_column(float[],int,int,int,vector<float>);
float gaussian(int,float);
std::vector <std::vector<float> > target_histogram(std::vector <std::vector<float> > ,int,int,float);
std::vector <std::vector<float> > column_sorting(float [],int,int);
vector <float> histo_column(float[],int,int,int);


#endif // MIRE_H
