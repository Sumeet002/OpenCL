#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "MIRE.h"
#include "borders.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(){
	
     
     //Gaussian Parameters
     const int SIGMA_MIN = 0; //minimal std deviation of Gaussian Weighting function
     const int SIGMA_MAX = 8; //maximal std deviation of Gaussian Weighting function
     const float DELTA = 0.5; //step between two consecutive std deviation
    
    
     Mat Src_Im = imread("test4.png",CV_LOAD_IMAGE_GRAYSCALE); //Load Image as Grayscale
     int w1 = Src_Im.cols;                             //Image Width
     int h1 = Src_Im.rows;                             //Image Height
   
     Mat_<float> Src_Im1;
     Src_Im.convertTo(Src_Im1,CV_32F);     
     
     imshow("Image",Src_Im);
     waitKey(30);
     
     //Border Effects:- Mirror Symmetry on Columns

     int W = w1 + 8*SIGMA_MAX;        //new width : 4 times SIGMA_MAX on each side
     
     Mat Src_Im2(h1,W,CV_32FC1,Scalar(0));  //new image allocation
   
     //Image Pointers
     float  *Src_Im1ptr =  Src_Im1.ptr<float>(0);
     float  *Src_Im2ptr =  Src_Im2.ptr<float>(0);

     borders(Src_Im1ptr,Src_Im2ptr,w1,h1,4*SIGMA_MAX);



     //Classic mirror symmetry on columns:
     //C1 C2 ...CN => C2 C1 | C1 C2 .....CN | CN CN-1

     //Central Function
     MIRE_Automatic(Src_Im2ptr,W,h1,SIGMA_MIN,SIGMA_MAX,DELTA);
 
       

    //Removing the Symmetry
     for(int x = 0 ; x < w1 ; x++ ){
        for(int y = 0 ; y < h1 ; y++){

            Src_Im1ptr[y*w1+x] = Src_Im2ptr[y*W+x+4*SIGMA_MAX];
        }
     }

     

     Src_Im2.release();  //Release temp Image

     //Dynamic correction: Imposing [0,255]     
     
     //Computing min and max of the output
     
     float min = Src_Im1ptr[0];
     float max = Src_Im1ptr[0];

     for(int i=1 ; i< w1*h1 ; i++){
         if(Src_Im1ptr[i] < min) min = Src_Im1ptr[i];
         if(Src_Im1ptr[i] > max) max = Src_Im1ptr[i];
     }

     for(int i=1 ; i< w1*h1 ; i++){
         Src_Im1ptr[i] = (255*(Src_Im1ptr[i]-min)/(max-min));
     }



     Mat Out_Im1;
     Src_Im1.convertTo(Out_Im1,CV_8UC1);
     //Output
     imshow("Final Denoised Image",Out_Im1);
     imwrite("test4result.png",Src_Im1);
     waitKey(30);
     while(1);
     
     return 0 ;

}




