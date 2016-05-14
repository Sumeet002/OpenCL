#include "borders.h"
#include <stdio.h>

void borders(float *Src_Image,float *Mod_Image,int w1,int h1, int B){

   //copying image in the middle

   printf("1\n");
   for(int column = 0; column < w1 ; column++){    //for columns in the middle
     for(int row = 0 ; row < h1 ; row++ ){         //for all rows
         Mod_Image[row*(w1+2*B) + B + column] = Src_Image[row*w1 + column];
     }
  }

   printf("2\n");
  //left side

   for(int column=-B ; column < 0 ; column ++){  //for all columns on left
     for(int row=0 ; row < h1 ; row++){          //for all rows
         Mod_Image[row*(w1+2*B) + B + column] = Src_Image[row*w1+(-column)];
     }
   }

   printf("3\n");
  //right side
   for(int column=w1 ; column < w1+ B ; column++){
     for(int row=0 ; row < h1 ; row++){
         Mod_Image[row*(w1+2*B) + B + column]= Src_Image[row*w1+(2*w1-column-1)];   
     }
   }

   printf("4\n");   
} 
  

