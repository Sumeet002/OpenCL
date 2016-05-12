#define FILTER_SIZE 3
#define IMAGE_W 960
#define IMAGE_H 540
__kernel void conv_kernel(const __global uchar* dsrc_img_ , __global uchar* dout_img_, const __global float* dconv_mat){

	/*int index = get_global_id(0);
	dout_img_[index] = dsrc_img_[index];*/
	
	int index = get_global_id(0);

	int fIndex = 0;
	float sumGRAY = 0.0;


	//pass the pixel through the kernel if it can be centered inside it
	if(index >= IMAGE_W*(FILTER_SIZE-(FILTER_SIZE/2))+FILTER_SIZE/2 &&
	   index < IMAGE_W*IMAGE_H-IMAGE_W*(FILTER_SIZE-(FILTER_SIZE/2))-FILTER_SIZE/2){
		
		int value=0;
		for(int y=0; y < FILTER_SIZE ; y++){
			int yOff = IMAGE_W*(y-(FILTER_SIZE/2));
            		for(int x=0;x<FILTER_SIZE;x++){
				int xOff = (x-(FILTER_SIZE/2));
                		value += dconv_mat[y*FILTER_SIZE+x]*dsrc_img_[index+xOff+yOff];
            		}
        	}
        
		dout_img_[index] = value;
	}
	//if it's in the edge keep the same value
	else{
		dout_img_[index] = dsrc_img_[index];
	}

}
