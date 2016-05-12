
__kernel void conv_kernel(const __global uchar* dsrc_img_ , __global uchar* dout_img_, __constant float* dconv_mat, const int W, const int H, const int size ){

	/*int index = get_global_id(0);
	dout_img_[index] = dsrc_img_[index];*/
	
	int index = get_global_id(0);
	int center = size/2;


	//pass the pixel through the kernel if it can be centered inside it
	if(index >= W*(size-center) + center && index < W*H-W*(size-center)-center){
		
		int value=0;
		for(int y=0; y < size ; y++){
			int yOff = W*(y-center);
            		for(int x=0;x<size;x++){
				int xOff = (x-center);
                		value += dconv_mat[y*size+x]*dsrc_img_[index+xOff+yOff];
            		}
        	}
        
		dout_img_[index] = value;
	}
	//if it's in the edge keep the same value
	else{
		dout_img_[index] = dsrc_img_[index];
	}

}
