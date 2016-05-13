__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

float FilterValue(__constant const float* filterWeights,
	const int x, const int y ,const int center)
{
	return filterWeights[(x+center) + (y+center)*(center*center + 1)];
}


__kernel void conv_kernel(__read_only image2d_t dsrc_img_ , __write_only image2d_t dout_img_ , __constant float* dconv_mat , const int size ){


	const int2 pos = {get_global_id(0), get_global_id(1)};
	int center = size/2;

    	float4 sum = (float4)(0.0f);

    	for(int y = -center; y <= center; y++){
        	for(int x = -center; x <= center; x++){
            		sum += FilterValue(dconv_mat, x, y,center)*read_imagef(dsrc_img_, sampler, pos + (int2)(x,y));
        	}
    	}	

    	write_imagef(dout_img_, (int2)(pos.x, pos.y), sum);


}
