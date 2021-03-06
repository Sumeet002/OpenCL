
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void borders_kernel(__read_only image2d_t dsrc_img_ , __write_only image2d_t dout_img_ , const int W , const int H , const int B){


	const int2 pos = {get_global_id(0),get_global_id(1)};
	
	float4 value = (float4)0.0f;

	//Mirror symmetry
	//C1 C2 ...CN => C2 C1 | C1 C2 .....CN | CN CN-1

	if(pos.x >= B && pos.x <= 2*B){//create left mirrored border
		value = read_imagef(dsrc_img_,sampler,(int2)(pos.x-B,pos.y));
		write_imagef(dout_img_, (int2)(2*B-pos.x, pos.y), value);
		write_imagef(dout_img_,(int2)(pos.x,pos.y),value);
	}else if(pos.x >= W && pos.x <= W+B){//create right mirrored border
		value = read_imagef(dsrc_img_,sampler,(int2)(pos.x-B,pos.y));
		write_imagef(dout_img_,(int2)((2*W+2*B - pos.x),pos.y),value);
		write_imagef(dout_img_,(int2)(pos.x,pos.y),value);
	}else if(pos.x >= B && pos.x <= W+B){//center image
		value = read_imagef(dsrc_img_,sampler,(int2)(pos.x-B,pos.y));
		write_imagef(dout_img_,(int2)(pos.x,pos.y),value);
	}

}
