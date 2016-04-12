//Angular to Linear Image Transformation kernel

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;


__kernel void downscale(__read_only image2d_t In_image , __write_only image2d_t Out_image){
	

	int2 in_pos=(int2)(get_global_id(0),get_global_id(1));
	float4 value;
	int2 out_pos;
	out_pos = in_pos/4;
	value=read_imagef(In_image,sampler,in_pos);
	
 	write_imagef(Out_image,out_pos,value);
	
}
