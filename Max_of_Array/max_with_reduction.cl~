kernel void max_with_reduction(global const float *A,local float *scratch,global float *B){

	int gid = get_global_id(0);
	int tnum = get_local_id(0); //thread number
	int wgNum = get_group_id(0); //work-group number
	int numItems=get_local_size(0);

	scratch[tnum]=A[gid];

	//all threads execute this simultaneously
	for(int offset = 1 ; offset<numItems;offset<<=1){
		int mask=(offset<<1)-1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if((tnum & mask) == 0){
			float other = scratch[tnum+offset];
			float mine = scratch[tnum];
			scratch[tnum]=(mine>other)?mine:other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	
	if(tnum==0)
		B[wgNum]=scratch[0];	

}
