kernel void hist_sim(global const uchar* Array, global int* Hist, int Width){ // Histogram calclulation that operates in Global Memory
	int id = get_global_id(0);
	int bin_index = (int)Array[id] / Width;
	atomic_inc(&Hist[bin_index]);
}

kernel void hist_loc_atom(global const uchar* Array, global uint* Hist, local uint* localHist, int Width){ // Local Histogram calclulation that operates in Local Memory with Atomic Operations
	int id = get_global_id(0);
	int local_id = get_local_id(0);
	int binIndex = (int)Array[id] / Width;
	localHist[local_id] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&localHist[binIndex]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if(local_id < 256){ // combine all local histograms into a global one
		atomic_add(&Hist[local_id], localHist[local_id]);
	}
}

kernel void cumul_hist(global int* Hist, global int* Hist_Cumul){
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++){
		atomic_add(&Hist_Cumul[i], Hist[id]);
	}
}

kernel void Hillis_Steele_Scan(global int* Hist, global int* Hist_Cumul){ // Inclusive Hillis-Steele Span-efficient Scan
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2){
		Hist_Cumul[id] = Hist[id];
		if(id >= stride)
			Hist_Cumul[id] += Hist[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

		C = Hist; Hist = Hist_Cumul; Hist_Cumul = C; // swap Hist & Hist_Cumul between steps
	}
}

kernel void Blelloch_Scan(global int* Hist, global int* Hist_Out){ // Exclusive Blelloch Work-efficient Scan
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			Hist[id] += Hist[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	}

	//down-sweep
	if (id == 0)
		Hist[N-1] = 0; // exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = Hist[id];
			Hist[id] += Hist[id - stride]; // reduce 
			Hist[id - stride] = t; // move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	}

	Hist_Out[id] = Hist[id];
}

kernel void LUT_hist(global int* Hist_Cumul, global int* Hist_LUT, int Width){
	int id = get_global_id(0);
	Hist_LUT[id] = (int)Hist_Cumul[id] * (double)255 / Hist_Cumul[255 / Width];
}

kernel void edit(global uchar* Array, global int* hist_LUT, global uchar* new_image_values, int Width){
	int id = get_global_id(0);
	new_image_values[id] = hist_LUT[(int)Array[id] / Width];
}