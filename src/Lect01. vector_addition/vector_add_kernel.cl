__kernel void vecadd(__global int *A, __global int *B, __global int *C){
	// global idx ¾ò´Â´Ù.
	int idx = get_global_id(0);

	C[idx] = A[idx] + B[idx];
}