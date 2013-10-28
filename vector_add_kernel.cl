__kernel void vector_add(__global int *A, __global int *B, __global int *C) {
    
    // Get the index of the current element
    int i = get_global_id(0);
    float if= i;
    int j = sin(if);

    // Do the operation
    C[i] = A[i] + j * B[i];
}
