#include <cudf/table/table.hpp>
#include "spatial_join_test_utility.hpp"

template <typename T>
using SBBox = thrust::pair<thrust::tuple<T, T>, thrust::tuple<T, T>>;

static void HandleCudaError( cudaError_t err,const char *file,int line )
{
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCudaError( err, __FILE__, __LINE__ ))

void bbox_table_to_csv(const std::unique_ptr<cudf::table>& bbox_tbl, const char * file_name,
     double * & h_x1,double * & h_y1,double * & h_x2,double * & h_y2);


