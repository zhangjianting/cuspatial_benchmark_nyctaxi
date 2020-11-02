#include <cuspatial/error.hpp>
#include <cuspatial/point_quadtree.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

int main()
{
  const int8_t max_depth = 3;
  uint32_t min_size      = 50;
  double scale         = 400.0;
  double x_min =  -1276.4949079170394, x_max = 1134.877311993804, y_min = -1111.7015761122134, y_max = 1191.572387431971;   

  cudf::test::fixed_width_column_wrapper<double> x(
    { -318.71851278,  -198.63590716,   -66.44028879,  -148.8954397 ,
        -154.50648452,  -838.00190316,   576.16578239,   539.80929602,
        -406.6821296 ,  -733.2121639 ,   260.53243823,  -287.89398491,
        70.97658166,  -159.66420857,   345.76937554,   347.37457183,
        -362.79868923,  -691.6819777 ,  -791.46919867,   305.18968955,
        -594.42962889,  -253.40817715,  -298.15701923,   -26.28364813,
        -968.13990292,    94.3892984 ,   261.94551192,    44.21104352,
        -155.44308585,    48.70008313,   199.52317282, -1386.29637821,
         977.95615413,   195.04666134,  -326.20429119,  -195.47668759,
         246.87088867,   -58.05196952, -1015.34223389,  1032.24643068,
         -55.27032862,   510.08635586,  -346.02492389,   768.18852712,
         143.17184445,   304.42191724,  -522.62668307,   605.57264484,
         344.90908227,   650.92311478,  -314.04377982,  -240.51355923,
        1151.95834884,  -530.00791136,   -67.97485034,   568.4456813 ,
        48.86248386,   291.47683988,  -199.72451463,   185.02794392,
        -653.26342587,   829.06533981,   -59.08202256,  -340.089102  ,
         333.19154102,  -230.35989369,  -667.1292357 ,  -673.3587529 ,
         346.88657635,   -79.78671907,   -66.85077983,   538.87190299,
        -563.41290438,  -365.33887643,  -192.43990459,    47.17579466,
         -21.08572565,  -143.44359619,   -30.81320105,   -53.65263815,
        -359.80219428,  -406.49649428,   137.25817886,  -445.4575415 ,
        -578.6776296 ,  -156.14612556,   -78.83350808,  1128.36174865,
        -352.35013793,   471.63036248,   373.5941671 ,  -594.4724776 ,
         386.6264887 ,  -591.9403201 , -1329.586119  ,   303.15976218,
        -877.94529172,   225.4672309 ,  -342.00544887,   829.77539809,
         534.25469966,  -226.69290193,  -343.91880551,  -607.03870155,
        -220.46131615,  -140.17774759,  -182.3467722 ,    78.35192764,
         289.26074886,   174.8272285 ,  -382.07196195,  -718.8957369 ,
         682.26592405,  -344.72459227,  -326.14679997,  -260.59465615,
        -921.53477508,  -238.98700202,  -239.827907  ,   310.17914917});

  cudf::test::fixed_width_column_wrapper<double> y(
    {349.22857455,     1.88544454,   465.92418706,   169.9824919 ,
        -7.8410558 ,    80.46408415,   -95.32674679,  -197.42475702,
        -133.86676845,  -564.00566574,   140.22085266,  -496.56180546,
         420.81563204,  -124.72929008,    24.74749083,   246.91838814,
         321.65723253,  -785.31170432,  -103.45183808,   440.08945604,
        -849.05290972,   193.6402377 , -1127.7821147 ,  -511.25342182,
        19.31527592,  -828.35755116,  -492.75536884,  -735.91750373,
         824.0674661 ,    82.11387774,   283.64513893,  -111.33755026,
        -176.71587438,  -808.23709433,  -145.91868137,  -380.74610591,
         428.96196215,   570.55093333,   733.28935779,   426.27596973,
        -299.32696846,  -557.94849298,   383.33159082,   178.14640874,
        -884.26922534,   177.74089637,   407.25991124,    29.46279459,
         -92.5268355 ,  -403.82424381,  -723.26734978,   400.14897467,
        -154.55722239,  -116.73333077,   866.36059346,   342.25055343,
         185.41250064,    71.03090259,   759.99743038,   859.79465371,
         464.75255574,   291.1122957 , -1047.3015356 ,    61.86095712,
         -65.0534771 ,    46.97661469,   471.52304366, -1369.83858359,
        -284.65602674,   134.95217747,  -233.42277303,  -708.45305656,
         434.48174345,   138.43595292,  -485.55228522,   157.40860226,
         410.79285602,     2.64632315,   400.28240172,    39.13008758,
        -197.61449133,  -579.7102582 ,   -42.96538349,    97.14646902,
         437.91638079,   -57.55373424,   228.70780311,  -482.30600687,
        -391.31457791,   -55.19464951,  -527.31423199,   410.12391866,
         231.56516466,   139.5478822 ,   169.45206261,  1010.52178074,
        -234.43209398, -1100.72064275,    99.65009845,   -25.30177048,
        -258.75952126,  -489.41492968,  -219.5947609 ,    90.66921461,
        -251.40835032,  1206.22683977,  -480.25219082,  -396.55868135,
       -1144.31002001,   125.74220751, -1008.2033139 ,  -269.72731669,
        -137.83526728,  -354.86398292,   869.43633873,   497.19719566,
         659.56843815,  -441.20940927,   564.29703226,   248.00047317});

    uint32_t repeat=10;
    std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> quadtree_on_points quadtree_pair;
    for(uint32_t k=0;k<repeat;k++)
    {
        quadtree_pair= cuspatial::quadtree_on_points(x, y, x_min, x_max, y_min, y_max, scale, max_depth, min_size);      
        std::unique_ptr<cudf::table> &quadtree = std::get<1>(quadtree_pair);

        CUSPATIAL_EXPECTS(
        quadtree->num_columns() == 5,
        "a quadtree table must have 5 columns (keys, levels, is_node, lengths, offsets)");  

        const uint32_t  *d_key=quadtree->view().column(0).template data<uint32_t>();
        const uint8_t  *d_lev=quadtree->view().column(1).template data<uint8_t>();
        const bool *d_sign=quadtree->view().column(2).template data<bool>();
        const uint32_t *d_len=quadtree->view().column(3).template data<uint32_t>();
        const uint32_t *d_fpos=quadtree->view().column(4).template data<uint32_t>();

        uint32_t num_quad=quadtree->view().num_rows();
        uint32_t *h_key=new uint32_t[num_quad];
        uint8_t  *h_lev=new uint8_t[num_quad];
        bool     *h_sign=new bool[num_quad];
        uint32_t *h_len=new uint32_t[num_quad];
        uint32_t *h_fpos=new uint32_t[num_quad];
        assert(h_key!=nullptr && h_lev!=nullptr && h_sign!=nullptr && h_len!=nullptr && h_fpos!=nullptr);

        EXPECT_EQ(cudaMemcpy(h_key,d_key,num_quad*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);
        EXPECT_EQ(cudaMemcpy(h_lev,d_lev,num_quad*sizeof(uint8_t),cudaMemcpyDeviceToHost),cudaSuccess);
        EXPECT_EQ(cudaMemcpy(h_sign,d_sign,num_quad*sizeof(bool),cudaMemcpyDeviceToHost),cudaSuccess);
        EXPECT_EQ(cudaMemcpy(h_len,d_len,num_quad*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);
        EXPECT_EQ(cudaMemcpy(h_fpos,d_fpos,num_quad*sizeof(uint32_t),cudaMemcpyDeviceToHost),cudaSuccess);

        printf("\t round:%d num_quad=%d\n",k,num_quad);
        for(uint32_t i=0;i<num_quad;i++)
           printf("%10d %10d %10d %10d %10d\n",h_key[i],h_lev[i],h_sign[i],h_len[i],h_fpos[i]);
    }
    return (0);
}
