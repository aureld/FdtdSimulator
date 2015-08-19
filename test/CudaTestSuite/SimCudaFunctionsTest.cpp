//SimCudaFunctionsTest.cpp: test class for FDTD CUDA functions wrapper
//Aurelien Duval 2015

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "simCuda3D/Cuda_grid.h"
#include "simCuda3D/Cuda_macros.h"
#include "simCuda3D/SimCudaFunctions.h"
#include "common_defs.h"


//param is the id of field (0:ex,1:ey,2:ez,3:hx...)
class SimCudaFunctionsTest : public testing::TestWithParam<int> {
protected:
    virtual void SetUp()
    {
        g = new grid();
        dg = new grid();
        g->nx = 2;
        g->domainSize = 2;
        g->nt = 2;
        g->Ca = new float[2];
        g->Ca[0] = 1.0;     g->Ca[1] = -1.0;
        g->Cb1 = new float[2];
        g->Cb1[0] = 1.0;    g->Cb1[1] = -1.0;
        g->Cb2 = new float[2];
        g->Cb2[0] = 1.0;    g->Cb2[1] = -1.0;
        g->Db1 = new float[2];
        g->Db1[0] = 1.0;    g->Db1[1] = -1.0;
        g->Db2 = new float[2];
        g->Db2[0] = 1.0;    g->Db2[1] = -1.0;
        g->srcField = (float *)cust_alloc(sizeof(float)*g->nt);
        g->ex = (float *)cust_alloc(sizeof(float)*g->domainSize);
        g->ey = (float *)cust_alloc(sizeof(float)*g->domainSize);
        g->ez = (float *)cust_alloc(sizeof(float)*g->domainSize);
        g->hx = (float *)cust_alloc(sizeof(float)*g->domainSize);
        g->hy = (float *)cust_alloc(sizeof(float)*g->domainSize);
        g->hz = (float *)cust_alloc(sizeof(float)*g->domainSize);
    }
    grid *g, *dg;

    
};

TEST_F(SimCudaFunctionsTest, CudaInitGrid_is_invalid_if_grid_is_not_initialized) {
    g = NULL;
    EXPECT_EQ(false, CudaInitGrid(g, dg));
}

TEST_P(SimCudaFunctionsTest, CudaInitGrid_is_invalid_if_a_coef_is_not_initialized) {
    switch (GetParam())
    {
    case 0:default: g->Ca = NULL; break;
    case 1: g->Cb1 = NULL; break;
    case 2: g->Cb2 = NULL; break;
    case 3: g->Db1 = NULL; break;
    case 4: g->Db2 = NULL; break;
    }
    EXPECT_EQ(false, CudaInitGrid(g, dg));
}

TEST_F(SimCudaFunctionsTest, CudaInitGrid_is_invalid_if_source_is_not_initialized) {
    g->srcField = NULL;
    EXPECT_EQ(false, CudaInitGrid(g, dg));
}

TEST_F(SimCudaFunctionsTest, CudaInitGrid_is_valid) {
    EXPECT_EQ(true, CudaInitGrid(g, dg));
}


TEST_P(SimCudaFunctionsTest, CudaInitFields_is_invalid_if_a_field_is_not_initialized) {
    switch (GetParam())
    {
        case 0:default: g->ex = NULL; break;
        case 1: g->ey = NULL; break;
        case 2: g->ez = NULL; break;
        case 3: g->hx = NULL; break;
        case 4: g->hy = NULL; break;
        case 5: g->hz = NULL; break;
    }
    EXPECT_EQ(false, CudaInitFields(g, dg));
}

INSTANTIATE_TEST_CASE_P(InstantiationName, SimCudaFunctionsTest, ::testing::Values(0, 1,2,3,4,5));

TEST_F(SimCudaFunctionsTest, CudaInitFields_is_valid) {
    EXPECT_EQ(true, CudaInitFields(g, dg));
}



