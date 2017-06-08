# usage:  testthat::auto_test("R/", "tests/testthat/")
# or ... test_dir("tests/testthat")

cu <- new(Cuda)

context("Class instantiation")
test_that("Object can be created", {  
  expect_is(cu,  "Rcpp_Cuda")
})

context("Kernels")
test_that("Kernel can be loaded", {  
  k <- system.file("extdata", "matrixtest.cu", package="cudalite")
  expect_true({cu$loadKernel(k); TRUE})
})

context("Data")
test_that("Data can be loaded", {  
  m <- matrix(1, nrow=10000, ncol=10000)
  expect_true({cu$loadMatrix(m); TRUE})
})
