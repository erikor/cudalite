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
  e <- system.file("extdata", "error.cu", package="cudalite")
  expect_true({cu$loadKernel(k); TRUE})
})
