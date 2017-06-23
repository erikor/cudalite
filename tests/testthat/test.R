# usage:  testthat::auto_test("R/", "tests/testthat/")
# or ... test_dir("tests/testthat")

cu <- new(Cuda)

context("Class instantiation")
test_that("Object can be created", {  
  expect_is(cu,  "Rcpp_Cuda")
})

context("Data")
test_that("Data can be loaded", {  
  m <- matrix(1, nrow=10000, ncol=10000)
  expect_true({cu$h2dMatrix(m); TRUE})
})

context("Kernels")
test_that("Kernel can be loaded", {  
  k <- system.file("extdata", "hello.cu", package="cudalite")
  expect_true({cu$loadKernel(k); TRUE})
})

test_that("Kernel can be launched", {  
  cu <- new(Cuda)  
  x <- matrix(rnorm(20), nrow=5, ncol=4)
  y <- matrix(0, nrow=5, ncol=4)
  xp <- cu$h2dMatrix(x);
  yp <- cu$h2dMatrix(y);
  k <- system.file("extdata", "stride.cu", package="cudalite")
  cu$loadKernel(k)
  cu$launchKernel(list(1,1,1), list(3,2,1), list(5, 4, xp, yp))
  res <- cu$d2hMatrix(yp)
  expect_true(all.equal(x, res))
})

hold <- function() {
  library(cudalite)
  cu <- new(Cuda)
  k <- system.file("extdata", "stride.cu", package="cudalite")
  cu$loadKernel(k)
  
  x <- matrix(rnorm(2000), nrow=100, ncol=20)
  y <- matrix(1.25, nrow=100, ncol=20)
  xp <- cu$h2dMatrix(x);
  yp <- cu$h2dMatrix(y);
  
  cu$launchKernel(list(1,1,1), list(10,100,1), list(100, 20, xp, yp))
  res <- cu$d2hMatrix(yp)
  all.equal(x, res)
}