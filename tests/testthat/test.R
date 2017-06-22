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
  cu$launchKernel(list(1,1,1), list(2,2,1), list(a=5))
  expect_true({TRUE})
})

test_that("Kernel can be launched", {  
  x <- matrix(rnorm(100000), nrow=1000, ncol=100)
  y <- matrix(0, nrow=1000, ncol=100)
  xp <- cu$h2dMatrix(x);
  yp <- cu$h2dMatrix(y);
  k <- system.file("extdata", "stride.cu", package="cudalite")
  cu$loadKernel(k)
  cu$launchKernel(list(10,10,1), list(10,10,1), list(1000, 100, xp, yp))
  res <- cu$d2hMatrix(yp)
  expect_true(all.equal(x, res))
})

