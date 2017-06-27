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

test_that("Vector can be loaded and retrieved", {  
  cu <- new(Cuda)
  x <- rnorm(20)
  x_dev <- cu$h2dVector(x)
  copy <- cu$d2hVector(x_dev)
  expect_true(all.equal(x, copy))
})

test_that("Matrix can be loaded and retrieved", {  
  cu <- new(Cuda)
  x <- matrix(rnorm(20), ncol=4, nrow=5)
  x_dev <- cu$h2dMatrix(x)
  copy <- cu$d2hMatrix(x_dev)
  expect_true(all.equal(x, copy))
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
  cu$launchKernel(list(1,1,1), list(3,2,1), list(5, 4, xp, yp), "kernexec")
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

#void kernexec(double nrow, double ncol, // matrix dimensions
#              double *pos,              // "positive" rows of matrix
#              double l,                 // length of positive rows array
#              double *x,                // matrix of ranked data
#              double *out)              // output matrix

test_that("Matrix lookup works", {  
  cu <- new(Cuda)  
  set.seed(1000)
  ix <- sample(1:1000, 500)
  x <- matrix(rnorm(1000000), nrow=1000, ncol=1000)
  xr <- floor(apply(x, 2, rank))
  r <- matrix(0, nrow=1000, ncol=1000)
  xrp <- cu$h2dMatrix(xr);
  ixp <- cu$h2dVector(ix-1);

  pen = -1 / (nrow(x) -length(ix));
  inc = 1 / length(ix);
  
  rp <- cu$h2dMatrix(r);
  cu$loadKernel("inst/extdata/ks.cu")
  cu$launchKernel(list(100, 10, 1), list(10, 100, 1), list(1000, 1000, ixp, 50, pen, inc, xrp, rp))
  res <- cu$d2hMatrix(rp)
  expect_true(all.equal(x, res))
})

f1 <- function() {
  rp <- cu$h2dMatrix(r);
  cu$launchKernel(list(100, 10, 1), list(10, 100, 1), list(1000, 1000, ixp, 50, pen, inc, xrp, rp))
  res <- cu$d2hMatrix(rp)
}

system.time(f1())

ks <- function(r, ix) {
  inc <- 1/length(ix)
  pen <- -1 / (length(r) - length(ix))
  k <- rep(pen, length(r))
  k[floor(r[ix])] <- 0
  k[floor(r[ix])] <- k[floor(r[ix])] + inc
  k
}

system.time(tt <- cbind(apply(xr, 2, ks, ix)))

