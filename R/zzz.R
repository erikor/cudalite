#' cudalite
#' 
#' Lite weight interface to CUDA
#' 
#' @docType package
#' @author Eric J. Kort 
#' @import Rcpp
#' @importFrom Rcpp evalCpp
#' @useDynLib cudalite
#' @name cudalite
loadModule("cuda", TRUE)


#' Cuda class method: loadKernel
#' 
#' Read a kernel from source file, compile it, and load it on device.
#' 
#' @param fn Name (including path) of kernel source file (typically .cu) which
#'   defines a \code{kernexec} function.
#'   
#' @details \bold{Note:} the name of the entry function for your kernel MUST be 
#' 'kernexec', since that is what will eventually be called when the kernel is
#' launched.  More flexible names may be supported in the future.
#' 
#' If the kernel fails to compile, an error will be thrown but the output will
#' not be terribly helpful for debugging.  In this event, try compiling the
#' kernel file on the command line with \code{nvcc yourkernel.cu}.  This will
#' always produce an error about missing \code{main} entry point, but any other
#' errors will need to be fixed.
#' 
#' Simple example kernels are in the "extdata" folder of this package (for a
#' listing try \code{dir(system.file("extdata", package="cudalite"))})
#' 
#' @examples 
#' \dontrun{
#' cu <- new(Cuda)
#' k <- system.file("extdata", "hello.cu", package="cudalite")
#' cu$loadKernel(k)
#' }
#' @name loadKernel
NULL

#' Cuda class method: launchKernel
#' 
#' Launch a loaded kernel with provided arguments.
#' 
#' @param grid  Three member list giving x, y, and z dimensions (in blocks) of
#'   grid.  The list members do not need to be named, but must be in x, y, z
#'   order.
#'   
#' @param block  Three member list giving x, y, and z dimensions (in threads) of
#'   each block. The list members do not need to be named, but must be in x, y,
#'   z order.
#'   
#' @param args  List of numerics and XPtrs to device memory (as returned by h2dVector & h2dMatrix). Type and 
#'   order of arguments must match kernel signature, and there is no checking 
#'   that this is the case.  Incorrect argument specification may either result 
#'   in memory faults or, worse, silent but unforeseen behavior.
#'   
#' @examples 
#' \dontrun{
#' cu <- new(Cuda)
#' 
#' # create data matrix and load on device
#' x <- cu$h2dMatrix(matrix(rnorm(20), nrow=5, ncol=4));
#' 
#' # create a matrix to hold results and load on device
#' y <- cu$h2dMatrix(matrix(0, nrow=5, ncol=4));
#' 
#' # load and launch kernel
#' k <- system.file("extdata", "stride.cu", package="cudalite")
#' cu$loadKernel(k)
#' cu$launchKernel(list(1,1,1), list(3,2,1), list(5, 4, x, y))
#' 
#' #retrieve results
#' res <- cu$d2hMatrix(y)
#' }
#' @name launchKernel
NULL

    
#' Cuda class methods: h2dVector, h2dMatrix
#' 
#' Load a numeric vector or matrix to device.
#' 
#' @param x Vector or Matrix (of type numeric) to load onto the device.  Memory
#'   is automatically allocated prior to loading.
#'   
#' @return An XPtr to the device pointer.  More precisely, an XPtr to a cuData
#'   object that encapsulates the device pointer along with the data's original
#'   dimensions so it can be properly formatted when subsequently retrieved.
#'   
#' @details Note that when the object returned by this function is garbage
#'   collected, the device memory will be automatically freed.  However, garbage
#'   collection occurs "sometime in the future" after the object is destroyed,
#'   so if you are allocating many data items on the device, you may wish to
#'   expliticly call the Cuda class method \code{dFree} to free it.
#'   
#' @examples 
#' \dontrun{
#' cu <- new(Cuda)
#' 
#' # create data vector and load on device
#' x <- rnorm(20)
#' x_dev <- cu$h2dVector(x)
#' 
#' # copy our data back from device
#' copy <- cu$d2hVector(x_dev)
#' all.equal(x, copy)
#' }
#' @usage 
#' cu <- new(Cuda) 
#' cu$h2dVector(x) 
#' ch$h2dMatrix(x)
#' @name h2d
#' @aliases h2dVector h2dMatrix
NULL


#' Cuda class methods: d2hVector, d2hMatrix
#' 
#' Retrieve vector or matrix data from device.
#' 
#' @param x XPtr to device memory, such as returned by \code{\link{h2dVector}}
#'   or \code{\link{h2dMatrix}}
#' @return A vector or matrix containing the data retrieved from the device.
#'   
#' @examples 
#' \dontrun{
#' cu <- new(Cuda)
#' 
#' # create data vector and load on device
#' x <- rnorm(20)
#' x_dev <- cu$h2dVector(x)
#' 
#' # copy our data back from device
#' copy <- cu$d2hVector(x_dev)
#' all.equal(x, copy)
#' }
#' @usage 
#' cu <- new(Cuda) 
#' cu$d2hVector(x) 
#' ch$d2hMatrix(x)
#' @name d2h
#' @aliases d2hVector d2hMatrix
NULL

