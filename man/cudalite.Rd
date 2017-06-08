\docType{package}
\name{cudalite}
\alias{cudalite-package}
\title{The cudalite package}
\section{Overview}{
The cudalite package provides an efficient, thin interface to basic CUDA functionality to facilitate analysis of R matrices and vectors on GPU.  See also the RCUDA package which has similar goals.

The functionality of the cudalite package is encapsulated in the \code{Cuda} class to preserve state between function calls.  The remainder of this document describes the Cuda class and its use.
}
\section{The Cuda class}{

  \subsection{Attributes}{
    \itemize{
      \item{None.}
    }
  }
  
  \subsection{Methods}{
    \itemize{

      \item \bold{\code{loadKernel(fn)}} Read the kernel specified by \code{fn}, compile it, and load to device.  
      \subsection{Arguments}
      \itemize{
        \item{\bold{\code{fn}}: The path to the kernel source file, typically a ".cu" file.}
        }
      \subsection{Return Value}
      None.  Called for its side effect of compiling and loading the kernel.

      \item \bold{\code{loadMatrix(x)}} Load a numeric matrix to device. 
      \subsection{Arguments}
      \itemize{
        \item{\bold{\code{x}}: The numeric matrix to be loaded to the device.  At present, it is up the end user to ensure the 
        matrix is not too large to fit into device memory.}
        }
      \subsection{Return Value}
      None.  Called for its side effect of loading data onto the GPU.
    }
  }
}
\usage{
library(cudalite)
cu <- new(Cuda)
k <- system.file("extdata", "matrixtest.cu", package="cudalite")
cu$loadKernel(k)
m <- matrix(1, ncol=3, nrow=3)
cu$loadMatrix(m)
}
}

