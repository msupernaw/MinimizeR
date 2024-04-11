
\name{minimizR}
\alias{minimizR}
\title{minimizR: Optimization using the L-BFGS routine}
\description{
Unconstrained and box-constrained optimization using the L-BFGS routine.
}
\usage{
      minimizR(start, objective, gradient, control = list())
}
\arguments{
\item{start}{Numeric vector, initial values for the parameters to be optimized.}
\item{objective}{Function to be minimized. Must take a numeric vector as an argument. Must return a scalar value.}
\item{gradient}{Gradient function for the function being minimized. Must take a numeric vector as an argument. Must return a numeric vector.}
\item{control}{An optional list of control variables.}
}
\section{Control Parameters}{
  Possible names in the \code{control} list and their default values
  are:
  \describe{
  \item{max_iterations}{The maximum number of iterations the routine will take finding a minimum. Default is 1000.}

  \item{routine}{options include:
                 l-bfgs,
                 bfgs,
                 frank-wolfe
                 cg,
                 newton,
                 ga,
                 sa, default is l-bfgs}

  \item{tolerance}{Convergence criteria. When the maximum gradient component is less than or equal to this value, the routine will return the results. Default is 1e-4.}

  \item{verbose}{If TRUE, runtime information will be printed to the console. Default is TRUE. }

 \item{iprint}{Print interval. Default is every 10 iterations.}

 \item{lb}{numeric vector of lower bounds. Must be the same size as start. }

\item{ub}{numeric vector of upper bounds. Must be the same size as start. }

 \item{hessian}{If TRUE, the hessian matrix will be included in the output. Default is FALSE. }
  }
}
\examples{\donttest{
library(minimizR)

#Rosenbrock function
objective <- function(x) {
  return( 100 * (x[2] - x[1] * x[1])^2 + (1 - x[1])^2 )
}

#Rosenbrock gradient function
gradient <- function(x) {
  return( c( -400 * x[1] * (x[2] - x[1] * x[1]) - 2 * (1 - x[1]), 200 * (x[2] - x[1] * x[1]) ) )
}

#estimated parameters
x<-c(-1.2, 0)


#minimizR function
opt<-minimizR(x,                   #initial parameters values
             objective,            #objective function
             gradient,             #gradient function
             control = list(       #control list
             tolerance = 1e-4,     #convergence criteria
             verbose = TRUE,       #print status
             iprint = 10,          #print interval
             hessian = TRUE))      #include hessian in the output
print(opt)
}}

\keyword{optimize}
