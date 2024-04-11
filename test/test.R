library(MinimizeR)
library(Rcpp)

Cos<-function(x){
  return(cos(x[1]))
}

Cos_dx<-function(x){
  return(-sin(x[1]))
}

x<-c(0.60)

lb_<-c(0.5)
ub_<-c(4.5)
opt<-MinimizeR(x,Cos, Cos_dx
               , control = list(routine = "frank_wolfe",
                                             hessian = TRUE,
                                             max_iterations = 1000,
                                             iprint = 10000000,
                                             lb = c(0.5),
                                             ub = c(4.5),
                                             max_line_searches = 1000))
opt
Cos(opt$parameters)
Cos_dx(opt$parameters)
x
