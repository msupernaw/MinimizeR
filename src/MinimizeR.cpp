
#include <Rcpp.h>
using namespace Rcpp;
#include "../inst/include/minimizeR.hpp"

// [[Rcpp::export]]
List rcpp_hello_world() {

    CharacterVector x = CharacterVector::create( "foo", "bar" )  ;
    NumericVector y   = NumericVector::create( 0.0, 1.0 ) ;
    List z            = List::create( x, y ) ;

    return z ;
}


// [[Rcpp::export]]
Rcpp::List MinimizeR(Rcpp::NumericVector par,
                     Rcpp::Function fn,
                     Rcpp::Function gr,
                     Rcpp::Nullable<Rcpp::List> control = R_NilValue){

  int iprint =10;
  size_t max_iterations =1000;
  double tolerance = 1e-4;
  bool verbose = true;
  size_t max_line_searches = 1000;
  Rcpp::NumericVector minb;
  Rcpp::NumericVector maxb;
  bool estimate_hessian = false;
  bool bounded = false;
  std::string routine = "l-bfgs";
  std::shared_ptr<optimization_routine> minimizer;

  enum minimizer_routine{
    LBFGS = 0,
    BFGS,
    UNKNOWN
  };

  Rcpp::List ctrl(control);
  if (control.isNotNull()) {
    if (ctrl.containsElementNamed("max_iterations")) {
      double maxi = ctrl["max_iterations"];
      if (maxi != 0) {
        max_iterations = maxi;
      }
    }

    if (ctrl.containsElementNamed("max_line_searches")) {
      double maxi = ctrl["max_line_searches"];
      if (maxi != 0) {
        max_iterations = static_cast<size_t>(maxi);
      }
    }

    if (ctrl.containsElementNamed("tolerance")) {
      double tol = ctrl["tolerance"];
      tolerance = tol;
    }

    if (ctrl.containsElementNamed("iprint")) {
      int print_interval = ctrl["iprint"];
      if (print_interval != 0) {
        iprint = print_interval;
      }
    }
    if (ctrl.containsElementNamed("verbose")) {
      bool is_verbose = ctrl["verbose"];
      if (!is_verbose) {
        verbose = is_verbose;
      }
    }

    if (ctrl.containsElementNamed("lb")) {
      minb = Rcpp::as<Rcpp::NumericVector>(ctrl["lb"]);
      bounded = true;
    }

    if (ctrl.containsElementNamed("ub")) {
      maxb =  Rcpp::as<Rcpp::NumericVector>(ctrl["ub"]);
      bounded = true;
    }

    if (ctrl.containsElementNamed("hessian")) {
      estimate_hessian = Rcpp::as<bool>(ctrl["hessian"]);
    }

    if (ctrl.containsElementNamed("routine")) {
      routine = Rcpp::as<std::string>(ctrl["routine"]);
    }

  }

  if(routine == "l-bfgs"){
    minimizer = lbfgs::instance;
  }else if(routine == "bfgs"){
    minimizer = bfgs::instance;
  }else if(routine == "frank_wolfe"){
    minimizer = frank_wolfe::instance;
  }




  minimizer->reset();
  minimizer->iprint = iprint;
  minimizer->estimate_hessian = estimate_hessian;
  minimizer->max_iterations  = max_iterations;
  minimizer->tolerance = tolerance;
  minimizer->of.fn = std::make_shared<Rcpp::Function>(fn);
  minimizer->of.gr = std::make_shared<Rcpp::Function>(gr);
  minimizer->max_line_searches = max_line_searches;

  std::valarray<double> P(par.size());
  std::valarray<double> MINB(P.size());
  std::valarray<double> MAXB(P.size());



  if(bounded){

    for(size_t i =0; i < P.size(); i++){
      P[i] = par[i];
      MINB[i] = minb[i];
      MAXB[i] = maxb[i];
    }
  }else{
    for(size_t i =0; i < P.size(); i++){
      P[i] = par[i];
      MINB[i] = -1.0*std::numeric_limits<real_t>::infinity();
      MAXB[i] = std::numeric_limits<real_t>::infinity();
    }
  }
  minimizer->initialize_parameter_set(P, MINB, MAXB);


  Rcpp::List ret = minimizer->run();
 return ret;

}
