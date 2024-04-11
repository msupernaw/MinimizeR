#ifndef MINIMIZER_HPP
#define MINIMIZER_HPP

#include <Rcpp.h>
#include <valarray>
#include <algorithm>
#include "Port.hpp"

typedef double real_t;

class variable_transformation {
public:
    /**
     * Convert the external value to its internal representation.
     *
     * @param val
     * @param min
     * @param max
     * @return
     */
    virtual real_t External2Internal(real_t val, real_t min_, real_t max_) const = 0;
    /**
     * Convert a variables internal value to its external representation.
     * @param val
     * @param min
     * @param max
     * @return
     */
    virtual real_t Internal2External(real_t val, real_t min_, real_t max_) const = 0;
    /**
     * The derivative of Internal2External.
     * @param val
     * @param min
     * @param max
     * @return
     */
    virtual real_t DerivativeInternal2External(real_t val, real_t min_, real_t max_) const = 0;
};

class tanh_transformation : public variable_transformation {
public:

public:

    tanh_transformation() {
    }

    virtual ~tanh_transformation() {
    }

    virtual real_t External2Internal(real_t val, real_t min_, real_t max_) const {
        return min_ + .5 * (max_ - min_)*(1.0 + std::tanh(val));
    }

    virtual real_t Internal2External(real_t val, real_t min_, real_t max_) const {
        return std::atanh(2.0 * (val - min_) / (max_ - min_) - 1.0);
    }

    virtual real_t DerivativeInternal2External(real_t val, real_t min_, real_t max_)const {
        return 2.0 / ((max_ - min_) * std::pow((1.0 - ((2.0 * (val - min_)) / max_ - min_ - 1.0)), 2.0));
    }

};

class sin_transformation : public variable_transformation {
public:

    sin_transformation() {
    }

    virtual ~sin_transformation() {
    }

    virtual real_t External2Internal(real_t val, real_t min_, real_t max_) const {
        if (val < min_ || val > max_) {
            std::cout << val << " (" << min_ << "," << max_ << ") value exceeds bounds...." << std::endl;
        }
        //            return std::asin((2.0 * val) / (max_ - min_) - min_ / (max_ - min_) - max_ / (max_ - min_));
        return std::asin((2.0 * (val - min_) / (max_ - min_)) - 1.0);
    }

    virtual real_t Internal2External(real_t val, real_t min_, real_t max_) const {
        //            return min_ + (0.5 * (max_ - min_))*(std::sin(val) + 1.);
        return min_ + (std::sin(val) + 1.0)*((max_ - min_) / 2.0);
    }

    virtual real_t DerivativeInternal2External(real_t val, real_t min_, real_t max_)const {
        return 0.5 * ((max_ - min_) * std::cos(val));
        //            return ((max_ - min_) * std::cos(val)) / 2.0;
    }
};

class logit_transformation : public variable_transformation {
public:

    logit_transformation() {
    }

    virtual ~logit_transformation() {
    }

    virtual real_t External2Internal(real_t val, real_t min_, real_t max_)const {
        if (val == min_) {
            val += static_cast<real_t> (1e-8);
        } else if (val == max_) {
            val -= static_cast<real_t> (1e-8);
        }

        real_t p = ((val) - min_) / (max_ - min_);
        return std::log(p / (1.0 - p));
    }

    virtual real_t Internal2External(real_t val, real_t min_, real_t max_) const {
        real_t p = std::exp(val) / (1.0 + std::exp(val));
        return p * (max_ - min_) + min_;
    }

    virtual real_t DerivativeInternal2External(real_t val, real_t min_, real_t max_)const {
        //            return ((max_-min_)*std::exp(val))/(std::exp(val)+1.0)-((max_-min_)*std::exp((2.0*val)))/std::pow((std::exp(val)+1.0),2.0);
        //            return ((max_−min_)*std::exp(val)/(std::exp(val)+1.0)−((max_−min_)*std::exp((2.0*val))/std::pow(std::exp(val)+1.0),2.0);
        return (std::exp(val) * std::log(M_E)*(max_ - min_)) / (std::exp(val) + 1.0)-
                (std::exp(static_cast<real_t> (2.0 * val)) * std::log(M_E)*(max_ - min_)) / std::pow((std::exp(val) + 1), 2.0);
    }
};

struct varibable_info {
    static std::shared_ptr<variable_transformation> transformation;
    real_t value;
    real_t min_m = -std::numeric_limits<double>::infinity();
    real_t max_m = std::numeric_limits<double>::infinity();
    bool bound = false;

    varibable_info(const varibable_info& other) :
    value(other.value), min_m(other.min_m), max_m(other.max_m), bound(other.bound) {
    }

    varibable_info() {
    }

    void update_value(const real_t& v) {
        if (this-> bound) {
            this->value = varibable_info::transformation->Internal2External(v, this->min_m, this->max_m);
        } else {
            this->value = v;
        }
    }

    real_t internal_value() {

        if (this-> bound) {
            return this->varibable_info::transformation->External2Internal(this->value, this->min_m, this->max_m);
        } else {
            return this->value;
        }
    }

    real_t scaled_gradient(const real_t& x) {
        if (this-> bound) {
            return varibable_info::transformation->DerivativeInternal2External(x, this->min_m, this->max_m);
        } else {
            return 1.0;
        }
    }

};

std::shared_ptr<variable_transformation> varibable_info::transformation =
        std::make_shared<sin_transformation>();

class objective_function {
public:
    std::vector<varibable_info> parameters;
    std::shared_ptr<Rcpp::Function> fn;
    std::shared_ptr<Rcpp::Function> gr;

    objective_function() {
    }

    Rcpp::NumericVector parameter_values() {
        Rcpp::NumericVector ret(this->parameters.size());
        for (size_t i = 0; i < ret.size(); i++) {
            ret[i] = parameters[i].value;
        }
        return ret;
    }

    void set(const Rcpp::NumericVector& v) {
        this->parameters.resize(v.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            this->parameters[i].value = v[i];
        }
    }

    void update(const Rcpp::NumericVector& v) {
        this->parameters.resize(v.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            this->parameters[i].update_value(v[i]);
        }
    }

    void update(const std::valarray<real_t>& v) {
        this->parameters.resize(v.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            this->parameters[i].update_value(v[i]);
        }
    }

    void set_bounds(const Rcpp::NumericVector& min_, const Rcpp::NumericVector& max_) {

        if (min_.size() != this->parameters.size()) {
            std::cout << "\"lb\" vector length not equal to parameter vector length." << std::endl;
            return;
        }


        if (max_.size() != this->parameters.size()) {
            std::cout << "\"lb\" vector length not equal to parameter vector length." << std::endl;
            return;
        }

        for (size_t i = 0; i < this->parameters.size(); i++) {
            if (min_[i] > (-1.0 * std::numeric_limits<double>::infinity())) {
                parameters[i].bound = true;
                parameters[i].min_m = min_[i];
            }

            if (max_[i] < (std::numeric_limits<double>::infinity())) {
                parameters[i].bound = true;
                parameters[i].min_m = max_[i];
            }
        }
    }

    void evaluate(real_t& f) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {

            v[i] = this->parameters[i].value;
        }
        Rcpp::Function& F = *fn.get();
        f = Rcpp::as<double>(F(v));
    }

    Rcpp::NumericVector gradient(const Rcpp::NumericVector& x) {

        Rcpp::Function& Gr = *gr.get();
        return Rcpp::as<Rcpp::NumericVector>(Gr(x));
    }

    void gradient(std::valarray<real_t>& g, real_t& maxgc) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        maxgc = -1.0;
        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {
            g[i] = gradient[i];
            if (std::fabs(g[i]) > maxgc) {

                maxgc = std::fabs(g[i]);
            }
        }
    }

    void gradient(real_t& f, std::valarray<real_t>& g) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        Rcpp::Function& F = *fn.get();
        f = Rcpp::as<double>(F(v));

        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {

            g[i] = gradient[i];
        }
    }

    void gradient(real_t& f, std::valarray<real_t>& g, real_t& maxgc) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        Rcpp::Function& F = *fn.get();
        f = Rcpp::as<double>(F(v));

        maxgc = -1.0;
        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {
            g[i] = gradient[i];
            if (std::fabs(g[i]) > maxgc) {

                maxgc = std::fabs(g[i]);
            }
        }

    }

    Rcpp::NumericVector gradient(real_t& maxgc) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }
        maxgc = -1.0;
        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector g = Rcpp::as<Rcpp::NumericVector>(Gr(v));

        for (size_t i = 0; i < g.size(); i++) {

            if (std::fabs(g[i]) > maxgc) {

                maxgc = std::fabs(g[i]);
            }
        }
        return g;

    }

    void gradient(std::vector<real_t>& g, real_t & maxgc) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        maxgc = -1.0;
        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {
            g[i] = gradient[i];
            if (std::fabs(g[i]) > maxgc) {

                maxgc = std::fabs(g[i]);
            }
        }
    }

    void gradient(real_t& f, std::vector<real_t>& g) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        Rcpp::Function& F = *fn.get();
        f = Rcpp::as<double>(F(v));

        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {

            g[i] = gradient[i];
        }
    }

    void gradient(real_t& f, std::vector<real_t>& g, real_t & maxgc) {
        Rcpp::NumericVector v(this->parameters.size());
        for (size_t i = 0; i < this->parameters.size(); i++) {
            v[i] = this->parameters[i].value;
        }

        Rcpp::Function& F = *fn.get();
        f = Rcpp::as<double>(F(v));

        maxgc = -1.0;
        Rcpp::Function& Gr = *gr.get();
        Rcpp::NumericVector gradient = Rcpp::as<Rcpp::NumericVector>(Gr(v));
        g.resize(gradient.size());
        for (size_t i = 0; i < g.size(); i++) {
            g[i] = gradient[i];
            if (std::fabs(g[i]) > maxgc) {

                maxgc = std::fabs(g[i]);
            }
        }
    }

};

class optimization_routine {
public:

    objective_function of;
    real_t tolerance = 1e-4;
    std::valarray<real_t> x;
    std::valarray<real_t> best;
    std::valarray<real_t> gradient;
    int max_iterations = 5000;
    unsigned int max_line_searches = 5000;
    int iprint = 10;
    bool estimate_hessian = false;
    real_t maxgc = -999.0;

    virtual Rcpp::List run() = 0;

    void reset() {
        of.parameters.clear();
    }

    void initialize_parameter_set(const std::valarray<real_t>& p,
            const std::valarray<real_t>& min,
            const std::valarray<real_t>& max) {

        this->of.parameters.resize(p.size());

        for (size_t i = 0; i < p.size(); i++) {
            of.parameters[i].value = p[i];
            if (min[i] != -1.0 * std::numeric_limits<real_t>::infinity()) {
                of.parameters[i].min_m = min[i];
                of.parameters[i].bound = true;
            }

            if (max[i] != std::numeric_limits<real_t>::infinity()) {

                of.parameters[i].max_m = max[i];
                of.parameters[i].bound = true;
            }
        }

    }

    Rcpp::NumericMatrix get_estimated_hessian() {

        int n = this->of.parameters.size();

        Rcpp::NumericVector x(n);
        for (size_t i = 0; i < n; i++) {
            x[i] = this->of.parameters[i].value;
        }

        Rcpp::NumericMatrix hessian(n, n);
        double epsilon = 1e-6;

        Rcpp::NumericVector grad_plus, grad_minus;
        for (int i = 0; i < n; ++i) {
            Rcpp::NumericVector x_plus = Rcpp::clone(x);
            x_plus[i] += epsilon;
            grad_plus = this->of.gradient(x_plus);

            Rcpp::NumericVector x_minus = clone(x);
            x_minus[i] -= epsilon;
            grad_minus = this->of.gradient(x_minus);

            Rcpp::NumericVector hessian_col = (grad_plus - grad_minus) / (2 * epsilon);
            hessian(_, i) = hessian_col;
        }
        return hessian;
    }

    real_t norm(const std::valarray<real_t>& v) {

        real_t ret = 0.0;
        unsigned int i;
        for (i = 0; i < v.size(); i++) {

            ret += v[i] * v[i];

        }
        return std::sqrt(ret);
    }

    real_t dot(const std::valarray<real_t>& a, const std::valarray<real_t>& b) {
        real_t ret = 0.0;

        for (size_t i = 0; i < a.size(); i++) {

            ret += a[i] * b[i];
        }

        return ret;
    }

    const std::valarray<real_t> Column(std::valarray<std::valarray<real_t> > &matrix, size_t column, size_t length) {

        std::valarray<real_t> ret(length);

        for (int i = 0; i < ret.size(); i++) {

            ret[i] = matrix[i][column];
        }
        return ret;
    }

    bool line_search(Rcpp::Function fn,
            Rcpp::Function gr,
            real_t& fx,
            real_t& function_value,
            std::valarray<real_t>& x,
            std::valarray<real_t>& best,
            std::valarray<real_t>& z,
            std::valarray<real_t>& gradient,
            std::valarray<real_t>& wg,
            real_t& maxgc, int& i,
            int& max_iterations,
            int max_line_searches = 1000) {

        real_t descent = 0;

        int nops = this->of.parameters.size();
        std::valarray<real_t> ng(nops);


        for (size_t j = 0; j < nops; j++) {
            descent += z[j] * wg[j];
        }//end for

        real_t norm_g = this->norm(gradient);
        real_t relative_tolerance = this->tolerance * std::max<real_t > (real_t(1.0), norm_g);

        descent *= real_t(-1.0); // * Dot(z, g);
        if ((descent > real_t(-0.00000001) * relative_tolerance /* tolerance relative_tolerance*/)) {
            z = wg + .001;

            max_iterations -= i;
            i = 0;

            descent = -1.0 * dot(z, wg);
        }//end if

        real_t step = i ? 1.0 : (1.0 / norm_g);

        if (step != step) {
            step = 1.0;
        }

        bool down = false;

        int ls;




        for (int j = 0; j < of.parameters.size(); j++) {
            best[j] = of.parameters[j].value;
        }


        for (ls = 0; ls < max_line_searches; ++ls) {

            // Tentative solution, gradient and loss
            std::valarray<real_t> nx = x - step * z;

            for (size_t j = 0; j < nops; j++) {

                // if (nx[j] != nx[j]) {
                //     //                    std::cout<<"Warning.....nx[j] != nx[j]\n";
                // }
                of.parameters[j].update_value(nx[j]);
            }

            //line_search

            // this->of.update(nx);
            this->of.evaluate(fx);

            if (fx <= function_value + tolerance * real_t(10e-4) * step * descent) { // First Wolfe condition

                for (size_t j = 0; j < nops; j++) {
                    best[j] = of.parameters[j].internal_value();
                }

                of.gradient(fx, ng);


                if (down || (-1.0 * dot(z, ng) >= 0.9 * descent)) { // Second Wolfe condition
                    x = nx;
                    gradient = ng;
                    function_value = fx;
                    return true;
                } else {
                    step *= 10.0; //2.0; //10.0;
                }
            } else {
                step /= 10.0; //*= .5; ///
                down = true;
            }
        }

        for (size_t j = 0; j < nops; j++) {
            of.parameters[j].value = best[j];
        }

        return false;
    }

};

class lbfgs : public optimization_routine {
public:

    static std::shared_ptr<lbfgs> instance;
    unsigned int max_history = 500;

    lbfgs() {
    }

    virtual ~lbfgs() {
    }

    virtual Rcpp::List run() {
        Rcpp::List ret;
        ret["method"] = "l-bfgs";
        int nops = this->of.parameters.size();



        this->x.resize(nops);
        this->best.resize(nops);
        this->gradient.resize(nops);

        for (int i = 0; i < nops; i++) {
            if (this->of.parameters[i].bound) {
                this->x[i] = this->of.parameters[i].internal_value();
            } else {
                this->x[i] = this->of.parameters[i].value;
            }
            this->gradient[i] = 0;
        }
        //
        //
        std::valarray<real_t> wg(nops);
        std::valarray<real_t> nwg(nops);
        std::valarray<real_t> ng(nops);


        //initial evaluation
        real_t fx(0.0);
        this->of.evaluate(fx);
        real_t function_value = fx;
        //
        //Historical evaluations
        std::valarray<real_t> px(nops);
        std::valarray<real_t> pg(nops);
        std::valarray<std::valarray<real_t> > dxs(std::valarray<real_t > (this->max_history), nops);
        std::valarray<std::valarray<real_t> > dgs(std::valarray<real_t > (this->max_history), nops);
        //search direction
        std::valarray<real_t> z(nops);

        this->of.gradient(fx, this->gradient, this->maxgc);


        std::valarray<real_t> p(this->max_history);
        std::valarray<real_t> a(this->max_history);
        int no_progress_count = 0;
        int i;
        for (int iteration = 0; iteration < this->max_iterations; iteration++) {
            i = iteration;

            for (int j = 0; j < nops; j++) {
                wg[j] = this->of.parameters[j].scaled_gradient(
                        this->of.parameters[j].internal_value()) * this->gradient[j];
            }

            if (((i % this->iprint) == 0) && i > 0) {
                std::cout << "Iteration " << i << "\n";

            }

            if (this->maxgc < this->tolerance) {
                if ((i % this->iprint) == 0) {
                    std::cout << "Iteration " << i << "\n";

                }


                Rcpp::NumericVector Gr = of.gradient(this->maxgc);

                ret["function_value"] = function_value;
                ret["converged"] = true;
                ret["iterations"] = iteration;
                ret["maxgc"] = this->maxgc;
                ret["parameters"] = of.parameter_values();
                ret["gradient"] = Gr;
                if (this->estimate_hessian) {
                    ret["hessian"] = this->get_estimated_hessian();
                }
                ret["message"] = "L-BFGS: Reached convergence.";

            }

            z = wg;

            if (i > 0 && this->max_history > 0) {

                size_t h = std::min<size_t > (i, this->max_history);
                size_t end = (i - 1) % h;

                //update histories
                for (size_t r = 0; r < nops; r++) {
                    dxs[r][end] = this->of.parameters[r].internal_value() - px[r];
                    dgs[r][end] = wg[r] - pg[r];
                }



                for (size_t j = 0; j < h; ++j) {
                    const size_t k = (end - j + h) % h;
                    p[k] = 1.0 / this->dot(this->Column(dxs, k, this->of.parameters.size()), this->Column(dgs, k, this->of.parameters.size()));

                    a[k] = p[k] * this->dot(this->Column(dxs, k, this->of.parameters.size()), z);
                    z -= a[k] * this->Column(dgs, k, this->of.parameters.size());
                }
                // Scaling of initial Hessian (identity matrix)
                z *= this->dot(this->Column(dxs, end, this->of.parameters.size()), this->Column(dgs, end, this->of.parameters.size())) / this->dot(this->Column(dgs, end, this->of.parameters.size()), Column(dgs, end, this->of.parameters.size()));

                for (size_t j = 0; j < h; ++j) {
                    const size_t k = (end + j + 1) % h;
                    const real_t b = p[k] * dot(this->Column(dgs, k, this->of.parameters.size()), z);
                    z += this->Column(dxs, k, this->of.parameters.size()) * (a[k] - b);
                }

            }//end if(i>0)

            for (size_t j = 0; j < nops; j++) {
                px[j] = this->of.parameters[j].internal_value();
                this->x[j] = px[j];
                pg[j] = wg[j];


            }//end for



            real_t fv = function_value;
            if (!this->line_search(*this->of.fn,
                    *this->of.gr,
                    fx,
                    function_value,
                    this->x,
                    this->best,
                    z,
                    this->gradient,
                    wg,
                    this->maxgc,
                    iteration,
                    this->max_iterations,
                    max_line_searches)) {

                of.evaluate(function_value);
                Rcpp::NumericVector Gr = of.gradient(this->maxgc);
                ret["function_value"] = function_value;

                if (this->maxgc <= this->tolerance) {
                    ret["converged"] = true;
                    ret["iterations"] = iteration;
                    ret["message"] = "L-BFGS: Reached convergence.";
                } else {
                    ret["converged"] = false;
                    ret["iterations"] = iteration;
                    ret["message"] = "L-BFGS: Max line searches.";
                }


                ret["maxgc"] = this->maxgc;
                ret["parameters"] = of.parameter_values();
                ret["gradient"] = Gr;
                return ret;

            }

            if ((fv - function_value) == 0.0 && no_progress_count == 15) {
                Rcpp::NumericVector Gr = of.gradient(this->maxgc);
                ret["function_value"] = function_value;
                ret["converged"] = false;
                ret["iterations"] = this->max_iterations;
                ret["maxgc"] = this->maxgc;
                ret["parameters"] = of.parameter_values();
                ret["gradient"] = Gr;
                ret["message"] = "L-BFGS: No progress";

            } else if ((fv - function_value) == 0.0) {
                no_progress_count++;
            }

        }

        return ret;
    }

};

std::shared_ptr<lbfgs> lbfgs::instance = std::make_shared<lbfgs>();

class bfgs : public optimization_routine {
public:
    static std::shared_ptr<bfgs> instance;

    virtual ~bfgs() {
    }
    //
    //    // Line search to find the step size
    //
    //    double lineSearch(const std::function<double(const std::vector<double>&)>& objective,
    //            const std::vector<double>& x,
    //            const std::vector<double>& direction) {
    //        double alpha = 1.0;
    //        double c = 0.1; // Armijo condition parameter
    //        double rho = 0.5; // Backtracking parameter
    //        double f0 = objective(x);
    //        double gradDotDir = 0.0;
    //        for (size_t i = 0; i < x.size(); ++i)
    //            gradDotDir += direction[i] * gradient(x)[i];
    //        while (objective({x[0] + alpha * direction[0], x[1] + alpha * direction[1]}) > f0 + c * alpha * gradDotDir)
    //            alpha *= rho;
    //        return alpha;
    //    }

    virtual Rcpp::List run() {
        Rcpp::List ret;


        int n = this->of.parameters.size();
        std::valarray<real_t> z(n);
        std::valarray<real_t> wg(n);
        this->x.resize(n);
        this->best.resize(n);
        this->gradient.resize(n);
        std::valarray<real_t> s(n);
        std::valarray<real_t> y(n);
        std::valarray<real_t> rho(n);
        std::valarray<real_t> alpha(n);

        //initial evaluation

        for (int i = 0; i < n; i++) {
            if (this->of.parameters[i].bound) {
                this->x[i] = this->of.parameters[i].internal_value();
            } else {
                this->x[i] = this->of.parameters[i].value;
            }
            this->gradient[i] = 0;
        }

        best = x;


        real_t fx(0.0);
        this->of.evaluate(fx);
        real_t function_value = fx;

        std::vector<std::valarray < real_t >> H(n); //, std::valarray<real_t>(n, 0.0));
        for (int i = 0; i < n; i++) {
            H[i] = std::valarray<real_t>(n);
        }
        std::cout << H[0].size() << " " << std::endl;



        for (size_t i = 0; i < n; ++i) {
            H[i][i] = 1.0; // Initialize H as identity matrix
        }



        int no_progress_count = 0;

        for (int iter = 0; iter < this->max_iterations; ++iter) {
            int i = iter;
            this->of.gradient(fx, this->gradient, this->maxgc);

            // Compute search direction: z = -H * gradient
            for (size_t i = 0; i < n; ++i) {
                z[i] = -std::inner_product(std::begin(H[i]), std::end(H[i]), std::begin(gradient), 0.0);
            }


            for (int j = 0; j < n; j++) {
                wg[j] = this->of.parameters[j].scaled_gradient(
                        this->of.parameters[j].internal_value()) * this->gradient[j];
            }

            if (((iter % this->iprint) == 0) && iter > 0) {
                std::cout << "Iteration " << iter << "\n";
            }


            bool success = line_search(*this->of.fn,
                    *this->of.gr, fx, function_value, this->x, this->best, z,
                    this->gradient, wg, this->maxgc, iter, this->max_iterations, this->max_line_searches);

            if (!success) {
                std::cerr << "BFGS: Line search failed to find suitable step size at iteration " << iter + 1 << std::endl;
                return ret;
            }

            // Update x
            x = best;

            this->of.update(x);
            this->of.evaluate(fx);
            this->of.gradient(this->gradient, this->maxgc);


            // Check for convergence
            if (this->maxgc <= this->tolerance) {
                Rcpp::NumericVector ret_g = this->of.gradient(this->maxgc);
                ret["function_value"] = fx;
                ret["converged"] = true;
                ret["iterations"] = iter;
                ret["maxgc"] = this->maxgc;
                ret["parameters"] = this->of.parameter_values();
                ret["gradient"] = ret_g;
                ret["message"] = "BFGS: Reached convergence";
                return ret;
            }

            // Update x: x_{k+1} = x_k + alpha_k * z_k
            // Update s and y
            for (size_t i = 0; i < n; ++i) {
                s[i] = best[i] - x[i];
                y[i] = gradient[i] - wg[i];
            }

            // Compute rho: rho_k = 1 / (s_k^T * y_k)
            real_t rho_denominator = std::inner_product(std::begin(s), std::end(s), std::begin(y), 0.0);
            if (std::abs(rho_denominator) < tolerance) {
                //std::cerr << "BFGS: Rho denominator is too small, skipping Hessian update." << std::endl;
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                rho[i] = 1.0 / rho_denominator * s[i] * y[i];
            }

            // Compute alpha: alpha_k = rho_k * y_k^T * z_k
            real_t alpha_value = std::inner_product(std::begin(y), std::end(y), std::begin(z), 0.0);
            if (std::abs(alpha_value) < tolerance) {
                //std::cerr << "BFGS: Alpha value is too small, skipping Hessian update." << std::endl;
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                alpha[i] = rho[i] * alpha_value;
            }

            // Compute Hessian update: H_{k+1} = H_k + rho_k * s_k * s_k^T - alpha_k * y_k * z_k^T
            // Compute Hessian update: H_{k+1} = H_k + rho_k * s_k * s_k^T - alpha_k * y_k * z_k^T
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    H[i][j] += rho[i] * s[i] * s[j] - alpha[i] * y[i] * z[j];
                }
            }

            if ((fx - function_value) == 0.0 && no_progress_count == 15) {
                Rcpp::NumericVector Gr = of.gradient(this->maxgc);
                ret["function_value"] = function_value;
                ret["converged"] = false;
                ret["iterations"] = this->max_iterations;
                ret["maxgc"] = this->maxgc;
                ret["parameters"] = of.parameter_values();
                ret["gradient"] = Gr;
                ret["message"] = "BFGS: No progress";

            } else if ((fx - function_value) == 0.0) {
                no_progress_count++;
            }


        }

        this->of.update(x);
        this->of.evaluate(fx);
        //        this->of.gradient(this->gradient, this->maxgc);
        Rcpp::NumericVector ret_g = this->of.gradient(this->maxgc);
        ret["function_value"] = fx;
        if (this->maxgc <= this->tolerance) {
            ret["converged"] = true;
            ret["iterations"] = this->max_iterations;
        } else {
            ret["converged"] = false;
            ret["iterations"] = this->max_iterations;
        }
        ret["maxgc"] = this->maxgc;
        ret["parameters"] = this->of.parameter_values();
        ret["gradient"] = ret_g;
        ret["message"] = "BFGS: Maximum number of iterations reached without convergence";



        return ret;
    }

};

std::shared_ptr<bfgs> bfgs::instance = std::make_shared<bfgs>();

class frank_wolfe : public optimization_routine {
public:
    static std::shared_ptr<frank_wolfe> instance;

    frank_wolfe() {
    }

    virtual ~frank_wolfe() {
    }

    std::valarray<double> linearApproximation(int i, const std::valarray<double>& gradient, const std::valarray<double>& direction, double& stepSize) {
        // Compute step size using line search or other methods
        // Here, we use a fixed step size of 1/norm(direction)
        stepSize = i ? 1.0 : 1.0 / std::sqrt(std::inner_product(std::begin(direction), std::end(direction), std::begin(direction), 0.0));
        // Compute the linear approximation
        std::valarray<double> approximation(gradient.size());
        for (size_t i = 0; i < gradient.size(); ++i) {
            approximation[i] = -stepSize * gradient[i];
        }
        return approximation;
    }

    // Perform a backtracking line search to find a step size that satisfies the Armijo condition

    double lineSearch(const std::valarray<double>& x, const std::valarray<double>& direction, const std::valarray<double>& gradient) {
        const double alpha = 0.5; // Backtracking factor
        const double beta = 0.5; // Step size reduction factor
        double stepSize = 1.0; // Initial step size

        double fx;
        this->of.update(x);
        this->of.evaluate(fx);

        double slope = std::inner_product(std::begin(direction), std::end(direction), std::begin(gradient), 0.0);

        while (true) {
            std::valarray<double> newX(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                newX[i] = x[i] + stepSize * direction[i];
            }

            double newFx;
            this->of.update(newX);
            this->of.evaluate(newFx); //= objectiveFunction(newX);
            if (newFx <= fx + alpha * stepSize * slope) {
                return stepSize;
            }
            stepSize *= beta;
        }
    }

    virtual Rcpp::List run() {
        Rcpp::List ret;

        int n = this->of.parameters.size();
        std::valarray<real_t> z(n);
        std::valarray<real_t> wg(n);
        this->x.resize(n);
        this->best.resize(n);
        this->gradient.resize(n);

        //initial evaluation

        for (int i = 0; i < n; i++) {
            if (this->of.parameters[i].bound) {
                this->x[i] = this->of.parameters[i].internal_value();
            } else {
                this->x[i] = this->of.parameters[i].value;
            }
            this->gradient[i] = 0;
        }

        best = x;

        double fx;
        this->of.update(x);
        this->of.evaluate(fx);

        for (int iter = 0; iter < this->max_iterations; ++iter) {
            std::valarray<double> grad(n);
            this->of.update(x);
            this->of.evaluate(fx);
            this->of.gradient(grad, this->maxgc);


            if (/*std::sqrt(std::inner_product(std::begin(direction), std::end(direction), std::begin(direction), 0.0)) */this->maxgc < tolerance) {
                std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
                Rcpp::NumericVector ret_g = this->of.gradient(this->maxgc);
                ret["function_value"] = fx;
                ret["parameters"] = this->of.parameter_values();
                ret["maxgc"] = this->maxgc;
                return ret;
            }

            // Step 1: Solve the linear subproblem to find the direction
            int n = x.size();
            std::valarray<double> direction(n);
            double maxDotProduct = -1; // Initializing with a negative value for comparison
            for (int i = 0; i < n; ++i) {
                direction[i] = -grad[i];
                maxDotProduct = std::max(maxDotProduct, grad[i] * grad[i]);
            }

            // Step 2: Determine the step size
            double stepSize = 2.0 / (iter + 2);

            // Step 3: Update x using the direction and step size
            for (int i = 0; i < n; ++i) {
                x[i] += stepSize * direction[i];
            }
        }


        std::cout << "Maximum iterations reached without convergence." << std::endl;
        Rcpp::NumericVector ret_g = this->of.gradient(this->maxgc);
        this->of.update(x);
        this->of.evaluate(fx);
        ret["function_value"] = fx;
        ret["parameters"] = this->of.parameter_values();
        ret["maxgc"] = this->maxgc;
        return ret;
    }

};

std::shared_ptr<frank_wolfe> frank_wolfe::instance = std::make_shared<frank_wolfe>();



class port_hunteler: public optimization_routine{
public:
  static std::shared_ptr<port_hunteler> instance;

  virtual Rcpp::List run(){
    Rcpp::List ret;

    port::integer n = this->of.parameters.size();
    std::vector<real_t> g(n, 0.0);
    std::vector<real_t> d(n, 0.0);
    std::vector<real_t> x_(n, 0.0);
    std::vector<real_t> b(n * 2, 0.0);
    port::integer lv = 71 + n * (n + 13) / 2;
    std::vector<real_t> v(lv, 0.0);
    port::integer liv = 60 + n;
    std::vector<port::integer>iv(liv, 0);
    v[0] = 2;
    std::valarray<real_t> z(n);
    std::valarray<real_t> wg(n);
    std::vector<real_t> wg_(n);
    this->best.resize(n);
    this->x.resize(n);
    this->gradient.resize(n);

    for (int i = 0; i < n; i++) {
      d[i] = 0.001;
      if (this->of.parameters[i].bound) {
        this->x[i] = this->of.parameters[i].internal_value();
      } else {
        this->x[i] = this->of.parameters[i].value;
      }
      this->gradient[i] = 0;
    }

    best = x;

    double fx;
    this->of.update(x);
    this->of.evaluate(fx);
    real_t function_value = fx;
    this->of.gradient(this->gradient, this->maxgc);

    real_t previous_function_value;

    size_t iter = 0;

    do {


      for (int i = 0; i < n; i++) {
        x_[i] =  this->of.parameters[i].internal_value();
        x[i] = x_[i];
        g[i] = this->gradient[i];
        wg[i] = this->of.parameters[i].scaled_gradient(
          this->of.parameters[i].internal_value()) * this->gradient[i];
        wg_[i] = wg[i];
      }



      port::drmng_<real_t>(/*b.data(),*/ d.data(), &fx, wg_.data(), iv.data(), &liv, &lv, &n, v.data(), x_.data());



      if ((iv[0]) == 2) {


        for (int i = 0; i < n; i++) {
          this->of.parameters[i].update_value(x_[i]);
          this->x[i] = x_[i];
        }


        previous_function_value = fx;
        real_t f = 0.0;

        this->of.update(x);
        this->of.evaluate(fx);
        function_value = fx;
        this->of.gradient(this->gradient, this->maxgc);

        for (int i = 0; i < n; i++) {
          g[i] = this->gradient[i];
          wg[i] = this->of.parameters[i].scaled_gradient(
            this->of.parameters[i].internal_value()) * this->gradient[i];
          wg_[i] = wg[i];
        }
        z = wg;

        if (this->maxgc <= this->tolerance) {
          break;
        }


      } else {

        for (int i = 0; i < n; i++) {
          this->of.parameters[i].update_value(x_[i]);
          this->x[i] = x_[i];
        }



        previous_function_value = fx;
        // f = 0.0;
        this->of.update(x);
        this->of.evaluate(fx);
        function_value = fx;

        Rcpp::Rcout<<fx<<std::endl;

        if (fx != fx) {
          std::cout << "Objective Function signaling NaN";
        }
      }


      if (this->maxgc <= this->tolerance) {
        break;
      }


      iter++;

    } while (iter < this->max_iterations || (iv[0]) < 3);



    this->of.update(x);
    Rcpp::NumericVector ret_g = this->of.gradient(this->maxgc);
    this->of.evaluate(fx);
    ret["function_value"] = fx;
    ret["iterations"] = iter;
    ret["parameters"] = this->of.parameter_values();
    ret["maxgc"] = this->maxgc;


    return ret;
  }

};

std::shared_ptr<port_hunteler> port_hunteler::instance = std::make_shared<port_hunteler>();

//
// class cg: public optimization_routine{
// public:
//   static std::shared_ptr<cg> instance;
//
//   virtual Rcpp::List run(){
//     Rcpp::List ret;
//
//
//     return ret;
//   }
//
// };
//
// std::shared_ptr<cg> cg::instance = std::make_shared<cg>();
//
// class newton: public optimization_routine{
// public:
//   static std::shared_ptr<newton> instance;
//
//   virtual Rcpp::List run(){
//     Rcpp::List ret;
//
//
//     return ret;
//   }
//
// };


#endif
