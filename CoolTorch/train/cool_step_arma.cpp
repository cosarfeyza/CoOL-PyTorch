// Backend: C++ (Armadillo), compiled into the cool_ext_arma extension used by trainer.py.
// R equivalent: the Rcpp routine cpp_train_network_relu, called from
// CoOL_2_train_neural_network (CoOL_functions.R) — same algorithm, same non-negative
// forward pass and weight-update rules, re-implemented against Armadillo/libtorch
// instead of RcppArmadillo so it can be built as a PyTorch extension.
#include <torch/extension.h>
#include <armadillo>
#include <tuple>
#include <iostream>
#include <cstring>

// ----------------------------------------------------------
// relu helpers (same as rcpprelu / rcpprelu_neg)
// ----------------------------------------------------------

arma::mat rcpprelu(const arma::mat & x) {
    arma::mat m = x % (x > 0);
    return m;
}

arma::mat rcpprelu_neg(const arma::mat & x) {
    arma::mat m = x % (x < 0);
    return m;
}

// ----------------------------------------------------------
// Result struct instead of Rcpp::List
// ----------------------------------------------------------
struct CoolResult {
    arma::mat W1;
    arma::mat B1;
    arma::mat W2;
    arma::mat B2;
    arma::mat C2;
    arma::vec train_performance;
    arma::vec test_performance;
    arma::vec weight_performance;
    arma::vec baseline_risk_monitor;
    int       epochs;
};

// ----------------------------------------------------------
// Original cpp_train_network_relu rewritten to return CoolResult
//
// ----------------------------------------------------------
CoolResult cpp_train_network_relu_arma(
  const arma::mat & x,
  const arma::vec & y,
  const arma::vec & c,
  const arma::mat & testx,
  const arma::vec & testy,
  const arma::vec & testc,
  const arma::mat & W1_input,
  const arma::mat & B1_input,
  const arma::mat & W2_input,
  const arma::mat & B2_input,
  const arma::mat & C2_input,
  const arma::vec & ipw,
  double lr = 0.01,
  int    maxepochs = 100,
  double input_parameter_reg = 0.000001,
  int drop_out = 0,
  double fix_baseline_risk = -1
) {
  
  arma::arma_rng::set_seed(123);
  int nsamples   = y.size();
  int nfeatures  = x.n_cols;
  int hidden     = W1_input.n_cols;
  int sparse_data = 0;
  double mean_y  = arma::accu(y) / nsamples;

  std::cout << "CoOL (Armardillo backend)" << std::endl;

  // Loaded initialized weights.
  arma::mat W1(nfeatures, hidden, arma::fill::zeros);
  W1 = W1_input;

  arma::mat B1(1, hidden, arma::fill::zeros);
  B1 = B1_input;

  arma::mat W2(hidden, 1, arma::fill::zeros);
  W2 = W2_input;

  arma::mat B2(1, 1, arma::fill::zeros);
  B2 = B2_input;
  if (fix_baseline_risk >= 0) {
    B2 = fix_baseline_risk;
  }

  arma::mat C2(1, 1, arma::fill::zeros);
  C2 = C2_input;

  // W1 for the test data parameter qualification
  arma::mat W1_previous_step(nfeatures, hidden, arma::fill::zeros);
  W1_previous_step = W1_input;

  // Define temporary holders and predicted output
  arma::mat h(1, hidden);
  arma::vec o(nsamples);

  arma::vec trainperf(maxepochs, arma::fill::zeros);
  arma::vec testperf(maxepochs, arma::fill::zeros);

  trainperf.replace(0, arma::datum::nan);
  testperf.replace(0, arma::datum::nan);

  // monitor the difference in weights
  arma::vec trainweights(maxepochs, arma::fill::zeros);
  trainweights.replace(0, arma::datum::nan);

  // monitor the baseline risk
  arma::vec baseline_risks(maxepochs, arma::fill::zeros);
  baseline_risks.replace(0, arma::datum::nan);

  arma::vec index = arma::linspace<arma::vec>(0, nsamples - 1, nsamples);

  int row;
  arma::uword epoch = 0;

  // =========================
  // Main training loop
  // =========================
  for (epoch = 0; epoch < (arma::uword)maxepochs; epoch++) {
    // First we shuffle/permute all row indices before commencing
    arma::vec shuffle = arma::shuffle(index);

    // Step 1: Forward pass to get h and o.
    for (arma::uword rowidx = 0; rowidx < x.n_rows; rowidx++) {
      // This is the row we're working on right now
      row = static_cast<int>(shuffle(rowidx));
      // row = rowidx; // (commented in original)

      // h contains the output from the hidden layer.
      h = rcpprelu((x.row(row) * (W1)) + B1);

      // output layer
      o(row) = rcpprelu(h * W2 + B2 + c(row) * C2)(0,0);

      // Step 2: Backwards pass to update the parameters W1, B1, B2
      double E_outO = - (y(row) - o(row));
      arma::mat netO_wHO  = arma::trans(h);
      arma::mat netO_outH = arma::trans(W2);

      // All calculations done. Now do the updating
      if (drop_out == 0) {
        for (size_t g = 0; g < (size_t)W1.n_rows; g++) {
          W1.row(g) = rcpprelu(
            W1.row(g)
            - ipw(row) * lr * E_outO * (netO_outH % (h > 0)) * x(row, g)
            - ipw(row) * lr * input_parameter_reg
          ); // L1 regularized - penalized
        }
      }

      if (drop_out == 1) {
        for (size_t g = 0; g < (size_t)W1.n_rows; g++) {
          W1.row(g) = rcpprelu(
            W1.row(g)
            - ipw(row) * lr * E_outO * ((W1.row(g) > 0) % netO_outH % (h > 0)) * x(row, g)
            - ipw(row) * lr * input_parameter_reg
          ); // L1 regularized - penalized
        }
      }

      B1 = rcpprelu_neg(B1 - ipw(row) * lr * E_outO * (netO_outH % (h > 0)));

      if (fix_baseline_risk < 0) {
        B2 = rcpprelu(B2 - ipw(row) * lr / 10.0 * E_outO);

        // Update confounder weight
        netO_wHO = c(row);
        C2 = rcpprelu(C2(0,0) - ipw(row) * lr / 10.0 * E_outO * netO_wHO);
      }
    } // Row loop

    // Compute performance (train)
    arma::mat tmp = x * W1;
    double mean_perform = 0.5 *
      arma::accu(arma::square(
        y - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2
                      + B2(0,0) + C2(0,0) * c))
      )) / nsamples;

    // Compute performance on the validation (test) set
    tmp = testx * W1;
    double mean_val_perform = 0.5 *
      arma::accu(arma::square(
        testy - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2
                          + B2(0,0) + C2(0,0) * testc))
      )) / nsamples;

    trainperf(epoch) = mean_perform;
    testperf(epoch)  = mean_val_perform;

    // Calculating the mean squared difference in weight update
    double mean_w_diff = 0.0;
    for (size_t i = 0; i < (size_t)W1.n_rows; i++) {
      for (size_t g = 0; g < (size_t)W1.n_cols; g++) {
        double diff = (W1(i,g) - W1_previous_step(i,g));
        mean_w_diff += diff * diff;
      }
    }
    mean_w_diff = mean_w_diff / (W1.n_rows * W1.n_cols);
    trainweights(epoch) = mean_w_diff;
    W1_previous_step = W1;

    // Monitor the baseline risk
    baseline_risks(epoch) = B2(0,0);

    if (B2(0,0) == 0) {
      sparse_data = 1;
    }

    if (epoch % 10 == 0) {
      std::cout << epoch
                << " epochs: Train performance of "
                << mean_perform
                << ". Baseline risk estimated to "
                << B2(0,0)
                << "." << std::endl;

      if (B2(0,0) > mean_y) {
        std::cout
          << "Warning: The baseline risk (" << B2(0,0)
          << ") is higher than mean(Y) (" << mean_y
          << ")! Consider reducing the regularisation of the input parameters."
          << std::endl;
      }
      if (sparse_data == 1) {
        std::cout
          << "Warning: The baseline risk (" << B2(0,0)
          << ") has at one time been estimated to zero. Data may be too sparse."
          << std::endl;
      }
    }
  }  // End of all epochs

  arma::colvec trainp               = trainperf.elem(arma::find_finite(trainperf));
  arma::colvec testp                = testperf.elem(arma::find_finite(testperf));
  arma::colvec trainp_weights       = trainweights.elem(arma::find_finite(trainweights));
  arma::colvec baseline_risks_monitor = baseline_risks.elem(arma::find_finite(baseline_risks));

  CoolResult res;
  res.W1  = W1;
  res.B1  = B1;
  res.W2  = W2;
  res.B2  = B2;
  res.C2  = C2;
  res.train_performance     = trainp;
  res.test_performance      = testp;
  res.weight_performance    = trainp_weights;
  res.baseline_risk_monitor = baseline_risks_monitor;
  res.epochs = static_cast<int>(epoch + 1);

  return res;
}

// ----------------------------------------------------------
// Tensor <-> arma conversion helpers
// ----------------------------------------------------------

inline torch::Tensor to_cpu_double_contig(const torch::Tensor& t) {
    return t.to(torch::kCPU, torch::kFloat64).contiguous();
}



inline arma::mat tensor_to_arma_mat(const torch::Tensor& t) {
    auto tt = t.to(torch::kCPU, torch::kFloat64).contiguous();
    TORCH_CHECK(tt.dim() == 2, "Expected 2D tensor for arma::mat");

    const auto n_rows = tt.size(0);
    const auto n_cols = tt.size(1);

    arma::mat M(n_rows, n_cols);

    const double* data = tt.data_ptr<double>();
    // PyTorch: row-major → index = i * n_cols + j
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            M(i, j) = data[i * n_cols + j];
        }
    }
    return M;
}


inline arma::vec tensor_to_arma_vec(const torch::Tensor& t) {
    auto tt = to_cpu_double_contig(t.view({-1}));
    const auto n = tt.size(0);

    // Create Armadillo vector that copies data from PyTorch tensor
    return arma::vec(tt.data_ptr<double>(),
                     static_cast<arma::uword>(n),
                     /*copy_aux_mem=*/true);
}

inline torch::Tensor arma_mat_to_tensor(const arma::mat& M) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    torch::Tensor t = torch::empty({(long)M.n_rows, (long)M.n_cols}, options);

    double* data = t.data_ptr<double>();
    const int64_t n_rows = M.n_rows;
    const int64_t n_cols = M.n_cols;

    
    for (int64_t i = 0; i < n_rows; ++i) {
        for (int64_t j = 0; j < n_cols; ++j) {
            data[i * n_cols + j] = M(i, j);
        }
    }
    return t;
}


inline torch::Tensor arma_vec_to_tensor(const arma::vec& v) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    torch::Tensor t = torch::empty({(long)v.n_elem}, options);
    std::memcpy(t.data_ptr<double>(), v.memptr(),
                v.n_elem * sizeof(double));
    return t;
}


std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,  // W1,B1,W2,B2,C2
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor                  // train,test,weights,baseline
>
cool_train_block(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor c,
    torch::Tensor testx,
    torch::Tensor testy,
    torch::Tensor testc,
    torch::Tensor W1_in,
    torch::Tensor B1_in,
    torch::Tensor W2_in,
    torch::Tensor B2_in,
    torch::Tensor C2_in,
    torch::Tensor ipw,
    double lr,
    int64_t maxepochs,
    double input_parameter_reg,
    int64_t drop_out,
    double fix_baseline_risk
) {
    arma::arma_rng::set_seed(123);
    // Tensor -> arma
    arma::mat X      = tensor_to_arma_mat(x);
    arma::vec Y      = tensor_to_arma_vec(y);
    arma::vec C      = tensor_to_arma_vec(c);

    arma::mat Xtest  = tensor_to_arma_mat(testx);
    arma::vec Ytest  = tensor_to_arma_vec(testy);
    arma::vec Ctest  = tensor_to_arma_vec(testc);

    arma::mat W1     = tensor_to_arma_mat(W1_in);   // nfeatures x hidden
    arma::mat B1     = tensor_to_arma_mat(B1_in);   // 1 x hidden
    arma::mat W2     = tensor_to_arma_mat(W2_in);   // hidden x 1
    arma::mat B2     = tensor_to_arma_mat(B2_in);   // 1 x 1
    arma::mat C2     = tensor_to_arma_mat(C2_in);   // 1 x 1
    arma::vec IPW    = tensor_to_arma_vec(ipw);

    // Call original Armadillo implementation
    CoolResult res = cpp_train_network_relu_arma(
        X, Y, C,
        Xtest, Ytest, Ctest,
        W1, B1, W2, B2, C2,
        IPW,
        lr,
        static_cast<int>(maxepochs),
        input_parameter_reg,
        static_cast<int>(drop_out),
        fix_baseline_risk
    );

    // arma -> Tensor
    torch::Tensor W1_out = arma_mat_to_tensor(res.W1);
    torch::Tensor B1_out = arma_mat_to_tensor(res.B1);
    torch::Tensor W2_out = arma_mat_to_tensor(res.W2);
    torch::Tensor B2_out = arma_mat_to_tensor(res.B2);
    torch::Tensor C2_out = arma_mat_to_tensor(res.C2);

    torch::Tensor trainperf_t  = arma_vec_to_tensor(res.train_performance);
    torch::Tensor testperf_t   = arma_vec_to_tensor(res.test_performance);
    torch::Tensor weightperf_t = arma_vec_to_tensor(res.weight_performance);
    torch::Tensor baselines_t  = arma_vec_to_tensor(res.baseline_risk_monitor);

    return std::make_tuple(
        W1_out, B1_out, W2_out, B2_out, C2_out,
        trainperf_t, testperf_t, weightperf_t, baselines_t
    );
}

// PyBind module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cool_train_block",
        &cool_train_block,
        "CoOL training block (exact Armadillo cpp_train_network_relu backend)"
    );
}
