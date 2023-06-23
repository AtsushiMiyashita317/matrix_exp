#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <iostream>

#include <torch/extension.h>

namespace {
template <typename scalar_t, int ROW, int COL>
using array2d = std::array<std::array<scalar_t, COL>, ROW>;

// we consider 6 Taylor expansions of degree
// 1, 2, 4, 8, 12, 18
constexpr int total_n_degs = 6;

torch::Tensor operator_1_norm(const torch::Tensor& tensor) {
  return std::get<0>(tensor.abs().sum(-2).max(-1));
}

// Allocates a buffers of uninitialized or zero values
// of shape [n_copies, a.size()]
torch::Tensor _allocate_buffer(const torch::Tensor& a, int n_copies, bool is_zero = false) {
  auto res = at::empty(
    {n_copies, a.size(0), a.size(1), a.size(2)},
    a.options().memory_format(at::MemoryFormat::Contiguous)
  );

  if (is_zero) {
    res.zero_();
  }

  return res;
}

inline torch::Tensor& pointsym_matmul_out(torch::Tensor& out, const torch::Tensor& self, const torch::Tensor& other) {
  auto n = self.size(-1);
  auto view_out1 = out.narrow(-2, 0, n);
  at::matmul_out(view_out1, self.narrow(-2, 0, n), other.narrow(-2, 0, n));
  view_out1.add_(
    at::matmul(
      self.narrow(-2, n-1, n).narrow(-1, 0, n-1).flip({-1,-2}).conj(), 
      other.narrow(-2, n, n-1)
    )
  );
  auto view_out2 = out.narrow(-2, n, n-1);
  at::matmul_out(view_out2, self.narrow(-2, n, n-1), other.narrow(-2, 0, n));
  view_out2.add_(
    at::matmul(
      self.narrow(-2, 0, n-1).narrow(-1, 0, n-1).flip({-1,-2}).conj(), 
      other.narrow(-2, n, n-1)
    )
  );
  return out;
}

inline torch::Tensor& matmul_out_sw1(torch::Tensor& out, const torch::Tensor& self, const torch::Tensor& other) {
  auto n = self.size(-1);
  auto m = other.size(-2);
  if (n == m) {
    return at::matmul_out(out, self, other);
  }
  else {
    return pointsym_matmul_out(out, self, other);
  }
}

inline torch::Tensor matmul_sw1(const torch::Tensor& self, const torch::Tensor& other) {
  auto out = at::empty_like(self, self.options().memory_format(at::MemoryFormat::Contiguous));
  return matmul_out_sw1(out, self, other);
}

inline torch::Tensor& triblock_matmul_out(torch::Tensor& out, const torch::Tensor& self, const torch::Tensor& other) {
  auto n = self.size(-1);
  auto view_out1 = out.narrow(-2, 0, n);
  auto view_out2 = out.narrow(-2, n, n);
  // matmul_out_sw1(view_out1, self.narrow(-2, 0, n), other.narrow(-2, 0, n));
  at::matmul_out(view_out1, self.narrow(-2, 0, n), other.narrow(-2, 0, n));
  // matmul_out_sw1(view_out2, self.narrow(-2, 0, n), other.narrow(-2, n, n));
  at::matmul_out(view_out2, self.narrow(-2, 0, n), other.narrow(-2, n, n));
  // view_out2.add_(matmul_sw1(self.narrow(-2, n, n), other.narrow(-2, 0, n)));
  view_out2.add_(at::matmul(self.narrow(-2, n, n), other.narrow(-2, 0, n)));
  return out;
}

inline torch::Tensor& matmul_out_sw2(torch::Tensor& out, const torch::Tensor& self, const torch::Tensor& other) {
  auto n = self.size(-1);
  auto m = other.size(-2);
  // if ((n != m) && (m%2 == 0)) {
  if (n != m) {
    return triblock_matmul_out(out, self, other);
  }
  else {
    // return matmul_out_sw1(out, self, other);
    return at::matmul_out(out, self, other);
  }
}

inline torch::Tensor matmul_sw2(const torch::Tensor& self, const torch::Tensor& other) {
  auto out = at::empty_like(self, self.options().memory_format(at::MemoryFormat::Contiguous));
  return matmul_out_sw2(out, self, other);
}

inline torch::Tensor& my_matmul_out(torch::Tensor& out, const torch::Tensor& self, const torch::Tensor& other) {
  return matmul_out_sw2(out, self, other);
}

inline torch::Tensor my_matmul(const torch::Tensor& self, const torch::Tensor& other) {
  return matmul_sw2(self, other);
}

// Makes `buffer` to store `num_matrices` number of matrices needed for
// compute the matrix exponentials of different orders, i.e.
// first `num_matrices` matrices from the list l := {I, A, A^2, A^3, A^6}
// in a contiguous block of memory such that
// buffer[0, ...] = l[0], // I
// buffer[1, ...] = l[1], // A
// ...
// buffer[num_matrices - 1, ...] = l[num_matries - 1]
void _fill_matrix_powers(torch::Tensor& buffer, const torch::Tensor& a, int num_matrices) {
  auto a_sizes_minus_last = a.sizes().vec();
  auto n = a.size(-1);
  a_sizes_minus_last.erase(a_sizes_minus_last.end()-2);
  // fill I
  if (n == a.size(-2)) {
    buffer.select(0, 0).copy_(
      at::diag_embed(
        at::ones({1}, buffer.options())
          .expand(a_sizes_minus_last)
      )
    );
  }
  else {
    buffer.select(0, 0).narrow(-2, 0, n).copy_(
      at::diag_embed(
        at::ones({1}, buffer.options())
          .expand(a_sizes_minus_last)
      )
    );
    buffer.select(0, 0).narrow(-2, n, a.size(-2)-n).zero_();
  }

  // fill a
  buffer.select(0, 1).copy_(a);

  // fill a^2
  if (2 <= num_matrices - 1) {
    // out for a^2
    auto view_out = buffer.select(0, 2);
    my_matmul_out(
      view_out,
      buffer.select(0, 1),
      buffer.select(0, 1)
    );
  }

  // fill a^3
  if (3 <= num_matrices - 1) {
    // out for a^3
    auto view_out = buffer.select(0, 3);
    my_matmul_out(
      view_out,
      buffer.select(0, 1),
      buffer.select(0, 2)
    );
  }

  // fill a^6
  if (4 <= num_matrices - 1) {
    // out for a^6
    auto view_out = buffer.select(0, 4);
    my_matmul_out(
      view_out,
      buffer.select(0, 3),
      buffer.select(0, 3)
    );
  }
}

inline torch::Tensor _move_memory_if_cuda_input(
  const torch::Tensor& mem,
  const torch::Tensor& in
) {
  return (in.device().type() == at::kCUDA)
    ? mem.to(at::device_of(in).value())
    : mem;
}

// convert a 1D blob to a 2D Tensor of size [1, blob.size()]
// such that blob.device() == in.device())
// designed to be used with _compute_linear_combination
template <typename scalar_t>
inline torch::Tensor _blob_to_Tensor(
  std::initializer_list<scalar_t> blob,
  const torch::Tensor& in
) {
  // we convert to void* expecitly because begin() returns
  // a pointer to a constant.
  // Blob is assumed to be a 1D array, that is why
  // we also insert a fake dimension so that the result could directly
  // be used in _compute_linear_combination
  auto tensor = at::from_blob((void*)blob.begin(), blob.size(),
    c10::toRealValueType(in.scalar_type())).unsqueeze(0);
  return _move_memory_if_cuda_input(tensor, in);
}

template <typename scalar_t>
inline torch::Tensor _linear_combination(
    const torch::Tensor& t,
    std::initializer_list<scalar_t> blob) {
  // _blob_to_Tensor converts blob to a 2D tensor for _compute_linear_combination.
  // If this tensor is of shape (1, *), the result of _compute_linear_combination
  // is going to be of shape (1, *t.shape) so we squeeze(0) so that
  // for any t with t.dim() >= 1: t.dim() == _compute_linear_combination(t, ...).dim().
  return at::_compute_linear_combination(
      t, _blob_to_Tensor<scalar_t>(blob, t))
    .squeeze(0);
}

// I + A
torch::Tensor compute_T1(const torch::Tensor& self, torch::Tensor& buffer) {
  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, 2);

  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);

    // 2 for {I, A}
    _fill_matrix_powers(As, A, 2);
    at::sum_out(view_out, As, 0);
  };
  return out;
}

// I + A + A^2 / 2
torch::Tensor compute_T2(const torch::Tensor& self, torch::Tensor& buffer) {
  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, 3);

  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);

    // 3 for {I, A, A^2}
    _fill_matrix_powers(As, A, 3);
    As.select(0, 2).div_(2.0);
    at::sum_out(view_out, As, 0);
  };
  return out;
}

// I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
template <typename scalar_t>
torch::Tensor compute_T4(const torch::Tensor& self, torch::Tensor& buffer) {
  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, 4);

  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);
    
    // 3 for {I, A, A^2}
    _fill_matrix_powers(As, A, 3);

    // output for A^2 * (I / 2 + A / 6 + A^2 / 24)
    auto tmp = As.select(0, 3);
    my_matmul_out(
      tmp,
      // contains A^2
      As.select(0, 2),
      // computes (I / 2 + A / 6 + A^2 / 24)
      _linear_combination<scalar_t>(
        As.narrow(0, 0, 3),
        {1 / 2.0, 1 / 6.0, 1 / 24.0}
      )
    );

    // I + A + A^2 * (I / 2 + A / 6 + A^2 / 24)
    view_out.copy_(
      _linear_combination<scalar_t>(
        As, {1.0, 1.0, 0.0, 1.0}
      )
    );
  };
  return out;
}

template <typename scalar_t>
torch::Tensor compute_T8(const torch::Tensor& self, torch::Tensor& buffer) {
  constexpr scalar_t sqrt_177 = 0.1330413469565007072504e+2;
  constexpr scalar_t x3 = 2. / 3.;
  constexpr scalar_t x1 = x3 * ((1. + sqrt_177) / 88.);
  constexpr scalar_t x2 = x3 * ((1. + sqrt_177) / 352.);
  constexpr scalar_t x4 = (-271. + 29. * sqrt_177) / (315. * x3);
  constexpr scalar_t x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
  constexpr scalar_t x6 = (-99. + 11. * sqrt_177) / (5040. * x3);
  constexpr scalar_t x7 = (89. - sqrt_177) / (5040. * x3);
  constexpr scalar_t y2 = (857. - 58. * sqrt_177) / 630.;

  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, 5);
  
  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);
    
    // 3 for {I, A, A^2}
    _fill_matrix_powers(As, A, 3);

    // output for A4
    auto tmp = As.select(0, 3);
    // A4 =  A2 * (x1 * A + x2 * A2)
    my_matmul_out(
      tmp,
      // As.select(0, 2) = A^2
      As.select(0, 2),
      _linear_combination<scalar_t>(
        // extract {A, A^2} from As
        As.narrow(0, 1, 2),
        {x1, x2}
      )
    );

    // output for A8
    tmp = As.select(0, 4);
    // A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4)
    my_matmul_out(
      tmp,
      // x3 * A2 + A4
      _linear_combination<scalar_t>(
        As.narrow(0, 2, 2),
        {x3, 1.0}
      ),
      _linear_combination<scalar_t>(
        As.narrow(0, 0, 4),
        {x4, x5, x6, x7}
      )
    );

    // return I + A + y2 * A2 + A8;
    view_out.copy_(
      _linear_combination<scalar_t>(
        As, {1.0, 1.0, y2, 0.0, 1.0}
      )
    );
  }
  return out;
}

template <typename scalar_t>
torch::Tensor compute_T12(const torch::Tensor& self, torch::Tensor& buffer) {
  constexpr int num_prods = 4;
  array2d<scalar_t, num_prods, num_prods> b = {{
    {
      9.0198e-16,
      0.46932117595418237389,
      -0.20099424927047284052,
      -0.04623946134063071740
    },
    {
      5.31597895759871264183,
      1.19926790417132231573,
      0.01179296240992997031,
      0.01108844528519167989
    },
    {
      0.18188869982170434744,
      0.05502798439925399070,
      0.09351590770535414968,
      0.00610700528898058230
    },
    {
      -2.0861320e-13,
      -0.13181061013830184015,
      -0.02027855540589259079,
      -0.00675951846863086359
    }
  }};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
    reinterpret_cast<void*>(&b),
    {num_prods, num_prods},
    {num_prods, 1},
    c10::toRealValueType(self.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, self);

  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, num_prods);
  
  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);
    
    _fill_matrix_powers(As, A, num_prods);

    auto Bs = at::native::_compute_linear_combination(As, bs);

    // output for A6
    auto tmp = As.select(0, 0);
    // compute A6
    Bs.select(0, 2).add_(my_matmul_out(
      tmp,
      Bs.select(0, 3),
      Bs.select(0, 3)
    ));

    view_out.copy_(
      Bs.select(0, 0).add_(my_matmul_out(
        tmp,
        Bs.select(0, 1).add_(Bs.select(0, 2)),
        Bs.select(0, 2)
      ))
    );
  }
  return out;
}

template <typename scalar_t>
torch::Tensor compute_T18(const torch::Tensor& self, torch::Tensor& buffer) {
  constexpr int num_prods = 5;
  array2d<scalar_t, num_prods, num_prods> b = {{
    {
      0.,
      -1.00365581030144618291e-01,
      -8.02924648241156932449e-03,
      -8.92138498045729985177e-04,
      0.
    },
    {
      0.,
      3.97849749499645077844e-01,
      1.36783778460411720168e+00,
      4.98289622525382669416e-01,
      -6.37898194594723280150e-04
    },
    {
      -1.09676396052962061844e+01,
      1.68015813878906206114e+00,
      5.71779846478865511061e-02,
      -6.98210122488052056106e-03,
      3.34975017086070470649e-05
    },
    {
      -9.04316832390810593223e-02,
      -6.76404519071381882256e-02,
      6.75961301770459654925e-02,
      2.95552570429315521194e-02,
      -1.39180257516060693404e-05
    },
    {
      0.,
      0.,
      -9.23364619367118555360e-02,
      -1.69364939002081722752e-02,
      -1.40086798182036094347e-05
    }
  }};

  // gather coefficients `b` from above into a tensor,
  // and move them to device `device_of(A)`
  auto bs = at::from_blob(
    reinterpret_cast<void*>(&b),
    {num_prods, num_prods},
    {num_prods, 1},
    c10::toRealValueType(self.scalar_type())
  );
  bs = _move_memory_if_cuda_input(bs, self);

  int max_batch = buffer.size(1);
  int n_iter = (self.size(0) + max_batch -1) / max_batch;
  auto out = at::empty_like(self);
  auto view_buffer = buffer.narrow(0, 0, num_prods);
  
  for (const auto i: c10::irange(n_iter)) {
    int begin = i*max_batch;
    int size = self.size(0) - begin;
    size = std::min(max_batch, size);
    auto As = view_buffer.narrow(1, 0, size);
    auto A = self.narrow(0, begin, size);
    auto view_out = out.narrow(0, begin, size);

    _fill_matrix_powers(As, A, num_prods);

    auto Bs = at::native::_compute_linear_combination(As, bs);

    // tmp buffer for this matrix product
    auto tmp = As.select(0, 0);
    // compute A9
    Bs.select(0, 3).add_(my_matmul_out(
      tmp,
      Bs.select(0, 0),
      Bs.select(0, 4))
    );

    view_out.copy_(
      Bs.select(0, 1).add_(my_matmul_out(
        tmp,
        Bs.select(0, 2).add_(Bs.select(0, 3)),
        Bs.select(0, 3)
      ))
    );
  }
  return out;
}

template <typename scalar_t>
void compute_T18_scale_square(
  torch::Tensor& mexp_out,
  const torch::Tensor& a,
  const torch::Tensor& norm,
  scalar_t theta,
  torch::Tensor& buffer
) {
  // Scale
  const auto s = at::max(
    at::zeros_like(norm),
    at::ceil(at::log2(norm / theta))
  ).unsqueeze(-1).unsqueeze(-1).to(at::kLong);
  const auto pow2s = at::pow(2, s);
  const auto a_scaled = a / pow2s;

  // Square
  auto mexp_scaled = compute_T18<scalar_t>(a_scaled, buffer);
  auto s_cpu = (s.device().type() == at::kCPU)
    ? s : s.to(at::kCPU);
  for (const auto i : c10::irange(mexp_scaled.size(0))) {
    auto s_val = s_cpu.select(0, i).template item<int64_t>();
    auto mexp = mexp_scaled.select(0, i);
    for (const auto p C10_UNUSED : c10::irange(s_val)) {
      mexp = my_matmul(mexp, mexp);
    }
    mexp_out.select(0, i).copy_(mexp);
  }
}

template <typename scalar_t>
torch::Tensor mexp_impl(
  const torch::Tensor& a,
  std::array<scalar_t, total_n_degs> thetas,
  int max_length,
  bool compute_highest_degree_approx = false
) {
  auto res = at::empty_like(a);
  int volume = a.size(-2) * a.size(-1);
  int max_batch = (max_length * max_length + volume -1) / volume;
  int full = a.size(0);
  max_batch = std::min(max_batch, full);
  auto buffer = _allocate_buffer(a.narrow(0, 0, max_batch), 5);
  const auto norm = operator_1_norm(a);
  // `norm_cpu` is used to decide which Tensors require which approximation
  // based on their norm. This decision takes place on CPU.
  // It requires moving data back and forth between devices when `a` is on CUDA,
  // but at the cost of only one sigle CPU-CUDA synchronization (instead of 6),
  // and better performance overall (benchmarked).
  const auto norm_cpu = (a.device().type() == at::kCUDA)
    ? norm.to(at::kCPU) : norm;

  if (!compute_highest_degree_approx) {
    constexpr std::array<
      torch::Tensor(*)(const torch::Tensor&, torch::Tensor&),
      total_n_degs - 1>
    compute_Ts = {
      compute_T1, compute_T2, compute_T4<scalar_t>,
      compute_T8<scalar_t>, compute_T12<scalar_t>
    };

    for (int i = 0; i < total_n_degs - 1; ++i) {
      auto norm_lower_bound = (i == 0) ? static_cast<scalar_t>(-1) : thetas[i - 1];
      auto norm_upper_bound = thetas[i];
      // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
      auto idx_curr_norm_interval = (
        (norm_lower_bound < norm_cpu) * (norm_cpu <= norm_upper_bound)
      ).nonzero().squeeze(-1);

      if (idx_curr_norm_interval.numel()) {
        auto idx_to_device = _move_memory_if_cuda_input(
          idx_curr_norm_interval, a
        );
        auto sub_a = at::index_select(a, 0, idx_to_device);
        res.index_put_({idx_to_device}, compute_Ts[i](sub_a, buffer));
      }
    }

    // nonzero returns a 2D tensor, hence squeeze(-1) to make it 1D
    auto idx_large_norm = (norm_cpu >= thetas[total_n_degs - 2])
      .nonzero().squeeze(-1);

    if (idx_large_norm.numel()) {
      auto idx_to_device = _move_memory_if_cuda_input(
        idx_large_norm, a
      );
      auto a_large_norm = at::index_select(a, 0, idx_to_device);
      auto large_norm_subset = at::index_select(norm, 0, idx_to_device);
      auto mexp_out = at::empty_like(a_large_norm);

      compute_T18_scale_square(
        mexp_out,
        a_large_norm,
        large_norm_subset,
        thetas[total_n_degs - 1],
        buffer
      );
      res.index_put_({idx_large_norm}, mexp_out);
    }

    return res;
  }

  compute_T18_scale_square(
    res, a, norm,
    thetas[total_n_degs - 1],
    buffer
  );

  return res;
}

// matrix exponential
torch::Tensor mexp(const torch::Tensor& a, int max_length, bool compute_highest_degree_approx = false) {
  // squash batch dimensions to one dimension for simplicity
  const auto a_3d = a.view({-1, a.size(-2), a.size(-1)});
  
  if (a.scalar_type() == at::ScalarType::Float
      || a.scalar_type() == at::ScalarType::ComplexFloat) {
    constexpr std::array<float, total_n_degs> thetas_float = {
      1.192092800768788e-07, // deg 1
      5.978858893805233e-04, // deg 2
      5.116619363445086e-02, // deg 4
      5.800524627688768e-01, // deg 8
      1.461661507209034e+00, // deg 12
      3.010066362817634e+00  // deg 18
    };

    return mexp_impl<float>(a_3d, thetas_float, max_length, compute_highest_degree_approx)
      .view(a.sizes());
  }
  else { // if Double or ComplexDouble
    constexpr std::array<double, total_n_degs> thetas_double = {
      2.220446049250313e-16, // deg 1
      2.580956802971767e-08, // deg 2
      3.397168839976962e-04, // deg 4
      4.991228871115323e-02, // deg 8
      2.996158913811580e-01, // deg 12
      1.090863719290036e+00  // deg 18
    };

    return mexp_impl<double>(a_3d, thetas_double, max_length, compute_highest_degree_approx)
      .view(a.sizes());
  }
}

// TODO This should be deprecated in favor of linalg_matrix_exp_differential
//      in FunctionsManual.cpp
torch::Tensor backward_mexp(
    const torch::Tensor& self, const torch::Tensor& grad, int max_length
  ) {
  // int n = self.size(-1);
  // torch::Tensor self_transposed = at::empty_like(self);
  // if (n == self.size(-2)) {
  //   self_transposed.copy_(self.mH());
  // }
  // else {
  //   self_transposed.narrow(-2, 0, n).copy_(self.narrow(-2, 0, n).mH());
  //   self_transposed.narrow(-2, n, n-1).copy_(self.narrow(-2, n-1, n).narrow(-1, 0, n-1).flip({-1,-2}).mT());
  // }
  auto self_transposed = self.mH();
  auto self_transposed_sizes = self_transposed.sizes().vec();
  self_transposed_sizes[self.dim() - 2] <<= 1;
  // self_transposed_sizes[self.dim() - 1] <<= 1;

  int n = self.size(-2);
  auto meta_grad = at::zeros(self_transposed_sizes, grad.options());
  meta_grad.narrow(-2, 0, n).copy_(self_transposed);
  // meta_grad.narrow(-2, n, n).narrow(-1, n, n).copy_(self_transposed);
  meta_grad.narrow(-2, n, n).copy_(grad);

  auto grad_input = mexp(meta_grad, max_length).narrow(-2, n, n);
  return grad_input;
}
} // end anon namespace

// Computes the matrix exponential for a given batch of squared matrices.
// The implementaion is based on:
//
// Bader, P.; Blanes, S.; Casas, F.
// Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
// Mathematics 2019, 7, 1174.
//
torch::Tensor matrix_exp(const torch::Tensor& a, int max_length = 1024) {
  // squareCheckInputs(a, "linalg.matrix_exp");
  // checkFloatingOrComplex(a, "matrix_exp");

  // NoTF32Guard disable_tf32;

  // Trivial cases
  const auto n = a.size(-1);
  if (n == 0) {
    return a.clone();
  } else if (n == 1) {
    return a.exp();
  } else {
    return mexp(a, max_length);
  }
}

// TODO This should be deprecated in favor of linalg_matrix_exp_differential
//      in FunctionsManual.cpp
torch::Tensor matrix_exp_backward(
    const torch::Tensor& self, const torch::Tensor& grad, int max_length
  ) {
  // NoTF32Guard disable_tf32;
  return backward_mexp(self, grad, max_length);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matrix_exp, "matrix_exp forward");
  m.def("backward", &matrix_exp_backward, "matrix_exp backward");
}
