#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiLabelLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (target[index] == 0) {
      // Ignore label 0
      loss[index] = 0;
      counts[index] = 0;
    } else {
      bool c1 = target[index] > 0;
      bool c2 = input_data[index] >= 0;
      loss[index] = - input_data[index] * (c1 - c2) +
          log(1 + exp(input_data[index] - 2 * input_data[index] * c2));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int nthreads = bottom[0]->count();
  // const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = sigmoid_top_vec_[0]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultiLabelLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, target, loss_data,
      counts);
  Dtype loss;
  Dtype valid_count;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  caffe_gpu_asum(nthreads, counts, &valid_count);
  // top[0]->mutable_cpu_data()[0] = loss / num;
  top[0]->mutable_cpu_data()[0] = loss / valid_count;
}

// template <typename Dtype>
// __global__ void MultiLabelLossBackwardGPU(const int nthreads, 
//           const Dtype* sigmoid_output_data, const Dtype* target,
//           Dtype* bottom_diff, Dtype* counts) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//     if (target[index] == 0) {
//       // Ignore label 0
//       bottom_diff[index] = 0;
//       counts[index] = 0;
//     } else {
//       bottom_diff[index] = sigmoid_output_data[index] - (target[index] > 0);
//       counts[index] = 1;
//     }
//   }
// }

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // if (propagate_down[1]) {
  //   LOG(FATAL) << this->type()
  //              << " Layer cannot backpropagate to label inputs.";
  // }
  // if (propagate_down[0]) {
  //   // First, compute the diff
  //   const int nthreads = bottom[0]->count();
  //   // const int num = bottom[0]->num();
  //   const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
  //   const Dtype* target = bottom[1]->gpu_data();
  //   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  //   // Since this memory is never used for anything else,
  //   // we use it to avoid allocating new GPU memory.
  //   Dtype* counts = sigmoid_output_->mutable_gpu_diff();
  //   // NOLINT_NEXT_LINE(whitespace/operators)
  //   MultiLabelLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
  //       CAFFE_CUDA_NUM_THREADS>>>(nthreads, sigmoid_output_data, target,
  //       bottom_diff, counts);
  //   // Scale down gradient
  //   const Dtype loss_weight = top[0]->cpu_diff()[0];
  //   // caffe_scal(count, loss_weight / num, bottom_diff);
  //   // Dtype valid_count;
  //   // caffe_gpu_asum(nthreads, counts, &valid_count);
  //   // caffe_scal(count, loss_weight / valid_count, bottom_diff);
  //   // Did not find a good way to parallelize gradient normalization
  // }
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiLabelLossLayer);


}  // namespace caffe
