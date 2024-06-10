#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "MULTI_LABEL_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  CHECK_EQ(bottom[0]->num_axes(), 2)
      << "Currently only supports fully connected layer outputs, "
      << "since gradient normalization becomes tricky for general "
      << "convolution layer outputs.";
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  int valid_count = 0;
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    // Ignore label 0
    if (target[i] == 0) { continue; }
    loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    ++valid_count;
  }
  // top[0]->mutable_cpu_data()[0] = loss / num;
  top[0]->mutable_cpu_data()[0] = loss / valid_count;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    // const int num = bottom[0]->num();
    const int dim = bottom[0]->shape(1);
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // int valid_count = 0;
    // Count non-ignored labels for each class separately.
    // Assume fully connected layer outputs.
    // Since this memory is never used for anything else,
    // we use it to avoid allocating new CPU memory.
    Dtype* class_count = sigmoid_output_->mutable_cpu_diff();
    caffe_set(dim, (Dtype)0., class_count);
    for (int i = 0; i < count; ++i) {
      if (target[i] == 0) {
        // Ignore label 0
        bottom_diff[i] = 0;
      } else {
        bottom_diff[i] = sigmoid_output_data[i] - (target[i] > 0);
        // ++valid_count;
        ++class_count[i % dim];
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // caffe_scal(count, loss_weight / num, bottom_diff);
    // caffe_scal(count, loss_weight / valid_count, bottom_diff);
    for (int i = 0; i < count; ++i) {
      if (target[i] != 0) {
        bottom_diff[i] *= loss_weight / class_count[i % dim];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MultiLabelLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelLossLayer);
REGISTER_LAYER_CLASS(MultiLabelLoss);

}  // namespace caffe
