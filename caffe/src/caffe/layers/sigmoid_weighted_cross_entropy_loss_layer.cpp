#include <vector>

#include "caffe/layers/sigmoid_weighted_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  CHECK_EQ(bottom[0]->count(), bottom[2]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to weight inputs.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype* weight = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    Dtype* tmp = new Dtype[count << 1];
    Dtype* tmp1 = tmp + count;

    // diff: 1/2
    caffe_set(count, (Dtype)0.5, bottom_diff);
    // diff: 1/2 * \hat{p}
    caffe_mul(count, bottom_diff, sigmoid_output_data, bottom_diff);
    // diff: 1/2 * (1-p) * \hat{p}
    caffe_set(count, (Dtype)1, tmp1);
    caffe_sub(count, tmp1, target, tmp);
    caffe_mul(count, bottom_diff, tmp, bottom_diff);
    // diff: 1/2(1-w) * (1-p) * \hat{p}
    caffe_sub(count, tmp1, weight, tmp);
    caffe_div(count, bottom_diff, tmp, bottom_diff);

    // tmp: 1-\hat{p}
    caffe_sub(count, tmp1, sigmoid_output_data, tmp);
    // tmp: p * (1-\hat{p})
    caffe_mul(count, tmp, target, tmp);
    // tmp: -1/2 * p * (1-\hat{p})
    caffe_set(count, (Dtype)-0.5, tmp1);
    caffe_mul(count, tmp, tmp1, tmp);
    // tmp: -1/2w * p * (1-\hat{p})
    caffe_div(count, tmp, weight, tmp);
    // diff: -(1/2w * p * (1-\hat{p}) - 1/2(1-w) * (1-p) * \hat{p})
    caffe_add(count, bottom_diff, tmp, bottom_diff);

    delete[] tmp;

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidWeightedCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidWeightedCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidWeightedCrossEntropyLoss);

}  // namespace caffe
