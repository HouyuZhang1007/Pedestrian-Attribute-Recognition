#include <vector>

#include "caffe/layers/sigmoid_weighted_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidWeightedCrossEntropyLossLayer<Dtype>::Backward_gpu(
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
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    const Dtype* weight = bottom[2]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    Dtype *tmp, *ones;
    CUDA_CHECK(cudaMalloc(&tmp, count * sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc(&ones, count * sizeof(Dtype)));
    caffe_gpu_set(count, (Dtype)1, ones);

    // diff: (1-p)
    caffe_gpu_sub(count, ones, target, bottom_diff);
    // diff: (1-p) * \hat{p}
    caffe_gpu_mul(count, bottom_diff, sigmoid_output_data, bottom_diff);
    // diff: 1/2 * (1-p) * \hat{p}
    caffe_gpu_scal(count, (Dtype)0.5, bottom_diff);
    // diff: 1/2(1-w) * (1-p) * \hat{p}
    caffe_gpu_sub(count, ones, weight, tmp);
    caffe_gpu_div(count, bottom_diff, tmp, bottom_diff);

    // tmp: 1-\hat{p}
    caffe_gpu_sub(count, ones, sigmoid_output_data, tmp);
    // tmp: p * (1-\hat{p})
    caffe_gpu_mul(count, tmp, target, tmp);
    // tmp: -1/2 * p * (1-\hat{p})
    caffe_gpu_scal(count, (Dtype)-0.5, tmp);
    // tmp: -1/2w * p * (1-\hat{p})
    caffe_gpu_div(count, tmp, weight, tmp);
    // diff: -(1/2w * p * (1-\hat{p}) - 1/2(1-w) * (1-p) * \hat{p})
    caffe_gpu_add(count, bottom_diff, tmp, bottom_diff);

    CUDA_CHECK(cudaFree(tmp));
    CUDA_CHECK(cudaFree(ones));

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidWeightedCrossEntropyLossLayer);


}  // namespace caffe
