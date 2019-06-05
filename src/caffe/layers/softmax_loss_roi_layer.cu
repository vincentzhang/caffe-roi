#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_roi_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ bool is_pt_inside_roi(const int x, const int y, const Dtype* roi, 
        const int kernel_h, const int kernel_w, const int mode){
	// check if the point (x,y) is inside the ROI
    // if mode == 1, padding is added: x can be kernel/2 away from the ROI boundary to be considered still inside an ROI
    // if mode == 0, no padding is added, the position of x is directly used to determine if inside an ROI
    if (mode == 1)
	    return (roi[1]-kernel_w/2 <= x) && (x <= roi[3]+kernel_w/2) && 
            (roi[2]-kernel_h/2 <= y) && (y <= roi[4]+kernel_h/2);
    else 
	    return (roi[1] <= x) && (x <= roi[3]) && 
            (roi[2] <= y) && (y <= roi[4]);
}

template <typename Dtype>
__device__ bool any_rois_contain_pt(const int x, const int y, const Dtype* rois, const int num_rois, 
        const int kernel_h, const int kernel_w, const int mode){
	for (int i = 0; i< num_rois; i++){
		if ( is_pt_inside_roi<Dtype>(x,y,&rois[5*i],kernel_h,kernel_w,mode) )
            return true;
	}
    return false;
}

template <typename Dtype>
__global__ void SoftmaxLossROIForwardGPU(const int nthreads,
          const Dtype* rois, const int num_rois, const int width, const int height,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    // if wants to ignore some labels or to include ROI
    int w_offset = s % width;
    int h_offset = (s / width) % height;
    Dtype num_rois_contain = any_rois_contain_pt(w_offset, h_offset, rois, num_rois, 0, 0, 0) ? 1:0;
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      loss[index] *= num_rois_contain; // if the pixel is outside ROI, exclude it from the loss
      counts[index] = 1 * num_rois_contain;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossROILayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  // rois
  const Dtype* rois = bottom[2]->gpu_data();
  const int num_rois = bottom[2]->num(); // number of ROIs
  const int dim = prob_.count() / outer_num_; // H * W * Channel
  // outer_num_ = batch
  // innter_num_ = H * W
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossROIForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, rois, num_rois, width_, height_,prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossROIBackwardGPU(const int nthreads, const Dtype* rois, 
          const int num_rois, const int width, const int height, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    // if wants to ignore some labels or to include ROI
    int w_offset = s % width;
    int h_offset = (s / width) % height;
    Dtype num_rois_contain = any_rois_contain_pt(w_offset, h_offset, rois, num_rois, 0, 0, 0) ? 1:0;

    if ((has_ignore_label_ && label_value == ignore_label_) || (num_rois_contain == 0)) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossROILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // rois
    const Dtype* rois = bottom[2]->gpu_data();
    const int num_rois = bottom[2]->num(); // number of ROIs
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossROIBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, rois, num_rois, width_, height_, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);

    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossROILayer);

}  // namespace caffe
