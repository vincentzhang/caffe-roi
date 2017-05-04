#include <vector>

#include "caffe/layers/roi_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::rois_convert_coordinate(Dtype* rois, int num_rois, Dtype spatial_scale){
	// format of rois
	// Each ROI = [batch_index x1 y1 x2 y2]
	for ( int i = 0; i<num_rois; i++) {
	    int roi_batch_ind = rois[i*5+0];
	    int roi_start_w = round(rois[i*5+1] * spatial_scale);
	    int roi_start_h = round(rois[i*5+2] * spatial_scale);
	    int roi_end_w = round(rois[i*5+3] * spatial_scale);
	    int roi_end_h = round(rois[i*5+4] * spatial_scale);
	    // write to the out_rois
	    rois[0] = roi_batch_ind;
	    rois[1] = roi_start_w;
	    rois[2] = roi_start_h;
	    rois[3] = roi_end_w;
	    rois[4] = roi_end_h;
	    rois += 5;
	}
}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  // conv: bottom[0]; rois: bottom[1]
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // Each ROI = [batch_index x1 y1 x2 y2]
  //Dtype* bottom_rois = bottom[1]->mutable_gpu_data();
  rois = bottom[1]->mutable_gpu_data();
  num_rois = bottom[1]->num();
  // rois = bottom_rois;
  Dtype* top_data = top[0]->mutable_gpu_data();
  // Hold the output rois in the coordinate space of conv5
  // a function to convert the coordinate system into conv5
  this->rois_convert_coordinate(rois, num_rois, spatial_scale_);
  // Convert ROI coordinates into the space of this conv fmap
  for (int n = 0; n < this->num_; ++n) {
    this->forward_gpu_gemm_roi(bottom_data + n * this->bottom_dim_, rois,
        num_rois, weight, top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
  // 1. Add all rois together: for overlapped regions, take the summation of
  //    activations.
  // 2. When reshaping the input image, we slide the kernel across the image,
  //    set the regions whose center pixel does not belong to any ROI to zero; 
  //    use the conv map for other regions.
}


template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  // Each ROI = [batch_index x1 y1 x2 y2]
  //Dtype* bottom_rois = bottom[top.size()]->mutable_gpu_data();
  //num_rois = bottom[top.size()]->num();
  //rois = rois_convert_coordinate(bottom_rois, num_rois, spatial_scale_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    // Does not affect diff wrt the bias
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm_roi(bottom_data + n * this->bottom_dim_, rois, num_rois,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm_roi(top_diff + n * this->top_dim_, rois, num_rois, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIConvolutionLayer);

}  // namespace caffe
