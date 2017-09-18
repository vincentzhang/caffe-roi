#include "caffe/layers/roi_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
  //LOG(INFO) << "output_shape : " << vec_to_string(this->output_shape_);

}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
    const vector<Blob<Dtype>*>& top) {
  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  ROIConvolutionParameter roi_conv_param = this->layer_param_.roi_convolution_param();
  spatial_scale_ = roi_conv_param.spatial_scale();
  //LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void rois_convert_coordinate_cpu(Dtype* rois, const int num_rois, const Dtype spatial_scale){
	// Each rois = [batch_index x1 y1 x2 y2]
    for (int index = 0; index< num_rois; index++) {
	    int i = index*5;
	    int roi_batch_ind = rois[i+0];
	    int roi_start_w = round(rois[i+1] * spatial_scale);
	    int roi_start_h = round(rois[i+2] * spatial_scale);
	    int roi_end_w = round(rois[i+3] * spatial_scale);
	    int roi_end_h = round(rois[i+4] * spatial_scale);
	    // write to the out_rois
	    rois[0+i] = roi_batch_ind;
	    rois[1+i] = roi_start_w;
	    rois[2+i] = roi_start_h;
	    rois[3+i] = roi_end_w;
	    rois[4+i] = roi_end_h;
    }
}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  // conv: bottom[0]; rois: bottom[1]
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // Each ROI = [batch_index x1 y1 x2 y2]
  rois = bottom[1]->mutable_cpu_data();
  num_rois = bottom[1]->num();
  Dtype* top_data = top[0]->mutable_cpu_data();
  // Hold the output rois in the coordinate space of conv5
  // a function to convert the coordinate system into conv5
  rois_convert_coordinate_cpu(rois, num_rois, spatial_scale_);
  for (int n = 0; n < this->num_; ++n) {
    this->forward_cpu_gemm_roi(bottom_data + n * this->bottom_dim_, rois, num_rois, weight,
      top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ROIConvolutionLayer);
REGISTER_LAYER_CLASS(ROIConvolution);

}  // namespace caffe

