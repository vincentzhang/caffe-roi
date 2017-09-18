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
void ROIConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(ROIConvolutionLayer);
REGISTER_LAYER_CLASS(ROIConvolution);

}  // namespace caffe

