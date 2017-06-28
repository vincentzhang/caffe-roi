#include <vector>

#include "caffe/layers/roi_conv_layer.hpp"
#include <stdio.h>

namespace caffe {

template <typename Dtype>
__global__ void rois_convert_coordinate(Dtype* rois, const int num_rois, const Dtype spatial_scale){
	// format of rois
	// Each ROI = [batch_index x1 y1 x2 y2]
  CUDA_KERNEL_LOOP(index, num_rois) {
//	printf("ready for converting %d rois\n", num_rois);
	int i = index*5;
	//for ( int i = 0; i<num_rois; i++) {
	 //printf("processing rois[%d]: [%f,%f,%f,%f,%f]\n", index, rois[i],rois[i+1],rois[i+2], rois[i+3], rois[i+4]);
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
	   // printf("post rois[%d] = [%f,%f,%f,%f,%f]\n",index,rois[i],rois[i+1],rois[i+2], rois[i+3], rois[i+4]);
}
}

template <typename Dtype>
__global__ void show_rois(Dtype* rois, const int num_rois){
	// format of rois
	// Each ROI = [batch_index x1 y1 x2 y2]
	printf("Showing %d rois\n", num_rois);
	for ( int i = 0; i<num_rois; i++) {
	 printf("Showing rois[%d]: [%f,%f,%f,%f,%f]\n", i, rois[i*5],rois[i*5+1],rois[i*5+2], rois[i*5+3], rois[i*5+4]);
  }
}

template <typename Dtype>
void ROIConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LOG(INFO) <<"In Forward_gpu";
  const Dtype* weight = this->blobs_[0]->gpu_data();
  // conv: bottom[0]; rois: bottom[1]
  const Dtype* bottom_data = bottom[0]->gpu_data();
  // Each ROI = [batch_index x1 y1 x2 y2]
  rois = bottom[1]->mutable_gpu_data();
  rois_orig = bottom[1]->cpu_data();
  //LOG(INFO) << "rois address: " << rois; 
  num_rois = bottom[1]->num();
  Dtype* top_data = top[0]->mutable_gpu_data();
  // Hold the output rois in the coordinate space of conv5
  // a function to convert the coordinate system into conv5
  rois_convert_coordinate<Dtype><<<CAFFE_GET_BLOCKS(num_rois),
                             CAFFE_CUDA_NUM_THREADS>>>(rois, num_rois, spatial_scale_);
  CUDA_POST_KERNEL_CHECK;
  // Convert ROI coordinates into the space of this conv fmap
  for (int n = 0; n < this->num_; ++n) {
    this->forward_gpu_gemm_roi(bottom_data + n * this->bottom_dim_, rois,
        num_rois, weight, top_data + n * this->top_dim_);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->gpu_data();
      this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
    }
  }
  rois = bottom[1]->mutable_gpu_data();
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
