#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/im2col_roi.hpp"
#include "caffe/util/math_functions.hpp"
#include <stdio.h>

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
__device__ int num_rois_contain_pt(const int x, const int y, const Dtype* rois, const int num_rois, 
        const int kernel_h, const int kernel_w){
	int num_rois_out = 0;
	for (int i = 0; i< num_rois; i++){
		if ( is_pt_inside_roi<Dtype>(x,y,&rois[5*i],kernel_h,kernel_w) )
			num_rois_out++;
	}
    return num_rois_out;
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
__global__ void roi_im2col_gpu_kernel(const int n, const Dtype* data_im,
    const Dtype* rois, const int num_rois,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col; // output
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset; // input
    int data_im_valid = 0; // determine if a point is outside the boundary of the image
    // copy the pixel val into the output if it's inside the boundary and inside a roi
    // (w_im, h_im) is the (x,y) of the point
    // if no roi contain this point, the value will be zero
    int num_rois_contain = any_rois_contain_pt(w_offset, h_offset, rois, num_rois, kernel_h, kernel_w, 0) ? 1:0;
    //int num_rois_contain = num_rois_contain_pt(w_im, h_im, rois, num_rois, kernel_h, kernel_w);// multiply by the number of ROIs
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h; // index of the input
        int w_im = w_offset + j * dilation_w;
        data_im_valid = 0;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
            data_im_valid = num_rois_contain; // if data is not valid, it will default to be zero, otherwise set depending on the ROI
        *data_col_ptr = data_im_valid * data_im_ptr[i * dilation_h * width + j * dilation_w];
        data_col_ptr += height_col * width_col;
        // fill in output one column at a time
        // increment height_col * wid_col each time, which corresponds to one item in the
        // kernel; each column is k * k long.
      }
    }
  }
}

template <typename Dtype>
void roi_im2col_gpu(const Dtype* data_im, const Dtype* rois, const int
    num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  roi_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, rois, num_rois, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void roi_im2col_gpu<float>(const float* data_im, const float* rois, const int num_rois,
    const int channels, const int height, const int width, 
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void roi_im2col_gpu<double>(const double* data_im, const double* rois, const int num_rois,
    const int channels, const int height, const int width, 
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);
//template int num_rois_contain_pt<float>(const int x, const int y, const float* rois, 
 //       const int num_rois, const int kernel_h, const int kernel_w);
//template int num_rois_contain_pt<double>(const int x, const int y, const double* rois, 
//        const int num_rois, const int kernel_h, const int kernel_w);

template <typename Dtype>
__global__ void roi_col2im_gpu_kernel(const int n, const Dtype* data_col,
    const Dtype* rois, const int num_rois,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    //int num_rois_contain = num_rois_contain_pt(w_im, h_im, rois, num_rois, kernel_h, kernel_w);
    //if (num_rois_contain == 0) {
    //    // if no roi contain this pt, this pixel does not propagate diff
    //    data_im[index] = 0;
    //    continue;
    // }
    if (!any_rois_contain_pt(w_im, h_im, rois, num_rois, kernel_h, kernel_w, 0)) {
        // if no roi contain this pt, this pixel does not propagate diff
        data_im[index] = 0;
        continue;
    }
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index]; // accumulate the deltas wrt this input pixel
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void roi_col2im_gpu(const Dtype* data_col, const Dtype* rois, const int num_rois,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  roi_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, rois, num_rois, 
      height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void roi_col2im_gpu<float>(const float* data_col, const float* rois,
        const int num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void roi_col2im_gpu<double>(const double* data_col, const double* rois,
        const int num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
__global__ void gen_mask_gpu_kernel(const int n,
        const Dtype* rois, const int num_rois, 
        const int channel, const int height, const int width,
        Dtype* mask) {
  CUDA_KERNEL_LOOP(index, n) {
	int h = index*5;
	//int roi_batch_ind = rois[h+0];
	int roi_start_w = rois[h+1];
    int roi_start_h = rois[h+2];
    int roi_end_w = rois[h+3];
    int roi_end_h = rois[h+4];
    int ind = 0;//index for accessing mask
    for (int k=0; k < channel ; k++) {
        for(int i=roi_start_h; i<roi_end_h; i++) {// height
            for (int j=roi_start_w; j<roi_end_w; j++){ // width
                //ind =  i * width + j;
                ind = (k * height + i) * width + j;
                //LOG(INFO) << "k=" << k << " i=" << i << " j=" <<  j << " ind=" << ind;
                mask[ind] = (Dtype)1; // 1 inside , 0 outside
            }
        }
    }
  }
}

template <typename Dtype>
__global__ void check_mask(const int n, int* total, Dtype* mask){
  CUDA_KERNEL_LOOP(index, n) {
        *total += mask[index];
    }
}

template <typename Dtype>
void mask_output_gpu(Dtype* output, Dtype* mask, const Dtype* rois, const int num_rois, 
        const int channel, const int height, const int width) {
    //LOG(INFO) << "mask output gpu started";
    //Dtype total=0;
    //const int count = 8515584;
    //caffe_gpu_asum(count, mask, &total);
    //check_mask<Dtype><<<CAFFE_GET_BLOCKS(8515584), CAFFE_CUDA_NUM_THREADS>>>(8515584, &total, mask); 
    //CUDA_POST_KERNEL_CHECK;
    const int num_kernels = num_rois;
    // NOLINT_NEXT_LINE(whitespace/operators)
    gen_mask_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, rois, num_rois, channel, height, width, mask);
    CUDA_POST_KERNEL_CHECK;
    //total = 0;
    //caffe_gpu_asum(count, mask, &total);
    // At this point, mask has been generated.
    // mask * output = output
    //LOG(INFO) << "  number of items in the blob: " << channel << " " << height << " " << width;
    caffe_gpu_mul(channel*height*width, mask, output, output);
    //total = 0;
    //caffe_gpu_asum(count, mask, &total);
    
}
// Explicit instantiation
template void mask_output_gpu<float>(float* output, float* mask, const float* rois, const int num_rois, 
        const int channel, const int height, const int width);
template void mask_output_gpu<double>(double* output, double* mask, const double* rois, const int num_rois, 
        const int channel, const int height, const int width);

}  // namespace caffe
