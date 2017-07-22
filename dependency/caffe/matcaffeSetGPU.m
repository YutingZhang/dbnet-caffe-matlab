function matcaffeSetGPU( GPU_ID )
% GPU_ID is 1-base

persistent CURRENT_GPU_ID

if isempty( GPU_ID ) || GPU_ID<0
    if ~isempty(CURRENT_GPU_ID)
        warning('You may be unable to switch back to GPU safely');
    end
    caffe.set_mode_cpu();
else
    if ~isempty(CURRENT_GPU_ID)
        assert( CURRENT_GPU_ID==GPU_ID, 'Cannot change GPU_ID' );
    else
        assert( GPU_ID<gpuDeviceCount, 'GPU_ID is out of range' );
        try
            gpuDevice( GPU_ID+1 );
            gpuArray(1);
        catch
            warning( 'set matlab gpuDevice failed' );
        end
        caffe.set_device( GPU_ID );
        CURRENT_GPU_ID = GPU_ID; 
        caffe.set_mode_gpu();
    end
end
