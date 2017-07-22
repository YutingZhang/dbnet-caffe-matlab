classdef sliced_storage < handle
    properties (GetAccess=public, SetAccess=protected)
        output_filename
        output_slice_dir
        
        fuse_timeout = 120
        cat_dim
        num_slices
        cache_size
        
        pending_save = []
        
    end
    methods
        function obj = sliced_storage( output_filename, num_slices, cat_dim )
            
            if ~exist( 'cat_dim', 'var' ) || isempty( cat_dim )
                cat_dim = 1;
            end
            if ~strncmpr(output_filename, '.v7.mat', length('.v7.mat'))
                output_filename = [output_filename, '.v7.mat'];
            end
            obj.output_filename = output_filename;
            obj.num_slices      = num_slices;
            obj.cat_dim         = cat_dim;
            
            obj.cache_size = num_slices+2;
            
            obj.output_slice_dir = [ output_filename '.slice.d' ];
            
            is_fused = boolean( exist(obj.output_filename, 'file') );
            if ~is_fused
                mkdir_p( obj.output_slice_dir );
            end
            
        end
        
        function a = is_fused( obj )
            a = boolean( exist(obj.output_filename, 'file') );
        end
        
        function set_cache_size( obj, cache_size ) % empty for = num_slices+2
            if ~exist('cache_size','var') || isempty(cache_size)
                cache_size = obj.num_slices + 2;
            end
            obj.cache_size = cache_size;
        end
        
        function set_fuse_timeout( obj, timeout )
            if ~exist('timeout','var') || isempty(timeout)
                timeout = 120;
            end
            obj.fuse_timeout = timeout;
        end
        
        function set_slice( obj, slice_id, slice_content )
            obj.join();
            if obj.is_fused
                warning('Cannot set slice to a fused storage');
                return;
            end
            assert( slice_id>=1 && slice_id<=obj.num_slices && ...
                (isinteger(slice_id) || floor(slice_id)==slice_id), ...
                'slice_id must be an integer in [0,num_slices]' );
            slice_fn = fullfile( fullfile( obj.output_slice_dir, [int2str(slice_id) '.v7.mat'] ) );
            if exist( slice_fn, 'file' )
                warning( 'slice %d exists, overwritten', slice_id );
            end
            obj.pending_save = try_parfeval(1, @save7_single, 0, slice_fn, slice_content) ;
            % cached_file_func( @save7_single, slice_fn, obj.output_filename, obj.cache_size, ...
            %     slice_content );
        end
        
        function join( obj )
            if ~isempty(obj.pending_save)
                try_fetchOutputs( obj.pending_save );
                obj.pending_save = [];
            end
        end
        
        function is_success = try_fuse( obj )
            obj.join();
            if obj.is_fused
                is_success = 1;
                return;
            end
            is_success = 0;
            fusing_tag_fn = [obj.output_filename, '.fusing'];
            if exist( fusing_tag_fn, 'file' )
                fprintf( 'Someone is fusing the slices.\n' );
                ftinfo = dir(fusing_tag_fn);
                time_diff = secdiff(ftinfo.datenum,now);
                fprintf( 'Time diff is %s sec (%d). ', floor(time_diff), ceil(obj.fuse_timeout) );
                if time_diff<obj.fuse_timeout
                    fprintf( 'Wait. \n' );
                    return;
                else
                    fprintf( 'Continue. \n' );
                end
            end
            % check slice availablility
            t1 = tic_print( 'Check slice availability: ' );
            is_printed = 0;
            ut = tic; touch_file( fusing_tag_fn );
            touch_file( fusing_tag_fn );
            for k=1:obj.num_slices
                slice_fn = fullfile( fullfile( obj.output_slice_dir, ...
                    [int2str(k) '.v7.mat'] ) );
                if ~exist(slice_fn, 'file')
                    return;
                end
                if toc(ut)>max(obj.fuse_timeout/2,1)
                    ut = tic; touch_file( fusing_tag_fn );
                    fprintf('sliced_storage:try_fuse : check slice availablity : %d / %d\n', ...
                        k, obj.num_slices);
                    is_printed = 1;
                end
            end
            clear slice_fn
            toc_print(t1);
            % fuse slice
            Dslice = cell( obj.num_slices,    1 );
            for k=1:obj.num_slices
                slice_fn = fullfile( fullfile( obj.output_slice_dir, ...
                    [int2str(k) '.v7.mat'] ) );
                % Dslice{k} = cached_file_func( @load7_single, slice_fn, obj.output_filename, obj.cache_size );
                Dslice{k} = load7_single( slice_fn );
                if toc(ut)>max(obj.fuse_timeout/2,1)
                    ut = tic; touch_file( fusing_tag_fn );
                    touch_file( fusing_tag_fn );
                    fprintf('sliced_storage:try_fuse : load slice : %d / %d\n', ...
                        k, obj.num_slices);
                    is_printed = 1;
                end
            end
            D = cat( obj.cat_dim, Dslice{:} );
            clear Dslice
            
            if is_printed
                fprintf('sliced_storage:try_fuse : going to save fused results\n');
            end
            ut = tic; touch_file( fusing_tag_fn );
            save7_single(obj.output_filename, D);
            if is_printed || toc(ut)>max(obj.fuse_timeout/2,1)
                fprintf('sliced_storage:try_fuse : fused results saved\n');
                is_printed = 1;
            end
            
            is_success = 1;
            
            % clean-up
            if is_printed
                fprintf('sliced_storage:try_fuse : going to clean up slice cache\n');
            end
            remove_file( fusing_tag_fn );
            remove_file_recursively( obj.output_slice_dir );
            if is_printed
                fprintf('sliced_storage:try_fuse : slice cache was removed\n');
            end
        end
        
        function fuse( obj )
            while ~obj.try_fuse()
                pause( obj.fuse_timeout );
            end
            %cached_file_func( [], [], obj.output_filename, 0 );
        end
        
        function a = slice_exists( obj, slice_id )
            slice_fn = fullfile( fullfile( obj.output_slice_dir, ...
                [int2str(slice_id) '.v7.mat'] ) );
            a = boolean( exist(slice_fn, 'file') );
        end
        
        function delete( obj )
            obj.join();
            %cached_file_func( [], [], obj.output_filename, 0 );
        end

    end
end
