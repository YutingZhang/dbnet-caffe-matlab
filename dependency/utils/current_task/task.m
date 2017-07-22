function varargout = task( varargin )

global cur_task_active__
global cur_task_title__
global cur_task_stage_state__
global cur_task_details__
global cur_task_attachments__

if isempty( varargin ) || varargin{1}(1)~='-'
    opts = '-reset';
    V0   = varargin;
else
    opts = varargin{1};
    V0   = varargin(2:end);
end

if ~ismember( opts, {'-new','-reset','-close', '-name'} )
    error( 'task : Unrecognized option : %s', opts );
end

if strcmp(opts,'-close')
    if ~isempty( V0 )
        error('task -close : There should not be any parameter.');
    end
    cur_task_title__  = '';
    cur_task_active__ = 0;
elseif strcmp(opts,'-name')
    if ~isempty( V0 )
        error('task -name : There should not be any parameter.');
    end
    if nargout>0
        varargout = {cur_task_title__};
    else
        fprintf('%s\n', cur_task_title__);
    end
else
    if strcmp(opts,'-new')
        cur_task_title__ = '';
    end

    if isempty(V0)
        if isempty(cur_task_title__)
            cur_task_title__ = '[untitled task]';
        end
    else
        V = V0;
        V(2,:) = {' '};
        cur_task_title__ = cat( 2, V{:} );
        cur_task_title__ = cur_task_title__(1:end-1);
    end
    cur_task_active__ = 1;
end

cur_task_stage_state__ = 0;
cur_task_details__ = '';
cur_task_attachments__ = {};

end
