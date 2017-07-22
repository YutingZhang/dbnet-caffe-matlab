function notify( first_arg, varargin )

vi = varargin;
aux_arg = '';
if first_arg(1)~='-'
    first_arg = ['-note=' first_arg];
end
first_arg = first_arg(2:end);

sep = find(first_arg=='=',1);
if ~isempty(sep)
    aux_arg=first_arg(sep+1:end);
    first_arg=first_arg(1:sep-1);
end

global cur_task_title__
global cur_task_auto_notify__
global cur_task_stage_state__
global cur_task_details__
global cur_task_attachments__

if isempty(cur_task_details__)
    cur_task_details__ = '';
end

if isempty(cur_task_title__)
    cur_task_title__ = '[untitled task]';
end

if isempty(cur_task_auto_notify__)
    cur_task_auto_notify__ = 0;
end

if isempty(cur_task_attachments__)
    cur_task_attachments__ = {};
end

switch first_arg
    case {'note','send','done'} % note < send < done
        to_send = 1;
        if strcmp(first_arg, 'done') 
            if ~isempty(cur_task_stage_state__) && cur_task_stage_state__
                current_task_exception_handler( 'notify -done : job is not done' );
                to_send = 0;
            end
            if isempty(aux_arg)
                aux_arg = 'done';
            else
                aux_arg = [ aux_arg ': done' ];
            end
        end
        if to_send
            vi(2,:) = {' '};
            content_str = cat(2, vi{:});
            if ismember(first_arg, {'send','done'})
                ct = [content_str(1:end-1), sprintf('\n'), cur_task_details__];
            else
                ct = content_str(1:end-1);
            end
            send_update( cur_task_title__, aux_arg, ct, cur_task_attachments__ );
            if ismember(first_arg, {'send','done'})
                cur_task_details__ = '';    % clear details buffer
                cur_task_attachments__ = {};
            end
        end
    case {'auto'}
        if isempty(vi) || strcmp(vi{1},'on')
            cur_task_auto_notify__ = 1;
        elseif strcmp(vi{1},'off')
            cur_task_auto_notify__ = 0;
        else
            error( 'Unrecognized command line' );
        end
    case {'noauto', 'no-auto'}
        cur_task_auto_notify__ = 0;
    case {'add-attach'}
        cur_task_attachments__ = [cur_task_attachments__, vi];
    case {'clear-attach'}
        cur_task_attachments__ = {};
    otherwise
        error( 'Unrecognized option' );
end

end
