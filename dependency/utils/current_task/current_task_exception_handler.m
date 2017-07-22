function current_task_exception_handler( varargin )

if ~isempty( varargin )
    error_msg = sprintf( varargin{:} );
    fprintf( '%s\n', error_msg );
end

global cur_task_auto_notify__

if ~isempty(cur_task_auto_notify__) && cur_task_auto_notify__
    lb = sprintf('\n');
    fprintf( 'DEBUG: auto notify\n' );
    error_msg = [error_msg, sprintf('\n\nCall stack:\n\n')];
    D = dbstack;
    for j = 3:length(D)
        error_msg = [error_msg, ...
            '----------------------------', lb , ...
            any2str(D(j)), lb ];
    end
    notify('interrupted',error_msg);
end

block_input(0.5);

end
