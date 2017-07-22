function E = try_parfeval( varargin )

persistent last_parpool_time
PARPOOL_INTERVAL = 30*60;

VARIN = varargin;
F = struct();
if isempty( varargin{1} ) || ...
        ( ischar(varargin{1}) && strcmp(varargin{1},'none') )
    F = [];
elseif isnumeric( varargin{1} )
    p=gcp('nocreate');
    if isempty(p)
        if isempty(last_parpool_time) || ...
                toc(last_parpool_time)>PARPOOL_INTERVAL
            last_parpool_time = tic;
            try
                p=parpool_strong( varargin{1} );
            catch
                warning( 'failed to create parpool' );
                F = [];
            end
        else
            F = [];
        end
    end
    VARIN{1} = p;
end

if isstruct(F)
    try
        F = parfeval( VARIN{:} );
    catch e
        F = [];
        warning( ['cannot start parfeval : ' e.identifier sprintf('\n') e.message] );
    end
end

if isa(varargin{2}, 'function_handle') 
    VARIN = varargin(2:end);
else
    VARIN = varargin;
end

E = struct( 'F', {F}, 'func', VARIN(1), 'numOutputs', VARIN(2), ...
    'ARGIN', {VARIN(3:end)} );



