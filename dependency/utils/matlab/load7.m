function varargout = load7( fn, varargin )
% load data saved by SAVE7
% load7( fn )
% load7( fn, varname1, varname2 )
% load7( var_struct )
% S = load7( ... )

assert( nargout<=1, 'At most one output' );

w_vnf_info=warning('query','MATLAB:load:variableNotFound');
warning('off','MATLAB:load:variableNotFound');
restoreWarning = onCleanup( @() warning(w_vnf_info.state,'MATLAB:load:variableNotFound') );

if isstruct(fn)
    S = fn;
else
    if isempty(varargin)
        S = load(fn);
    else
        S = load(fn,varargin{:},'sv_');
    end
end

if isfield(S,'sv_')

    sv = S.sv_;
    S = rmfield(S,'sv_');
    m = numel(sv);
    for k = m:-1:1 % must decode from the bottom
        vl = fieldnames(S);
        a = sv(k);
        unloaded_var = setdiff( a.var, vl );
        if ~isempty( unloaded_var )
            U = load( fn, unloaded_var{:} );
        end
        n = numel(a.var);
        V = cell(n,1);
        for j = 1:n
            subvar = a.var{j};
            ss = substruct('.',subvar);
            if ismember(subvar,unloaded_var)
                V{j} = subsref( U, ss );
            else
                V{j} = subsref( S, ss );
                S = rmfield(S,subvar);
            end
            V{j} = vec(V{j});
        end
        clear U
        V = cat(1, V{:} );
        V = reshape( V, a.size );
        S = subsasgn( S, substruct('.',a.name), V );
    end
end

if nargout==0
    vl = fieldnames(S);
    for k = 1:numel(vl)
        varname = vl{k};
        ss = substruct('.',varname);
        assignin( 'caller', varname, subsref( S, ss ) );
    end
else
    varargout = {S};
end

function a = vec(a)

a = reshape(a,numel(a),1);


