function save7( fn, varargin )
% always save mat in version7 format. 
%  for large variables, automatically do slicing 
% save7( fn )
% save7( fn, varname1, varname2 )
% save7( fn, var_struct )

% get data to save
vl = {}; S = [];
if isempty(varargin)
    vl = evalin('caller','who');
elseif isscalar(varargin) && isstruct( varargin{1} )
    S = varargin{1};
else
    vl = varargin;
    assert( all(cellfun( @ischar, vl )), ...
        'all arguments should be char' );
end

is_struct_input = 0;
vli = vl;
if isempty(vl)
    is_struct_input = 1;
    if isempty(S)
        S = struct();
    end
else
    cm = cell( 1, numel(vl) );
    for k = 1:numel(vl)
        cm{k} = sprintf( '''%s'', {%s}, ', vl{k}, vl{k} );
    end
    cm = cat(2, cm{:} );
    cm = cm(1:(end-2));
    cm = sprintf( 'struct( %s )', cm );
    S = evalin( 'caller', cm );
end

% reserve a control variable
sv = struct( 'name', {}, 'size', {}, 'var', {} );
subsv = substruct('.','sv_');
vl = fieldnames( S );
if ismember('sv_',vl)
    sv(end+1).name = 'sv_';
    sv(end).size = size(S.sv_);
    sv_nn = get_unconflicted_name('sv_',vl);
    S = subsasgn( S, subsref( S, substruct('.',sv_nn) ), ...
        subsref( S, subsv ) );
    sv(end).var = {sv_nn};
    [~,sv_loc] = ismember( 'sv_', vl );
    vl{sv_loc} = sv_nn;
end
S.sv_ = [];

% split data if necessary
vlc = fieldnames(S);
sl = 2^31;
for k = 1:numel(vl)
    V = [];
    if is_struct_input
        vname = vl{k};
        V = subsref( S, substruct('.',vname) );
        vinfo = whos('V');
        vinfo.name = vname;
    else
        vname = vli{k};
        vinfo = evalin('caller', sprintf('whos(''%s'')', vname) );
    end
    if vinfo.bytes > sl
        if isempty(V)
            V = subsref( S, substruct('.',vname) );
        end
        S = rmfield( S, vname );
        % vlc = setdiff(vlc,vname);
        vsp = slicing_var( V, sl, vname );
        
        sv(end+1).name = vname;
        sv(end).size = vinfo.size;
        sv(end).var  = {};
        
        n = numel(vsp);
        r = 1; vsp = [0;vsp];
        for i = 1:n
            [subvname, r] = get_unconflicted_name( vname, vlc, r );
            vlc = [vlc;{subvname}];
            sv(end).var = [sv(end).var; subvname];
            v_i = vec(V((vsp(i)+1):vsp(i+1)));
            S = subsasgn( S, substruct( '.', subvname ), v_i );
            r = r+1;
        end
    end
end

% save data to file

S.sv_ = sv;
save_from_struct(fn,S,'-v7');


function sp = slicing_var( V, sl, vname )

sp = slicing_var_v2( V, sl, vname );

function sp = slicing_var_v2( V, sl, vname )


sp = [];
q = 0;
while ~isempty(V)

    vinfo = whos('V');
    vb = vinfo.bytes;
    sf = ceil(vb/sl);
    n = numel(V); 
    p = max(1,floor(n/sf));
    if p == n,
        ab = vb;
    else
        A = V(1:p);
        ainfo = whos('A');
        clear A
        ab = ainfo.bytes;
    end
    
    % exponential search
    while ab>sl
        assert( p>=1, 'Cannot find a small enough slice for "%s"', vname );
        p = max(1,floor(p*0.75));
        A = V(1:p);
        ainfo = whos('A');
        clear A
        ab = ainfo.bytes;
    end
    
    q = q+p;
    sp = [sp;q];
    V = V(p+1:end);
end


function sp = slicing_var_v1( V, sl, vname )

n = numel(V); 
es = zeros(n,1);
for j = 1:n
    v_j   = V(j);
    einfo = whos('v_j');
    es(j) = einfo.bytes;
    assert( es(j)<=sl, 'Cannot find small enough slice for "%s"', vname );
end
clear v_j

cs = cumsum(es);

sp = [];
j = 1;
while j<=n
    j1 = find(cs<=sl,1,'last');
    sp = [sp;j1];
    cs = cs - cs(j1);
    j = j1+1;
end


function [final_name, final_id] = get_unconflicted_name( ...
    name_prefix, existing_names, base_id )

if nargin<3
    base_id = 0;
end

if base_id>0 || ismember(name_prefix,existing_names)
    k = max(1,base_id);
    final_name = [ name_prefix '__' int2str(k) ];
    while ismember( final_name, existing_names )
        k = k + 1;
        final_name = [ name_prefix '__' int2str(k) ];
    end
else
    k = base_id;
    final_name = name_prefix;
end

final_id = k;

function a = vec(a)

a = reshape(a,numel(a),1);

