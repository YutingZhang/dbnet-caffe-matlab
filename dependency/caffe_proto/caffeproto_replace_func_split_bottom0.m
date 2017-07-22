function X = caffeproto_replace_func_split_bottom0( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

X = [];

subS = varargin{1};
if ~isfield(subS,'bottom0')
    return;
end
X = rmfield( subS, 'bottom0' );
if isempty(subS.bottom0)
    return;
end
E = ~strcmp( subS.bottom0, subS.bottom );
if ~any(E), return; end

B = unique( subS.bottom0(E) );

P = struct([]);
for k = 1:length(B)
    G = struct();
    merged_idxb = strcmp( B{k}, subS.bottom0 );
    merged_idx = find( merged_idxb );
    G.name = {[B{k} ':[m]']};
    G.type = {'Eltwise'};
    G.bottom = subS.bottom(merged_idxb);
    G.top  = B(k);
    G.eltwise_param.operation = pbEnum('SUM');
    P = cat_struct(2,P,G);
    X.bottom( merged_idx(2:end) ) = {[]};
    X.bottom( merged_idx(1) ) = subS.bottom0( merged_idx(1) );
end
X.bottom( cellfun(@isempty, X.bottom) ) = [];

X = cat_struct( 2, P ,X );


