function [ X, varargout ] = caffeproto_replace_func_simplifyeltwise( varargin )

if strcmp(varargin{1},'extend')
    X = {  
        @(varargin) caffeproto_replace_func_simplifyeltwise( 'eltwise2power', varargin{:} ), ...
        { 'list', 'iterative', ...
            { 'list', 'iterative', ...
                @(varargin) caffeproto_replace_func_simplifyeltwise( 'merge-power', varargin{:} )
            }, ...
            @(varargin) caffeproto_replace_func_simplifyeltwise( 'empty-power', varargin{:} )
        }
        };
    return;
end

VAR_IN = varargin(2:end);

if ~isempty(VAR_IN) && strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
end

subS = VAR_IN{1};
subA = VAR_IN{2};

switch varargin{1}
    case 'eltwise2power'   
        if strcmp(VAR_IN{1},'adjacent')
            X = [0];
            return;
        end
        X = [];
        if ~strcmp( subS.type{1}, 'Eltwise' ), return; end
        if isscalar( subS.bottom )
            X.name = subS.name;
            X.type = {'Power'};
            X.bottom = subS.bottom;
            X.top    = subS.top;
            xc = 1;
            if strcmp( try_to_eval( 'subS.eltwise_param.operation.val()', 'SUM' ), 'SUM' ) ...
                    && ~isempty( try_to_eval( 'subS.eltwise_param.coeff', [] ) )
                xc = subS.eltwise_param.coeff;
            end
            X.power_param.scale = xc;
            if try_to_eval( 'subS.eltwise_param.learnable_coeff', 0 )
                X.power_param.learnable_scale = true;
                X.param(1) = struct('lr_mult',0,'decay_mult',1);
                X.param(2) = struct('lr_mult',0,'decay_mult',1);
                if isfield(subS,'param') && ~isempty(subS.param)
                    X.param(3) = canonical_blob_param( subS.param );
                else
                    X.param(3) = canonical_blob_param( struct() );
                end
            end
        end
    case 'merge-power'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0 1; 0 0];
            varargout = {1,2};
            return;
        end
        X = [];
        if ~strcmp( subS(1).type{1}, 'Power' ), return; end
        if ~strcmp( subS(2).type{1}, 'Power' ), return; end
        if isfield( subS(1), 'power_param' )
            p1 = subS(1).power_param;
        else
            p1 = struct();
        end
        if isfield( subS(2), 'power_param' )
            p2 = subS(2).power_param;
        else
            p2 = struct();
        end
        p = [];
        p1 = canonical_power_param(p1);
        p2 = canonical_power_param(p2);
        has_nonnegative_constraint = ...
            ( p1.nonnegative_shift || p1.nonnegative_scale || ...
            p2.nonnegative_shift || p2.nonnegative_scale );
        learnable_source = cell(1,3);
        % (shift + scale*x)^power
        if ~has_nonnegative_constraint && abs(p1.power-1)<eps && ~p1.learnable_power
            %  (shift2+scale2*(shift1 + scale1*x))^power2
            % =(shift2+scale2*shift1 + scale2*scale1*x)^power2
            p = struct();
            p.power = p2.power;
            p.scale = p1.scale*p2.scale;
            p.shift = p2.shift+p2.scale*p1.shift;
            if p2.learnable_power
                p.learnable_power = true;
                learnable_source{1} = [2 1];    % [source_id, blob_id]
            end
            if p1.learnable_scale || p2.learnable_scale
                p.learnable_scale = true;
                learnable_source{2} = [1 2; 2 2];    % [source_id, blob_id]
            end
            if p1.learnable_shift || p2.learnable_shift || p2.learnable_scale
                p.learnable_shift = true;
                learnable_source{3} = [1 3; 2 3; 2 2];    % [source_id, blob_id]
            end
        elseif ~has_nonnegative_constraint && abs(p1.shift)<eps && abs(p2.shift)<eps && ...
                ~p1.learnable_shift && ~p2.learnable_shift 
            %  (scale2*(scale1*x)^power1)^power2
            % =((scale2*scale1^power1)*x^power1)^power2
            % =(scale2*scale1^power1)^power2*(x^(power1*power2))
            % =A*x^(power1*power2)
            % =(log(A)/log(power1*power2)*x)^(power1*power2)
            p = struct();
            p.power = p1.power*p2.power;
            p.scale = p2.power*log((p2.scale*(p1.scale^p1.power)))/log(p.power);
            p.shift = 0;
            if p1.learnable_power || p2.learnable_power
                p.learnable_power = true;
                learnable_source{1} = [1 1; 2 1];    % [source_id, blob_id]
            end
            if p1.learnable_power || p2.learnable_power || ...
                    p1.learnable_scale || p2.learnable_scale
                p.learnable_scale = true;
                learnable_source{2} = [1 1; 2 1; 1 2; 2 2];    % [source_id, blob_id]
            end
        end
        if ~isempty(p)
            p = canonical_power_param(p);
            X = struct();
            X.name = subS(2).name;
            X.type = { 'Power' };
            X.bottom = subS(1).bottom;
            X.top    = subS(2).top;
            if abs(p.power-1)<eps
                p = rmfield(p,'power');
            end
            if abs(p.scale-1)<eps
                p = rmfield(p,'scale');
            end
            if abs(p.shift)<eps
                p = rmfield(p,'shift');
            end
            X.power_param = p;
            if isempty(fieldnames(p))
                X = rmfield(X,'power_param');
            end
            if p.learnable_power || p.learnable_scale || p.learnable_shift
                X.param = canonical_power_blob_param( [], X.power_param );
                if isfield(subS,'param')
                    pkparam = cell(1,2);
                    pkparam{1} = canonical_power_blob_param( subS(1), X.power_param );
                    pkparam{2} = canonical_power_blob_param( subS(2), X.power_param );
                    for k = 1:3
                        lr_mult_stack = 0;
                        for j = 1:size(learnable_source{k},1)
                            lr_mult_stack = [lr_mult_stack, ...
                                pkparam{learnable_source{k}(1,1)}(learnable_source{k}(1,2)).lr_mult];
                        end
                        X.param(k).lr_mult = max(lr_mult_stack);
                    end
                end
            else
                X.param = [];
            end
        end
    case 'empty-power'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0 1; 0 0];
            varargout = {1,[1 2]};
            return;
        end
        X = [];
        if ~strcmp( subS(2).type{1}, 'Power' ), return; end
        is_trivial = 0;
        if isfield( subS(2), 'power_param' )
            P = canonical_power_param( subS(2).power_param );
            if abs(P.power-1)<eps && abs(P.scale-1)<eps && abs(P.shift)<eps && ...
                    ~( P.learnable_power || P.learnable_scale || P.learnable_shift ),
                is_trivial = 1;
            end
        else
            is_trivial = 1;
        end
        if is_trivial
            [~, topidx]= ismember( subS(2).bottom{1}, subS(1).top );
            if size(subA(1).nextBlobs{topidx},2)<=1 
                % if do not affect other layers, then remove
                X = subS(1);
                X.top{topidx} = subS(2).top{1};
            else
                % use connector
                X = subS(1);
                X(2).name   = subS(2).name;
                X(2).type = {'Reshape'};
                X(2).bottom = subS(2).bottom;
                X(2).top    = subS(2).top;
                X(2).reshape_param.num_axes = 0;
            end
        end
    otherwise
        error( 'unrecognized branch' );
end

function p1 = canonical_power_param( p0 )

pdef.power = 1;
pdef.scale = 1;
pdef.shift = 0;
pdef.learnable_power = false;
pdef.learnable_scale = false;
pdef.learnable_shift = false;
pdef.nonnegative_scale = false;
pdef.nonnegative_shift = false;

p1 = xmerge_struct( 'always', 'always', pdef, p0 );

function p1 = canonical_blob_param( p0 )

pdef.lr_mult    = 1;
pdef.decay_mult = 1;
p1 = xmerge_struct( 'always', 'always', pdef, p0 );

function p1 = canonical_power_blob_param( p0, power_param )

p1 = repmat( struct(), 1, 3 );
for k = 1:3
    if length(p0)<1
        p1(k) = canonical_power_param( struct() );
    else
        p1(k) = canonical_power_param( p0(k) );
    end
end

if exist('power_param','var') && ~isempty(power_param)
    power_param = canonical_power_param( power_param );
    if ~power_param.learnable_power && ~power_param.learnable_scale && ...
            ~power_param.learnable_shift
        p1 = [];
    else
        if ~power_param.learnable_power
            p1(1).lr_mult = 0;
        end
        if ~power_param.learnable_scale
            p1(2).lr_mult = 0;
        end
        if ~power_param.learnable_shift
            p1(3).lr_mult = 0;
        end
    end
end
