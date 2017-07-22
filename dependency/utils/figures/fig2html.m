function fig2html( figure_handle, file_path, PARAM )
% fig2html( figure_handle, fn )
% exportfig to html

if ~exist('PARAM','var') || isempty(PARAM)
    PARAM = struct();
end

Pdef = struct();
Pdef.figure_prefix = '$$.figures/$$-'; % $$ for filename without ext
Pdef.standalone_html = 1;
Pdef.html_head = '';
Pdef.header_html = '';
Pdef.footer_html = '';

PARAM = xmerge_struct(Pdef,PARAM);

% open output file
[pdir,pure_fn, ext] = fileparts( file_path );
if ~ismember(lower(ext),{'.html','.htm'})
    file_path = [file_path '.html'];
    pure_fn   = [pure_fn, ext];
end
pure_fig_prefix = strrep( PARAM.figure_prefix, '$$', pure_fn );
fig_prefix = [ pdir, '/', pure_fig_prefix ];
mkpdir_p( [fig_prefix '1'] );

fh = fopenh( file_path, 'w' );

% get figure size
figurePosition = get(figure_handle,'Position');
figW = figurePosition(3);
figH = figurePosition(4);

if PARAM.standalone_html
    PARAM.html_head = strtrim(PARAM.html_head);
    if ~isempty(PARAM.html_head)
        PARAM.html_head = sprintf('\n%s\n',PARAM.html_head);
    end
    fprintf( fh.id, '<!doctype html>\n<html>\n<head>%s</head>\n<body>\n', PARAM.html_head );
end

% begin of figure
fprintf( fh.id, ['<div class="zmf-figwrap" style="margin:0; padding:0; border: none; ' ... 
    'background: none; position: relative;">\n'] );
if ~isempty(PARAM.header_html)
    fprintf( fh.id, ['<div class="zmf-figheader" style="margin:0; padding: 0.5em 0 0.5em 0; border: none; ' ... 
        'background: none; position: relative;">\n'] );
    fprintf( fh.id, '%s\n', PARAM.header_html );
    fprintf( fh.id, '</div>\n');
end
fprintf( fh.id, ['<div class="zmf-figure" style="margin:0; padding:0; border: none; ' ... 
    'background: none; position: relative;">\n'] );
fprintf( fh.id, ['<div class="zmf-canvas" style="margin:0; padding:0; border: none; ' ... 
    'background: none; position: relative; width: %dpx; height: %dpx; overflow: hidden">\n'], figW, figH );

% create an invisible figure for dumping
tfig = figure('Visible','Off','Color','none', 'Units', 'pixels', 'Position', [1,1,figW,figH] );
tfig_cleanup = onCleanup( @() delete(tfig) );

% make figure saved as it is (very important)
tfig.PaperPositionMode = 'auto';
tfig.InvertHardcopy    = 'off';

% find all axes
A = findobj(figure_handle,'Type','axes');
A = A(end:-1:1);  % reverse order to draw
for k = 1:numel(A)
    
    % create new dummy axes
    clf(tfig);
    B = copyobj(A(k),tfig);
    cmap = colormap(A);
    colormap(B,cmap);
    % fix the axes size
    set(B,'Units','pixels');
    set(B,'Position', get(B,'Position') );
    set(B,'XLim',get(B,'XLim') );
    set(B,'YLim',get(B,'YLim') );
    set(B,'ZLim',get(B,'ZLim') );
    % begin axes
    fprintf( fh.id, ['<div class="zmf-axes" style="margin:0; padding:0; border: none; ' ... 
        'background: none; position: absolute; left: 0; top: 0; width: %dpx; height: %dpx;">\n'], figW, figH );
    
    [az,el] = view(B);
    is_2d = (az==0 && el==90);
    if is_2d
        % layer location and size
        positionAxes = get(B,'Position');
        widthAxes = positionAxes(3); heightB = positionAxes(4);
        leftAxes  = positionAxes(1)-1; bottomAxis = positionAxes(2)-1;
        if strcmp( B.XDir, 'normal' )
            xdirNormal = 1;
            leftData=B.XLim(1);
        else % reverse
            xdirNormal = 0;
            leftData=B.XLim(2);
        end
        if strcmp( B.YDir, 'normal' )
            ydirNormal = 1;
            bottomData=B.YLim(1);
        else % reverse
            ydirNormal = 0;
            bottomData=B.YLim(2);
        end
        widthData = B.XLim(2)-B.XLim(1);
        hidthData = B.YLim(2)-B.YLim(1);
        ratioW = widthAxes/widthData;
        ratioH = heightB/hidthData;
        % figure out layers in axes
        G = findobj(B,'Visible','on');
        if strcmp( G(1).Type, 'axes' )
            G = G(2:end);
        end
        G = G(end:-1:1); % remove the axes itself and reverse order
        for i=1:length(G)
            G(i).Visible = 'off';
        end
        typeG = arrayfun( @(a) a.Type, G, 'UniformOutput', 0 );
        splitPoints = strcmpi( typeG, 'image' );
        splitPoints(1:(end-1)) = splitPoints(1:(end-1)) | splitPoints(2:end);
        splitPoints(end) = true;
        splitPoints = find(splitPoints);
        layerObjNum = diff([0;splitPoints]);
        G = mat2cell( G, layerObjNum, 1);
        for j = 1:numel(G)
            if j==2
                set(B,'Visible','off');
            end
            
            base_layer_url  = sprintf('%s%d-%d',pure_fig_prefix,k,j);
            base_layer_path = sprintf('%s%d-%d',fig_prefix,k,j);
            Q = G{j};
            save_as_svg = 1;
            if isscalar(Q) && strcmpi(Q.Type, 'image')
                U = Q.UserData;
                if isstruct(U) && isfield(U,'url') && ischar(U.url)
                    full_bitmap_url = U.url;
                else
                    full_bitmap_url  = [base_layer_url  '.png'];
                    full_bitmap_path = [base_layer_path '.png'];
                    image_obj_to_imfile( Q, full_bitmap_path );
                end

                im_w = Q.XData(2)-Q.XData(1)+1; im_h = Q.YData(2)-Q.YData(1)+1;
                px_w = im_w * ratioW;
                px_h = im_h * ratioH;
                if xdirNormal, leftDelta = Q.XData(1)-0.5-leftData;
                else leftDelta = leftData-(Q.XData(2)+0.5); end
                if ydirNormal, bottomDelta = Q.YData(1)-0.5-bottomData;
                else bottomDelta = bottomData-(Q.YData(2)+0.5); end
                px_l = leftAxes   + leftDelta*ratioW;
                px_b = bottomAxis + bottomDelta*ratioH;
                
                fprintf( fh.id, ['<img class="zmf-image" style="margin:0; padding:0; border: none; ' ... 
                    'position: absolute; left: %gpx; bottom: %gpx; width: %gpx; height: %gpx; ' ...
                    'image-rendering: pixelated;" src="%s" alt="%s"/>\n'], ...
                    px_l, px_b, px_w, px_h, full_bitmap_url, 'zmf-image' );
                
                save_as_svg = 0;
            end
            if save_as_svg
                for i=1:length(Q)
                    Q(i).Visible = 'on';
                end
                full_svg_url  = [base_layer_url  '.svg'];
                full_svg_path = [base_layer_path '.svg'];
                print(tfig,'-dsvg',full_svg_path);
                for i=1:length(Q)
                    Q(i).Visible = 'off';
                end
                fprintf( fh.id, ['<img class="zmf-layer" style="margin:0; padding:0; border: none; ' ... 
                    'position: absolute; left: %dpx; top: %dpx; width: %dpx; height: %dpx;" src="%s" alt="%s"/>\n'], ...
                    0, 0, ...
                    figW, figH, full_svg_url, 'zmf-layer' );
            end
        end
    else
        base_layer_url  = fullfile(subdir,sprintf('layer-%d',k));
        base_layer_path = fullfile(tmp_folder,base_layer_url);        
        full_svg_url  = [base_layer_url  '.svg'];
        full_svg_path = [base_layer_path '.svg'];
        print(tfig,'-dsvg',full_svg_path);
        fprintf( fh.id, ['<img class="zmf-layer" style="margin:0; padding:0; border: none; ' ... 
            'position: absolute; left: 0; top: 0; width: %dpx; height: %dpx;" src="%s" alt="%s"/>\n'], ...
            figW, figH, full_svg_url, 'zmf-layer' );
    end
    
    % end axes
    fprintf( fh.id, '</div>\n' );
end

% end of figure
fprintf( fh.id, '</div>\n' );
fprintf( fh.id, '</div>\n' );
if ~isempty(PARAM.footer_html)
    fprintf( fh.id, ['<div class="zmf-figfooter" style="margin:0; padding: 0.5em 0 0.5em 0; border: none; ' ... 
        'background: none; position: relative;">\n'] );
    fprintf( fh.id, '%s\n', PARAM.footer_html );
    fprintf( fh.id, '</div>\n');
end
fprintf( fh.id, '</div>\n' );

if PARAM.standalone_html
    fprintf( fh.id, '</body>\n</html>\n' );
end


function varargout = image_obj_to_imfile( Q, fn )

[I,C,Alpha] = image_obj_to_array( Q );

alpha_args = {};
if ~all( Alpha(:)==1 )
    alpha_args = {'Alpha',Alpha};
end

if isempty(C)
    imwrite( I, fn, alpha_args{:} );
else
    imwrite( I, C, fn, alpha_args{:} );
end

if nargout>0
    varargout = {size(I)};
end

function [I,C,Alpha] = image_obj_to_array( Q )

AX = Q.Parent;
cdata=Q.CData;
if size(cdata,3)>1 % should = 3
    I = cdata;
    C = []; % rgb image, do not use colormap
else
    cmap = colormap(AX);
    if isinteger(cdata)
        assert( strcmp(Q.CDataMapping,'direct'), ...
            'it does not make sense to scale integer' );
    else
        switch Q.CDataMapping
            case 'direct'
                % do nothing
            case 'scaled'
                cdata = (cdata-AX.CLim(1))/(AX.CLim(2)-AX.CLim(1))*size(cmap,1);
            otherwise
                error( 'Unrecognized CDataMapping' );
        end
        cdata = uint8( min(max(0,floor(cdata)), size(cmap,1)-1) );
    end
    if all(cmap(:,1)==cmap(:,2)) && all(cmap(:,1)==cmap(:,3))
        % standard gray image
        C = cmap((double(cdata)+1),1);
    else
        I = cdata;
        C = cmap;
    end
end

amap  = alphamap(AX);
adata = Q.AlphaData;
need_amap = 1;
if isinteger(adata)
    assert( strcmp(Q.ADataMapping,'direct'), ...
        'it does not make sense to scale integer' );
else
    switch Q.AlphaDataMapping
        case 'none'
            need_amap = 0;
        case 'direct'
            % do nothing
        case 'scaled'
            adata = (adata-AX.ALim(1))/(AX.ALim(2)-AX.ALim(1))*size(amap,1);
        otherwise
            error( 'Unrecognized CDataMapping' );
    end
    if need_amap
        adata = uint8(min(max(0,floor(adata)), size(amap,1)-1));
    end
end
if need_amap
    Alpha = zeros(size(adata(:)));
    Alpha(:) = amap(double(adata(:))+1);
else
    Alpha = adata;
end

if strcmp(AX.XDir,'reverse')
    I(:,:,:) = I(:,end:-1:1,:);
    Alpha(:,:,:) = Alpha(:,end:-1:1,:);
end

if strcmp(AX.YDir,'normal')
    I(:,:,:) = I(end:-1:1,:,:);
    Alpha(:,:,:) = Alpha(end:-1:1,:,:);
end

