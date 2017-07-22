function show_bboxes( I, boxes, v, box_color, display_type, tag_loc, ...
    common_text, bold_factor )

if ~exist( 'v', 'var' )
    v = [];
end

if ~exist( 'box_color', 'var' )
    box_color = '';
end

if ~exist( 'display_type', 'var' ) || isempty( display_type )
    display_type = 'rank'; % 'score'
end

if ischar(display_type)
    if strcmp(display_type,'score')
        dis_fmt = '%.2f';
    end
elseif iscell( display_type )
    if strcmp(display_type{1},'score')
    	dis_fmt = display_type{2};
        display_type = display_type{1};
    end
end

if ~exist( 'tag_loc', 'var' ) || isempty( tag_loc )
    tag_loc = 1; % 'score'
end

if ~exist( 'common_text', 'var' ) || isempty( common_text )
    common_text = '';
end

if isempty(box_color)
    box_color = 'red';
end

if ~exist( 'bold_factor', 'var' ) || isempty(bold_factor)
    bold_factor = 1;
end

if isscalar( bold_factor )
    bold_factor_inner = bold_factor;
else
    bold_factor_inner = bold_factor(2);
    bold_factor = bold_factor(1);
end

boundary_color = '';
text_bg_color = '';
if iscell(box_color)
    if length(box_color)>=4
        text_bg_color = box_color{4};
    end
    if length(box_color)>=3
        boundary_color = box_color{3};
    end
    if length(box_color)>=2
        text_color = box_color{2};
    end
    box_color  = box_color{1};
else
    text_color = 'white';
end

if isempty( text_bg_color )
    if isempty( boundary_color )
        text_bg_color = box_color;
    else
        text_bg_color = boundary_color;
    end
end

if ~isempty(I)
    imshow(I);
end

boxes = double(boxes);

hold on

if ~isempty(v)
    
    if length(v) ~= size(boxes,1)
        error( 'The length of v (vector) and boxes (1st dim) should be consistent.' );
    end
    
    if isnumeric(v)
        [~,sortedIdx] = sort(v);
        v = v(sortedIdx);
        boxes = boxes(sortedIdx,:);
        % cm = jet(length(v));
    end
end

for i = 1:size(boxes,1)
    y1 = boxes(i,1); x1 = boxes(i,2);
    y2 = boxes(i,3); x2 = boxes(i,4);
    X  = [x1 x2 x2 x1];
    Y  = [y1 y1 y2 y2];
        
    if ~isempty( boundary_color )
        line([x1, x2, x2, x1, x1], [y1, y1, y2, y2 ,y1], [0 0 0 0 0], 'Color', ...
            boundary_color, 'LineWidth', 3*bold_factor );
    end

    patch(X,Y,box_color,  'FaceColor','none','EdgeColor', box_color,  'LineWidth',1.5*bold_factor_inner);
    
    if ~isempty(v)
        switch display_type
            case 'rank'
                tt = int2str(size(boxes,1)-i+1);
            case 'score'
                tt = sprintf(dis_fmt,v(i));
            case 'str'
                tt = v{1};
            case 'none'
                tt = '';
            otherwise
                error( 'Unrecognized display type' );
        end
    else
        tt = '';
    end
    tt = [common_text tt];
    
    if ~isempty( tt )
        
        switch tag_loc
            case 1
                text( x1, y1, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case 2
                text( x2, y1, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case 3
                text( x2, y2, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case 4
                text( x1, y2, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case -1
                text( x1, y1, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case -2
                text( x2, y1, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'top', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case -3
                text( x2, y2, tt, ...
                    'HorizontalAlignment', 'left', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor);
            case -4
                text( x1, y2, tt, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'bottom', ...
                    'Color', text_color, 'BackgroundColor', text_bg_color, ...
                    'Margin', 1*bold_factor, 'FontSize', 10*bold_factor );
            otherwise
                error( 'Unrecognized tag_loc' );
        end
    end
end

hold off

end
