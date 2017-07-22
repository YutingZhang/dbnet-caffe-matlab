function [input_names, input_dims] = caffeproto_defined_inputs( S )

if isfield( S, 'input' ) && ~isempty(S.input)
    input_names = [S.input];
    input_dims  = cell( size(input_names) );
    if nargout>=2
        for j = 1:numel(input_names)
            input_dims{j} = S.input_shape(j).dim;
        end
    end
else
    input_layer_idx = find( strcmp( 'Input', [S.layer.name] ) );
    input_names = cell(1,numel(input_layer_idx));
    input_dims  = cell(1,numel(input_layer_idx));
    for k = input_layer_idx
        input_names{k} = S.layer(k).top;
        if nargout>=2
            input_dims{k} = cell(size(input_names{k}));
            for j = 1:numel(input_names{k})
                input_dims{k}{j} = S.layer(k).shape(j).dim;
            end
        end
    end
    input_names = cat(2,{},input_names{:});
    input_dims  = cat(2,{},input_dims{:});
end
