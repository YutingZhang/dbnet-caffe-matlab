function c = has_method_x( a, method_name )

has_method_field = ['has_',method_name];

if ismethod( a, has_method_field ) || ...
        isprop( a, has_method_field )
    c = a.(has_method_field);
else
    c = ismethod( a, method_name );
end

