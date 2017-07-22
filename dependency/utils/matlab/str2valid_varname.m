function a = str2valid_varname( s )

essential_list = ['a':'z','A':'Z'];
extened_list = ['0':'9','_'];

a = s;
a(~ismember(a,[essential_list,extened_list])) = '_';
if ~ismember(a(1), essential_list)
    a = ['VAR_',a];
end

