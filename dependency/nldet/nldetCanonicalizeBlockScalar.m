function a1 = nldetCanonicalizeBlockScalar( a0, a_def, HasCaffe )

if isempty(HasCaffe)
    a1 = [];
else
    if isempty(a0)
        a1 = a_def;
    else
        a1 = a0;
    end
    assert( ~isempty(a1), 'output should not be empty when using caffe net' );
end
